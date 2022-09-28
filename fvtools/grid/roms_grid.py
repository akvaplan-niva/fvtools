# ==================================================================================================================
#                                Routines to access specific ROMS models
# =================================================================================================================
import numpy as np
import os
import sys
import pandas as pd
import time as time_mod

from datetime import datetime, timedelta
from functools import cached_property
from netCDF4 import Dataset, num2date
from pyproj import Proj

def get_roms_grid(mother, projection, offset = 7500):
    '''
    Returns a ROMS object for the mother model of question
    - we currently have readers for NorShelf (mother=NS) and the IMR- (HI-NK) and MET (MET-NK) operated NorKyst models.

    NorKyst is a 800 m grid spacing ROMS model intended for Norway-scale fjord studies. NorShelf is a data assimilated
    ROMS model often used for SAR operational forecasting and other outside-of-the-coast applications.

    Note that this ROMS object is not initialized yet - in the sense that it does not yet know anything about its grid
    coordinates. It must be looped through a "roms_nesting_fg.make_filelist" type of function so that it can figure
    out where to download a grid file when ROMS.load_grid() is called.
    '''
    if mother == 'D-NS':
        ROMS = NorShelf(True)

    elif mother == 'H-NS':
        ROMS = NorShelf(False)

    elif mother == 'HI-NK':
        ROMS = HINorKyst()

    elif mother == 'MET-NK':
        ROMS = METNorKyst()

    else:
        raise InputError(f'{mother=} is not a valid option. See docstring for more info.')

    ROMS.Proj = projection
    return ROMS

class ROMSbase:
    '''
    Containing methods we need when trying to couple a FVCOM and ROMS model
    '''
    def load_grid(self, xbounds=None, ybounds=None, offset=7500):
        '''
        Load grid from ROMS output file
        - xbounds: limit of the domain in x-direction
        - ybounds: limit of the domain in the y-direction
        - offset:  offset in x-y direction (to make sure to include all relevant ROMS points)
        '''
        # Add xbounds and ybounds to self
        # ----
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.offset  = offset

        # Load grid positions
        self.load_from_nc()
        self.get_x_y_z()

    @property
    def path(self):
        if not hasattr(self, '_path'):
            self._path = self.test_date(datetime.now())
        return self._path
    
    @path.setter
    def path(self, var):
        self._path = var

    def load_from_nc(self):
        # Open the ROMS file
        # ----
        ncdata = Dataset(self.path, 'r')

        # Grid positions and setup
        # ----
        load_position_fields = ['lon_rho', 'lat_rho', 'lat_u', 'lat_v', 'lon_u', 'lon_v',
                                'h', 'angle', 'Cs_r', 's_rho', 'hc']
        for load in load_position_fields:
            setattr(self, load, ncdata.variables.get(load)[:])
        self.h_rho = self.__dict__.pop('h') #ncdata.variables.get('h')[:]

        load_mask = ['u', 'v', 'rho']
        for load in load_mask:
            setattr(self, f'{load}_mask', ((ncdata.variables.get(f'mask_{load}')[:]-1)*(-1)).astype(bool))
        ncdata.close()

    def get_x_y_z(self):
        '''
        compute z at each S level for mean sealevel, project roms lon,lat to fvcom projection
        '''
        self.get_z_levels()
        self.x_rho, self.y_rho = self.Proj(self.lon_rho, self.lat_rho, inverse=False)
        self.x_u,   self.y_u   = self.Proj(self.lon_u,   self.lat_u,   inverse=False)
        self.x_v,   self.y_v   = self.Proj(self.lon_v,   self.lat_v,   inverse=False)

    # Maks to reduce need to download so much
    @cached_property
    def fv_rho_mask(self):
        return self.crop_rho()

    @cached_property
    def fv_u_mask(self):
        return self.crop_u()

    @cached_property
    def fv_v_mask(self):
        return self.crop_v()

    @property
    def cropped_x_psi(self):
        umask = np.logical_and(self.fv_u_mask[1:,:], self.fv_u_mask[:-1,:])
        vmask = np.logical_and(self.fv_v_mask[:,1:], self.fv_v_mask[:,:-1])
        psi_mask = np.logical_and(umask,vmask)
        return (self.x_u[1:,:] + self.x_u[:-1,:])[psi_mask]/2

    @property
    def cropped_y_psi(self):
        umask = np.logical_and(self.fv_u_mask[1:,:], self.fv_u_mask[:-1,:])
        vmask = np.logical_and(self.fv_v_mask[:,1:], self.fv_v_mask[:,:-1])
        psi_mask = np.logical_and(umask,vmask)
        return (self.y_v[:,1:] + self.y_v[:,:-1])[psi_mask]/2

    # Indices to be used when downloading data
    @cached_property
    def m_ri(self):
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return min(rho_i)

    @cached_property
    def x_ri(self):
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return max(rho_i)

    @cached_property
    def m_rj(self):
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return min(rho_j)

    @cached_property
    def x_rj(self):
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return max(rho_j)

    @cached_property
    def m_ui(self):
        u_i, u_j = np.where(self.fv_u_mask)
        return min(u_i)

    @cached_property
    def x_ui(self):
        u_i, u_j = np.where(self.fv_u_mask)
        return max(u_i)

    @cached_property
    def m_uj(self):
        u_i, u_j = np.where(self.fv_u_mask)
        return min(u_j)

    @cached_property
    def x_uj(self):
        u_i, u_j = np.where(self.fv_u_mask)
        return max(u_j)

    @cached_property
    def m_vi(self):
        v_i, v_j = np.where(self.fv_v_mask)
        return min(v_i)

    @cached_property
    def x_vi(self):
        v_i, v_j = np.where(self.fv_v_mask)
        return max(v_i)

    @cached_property
    def m_vj(self):
        v_i, v_j = np.where(self.fv_v_mask)
        return min(v_j)

    @cached_property
    def x_vj(self):
        v_i, v_j = np.where(self.fv_v_mask)
        return max(v_j)

    # Cropped versions of the masks to comply with the cropped download
    @cached_property
    def cropped_rho_mask(self):
        return self.fv_rho_mask[self.m_ri:self.x_ri+1, self.m_rj:self.x_rj+1]

    @cached_property
    def cropped_u_mask(self):
        return self.fv_u_mask[self.m_ui:self.x_ui+1, self.m_uj:self.x_uj+1]

    @cached_property
    def cropped_v_mask(self):
        return self.fv_v_mask[self.m_vi:self.x_vi+1, self.m_vj:self.x_vj+1]

    @property
    def Land_rho(self):
        return self.rho_mask[self.fv_rho_mask]

    @property
    def Land_u(self):
        return self.u_mask[self.fv_u_mask]

    @property
    def Land_v(self):
        return self.v_mask[self.fv_v_mask]

    @property
    def cropped_x_rho(self):
        return self.x_rho[self.fv_rho_mask]

    @property
    def cropped_y_rho(self):
        return self.y_rho[self.fv_rho_mask]

    @property
    def cropped_x_u(self):
        return self.x_u[self.fv_u_mask]

    @property
    def cropped_y_u(self):
        return self.y_u[self.fv_u_mask]

    @property
    def cropped_x_v(self):
        return self.x_v[self.fv_v_mask]

    @property
    def cropped_y_v(self):
        return self.y_v[self.fv_v_mask]

    @property
    def cropped_x_rho_grid(self):
        return self.x_rho[self.m_ri:(self.x_ri+1), self.m_rj:(self.x_rj+1)]

    @property
    def cropped_y_rho_grid(self):
        return self.y_rho[self.m_ri:(self.x_ri+1), self.m_rj:(self.x_rj+1)]

    def get_z_levels(self, z = None):
        '''
        compute z-rho
        '''
        self.S = np.tile(np.zeros((self.lon_rho.shape))[:,:,None], (1,1,len(self.Cs_r)))
        for i, (s, c) in enumerate(zip(self.s_rho, self.Cs_r)):
            self.S[:, :, i] = (self.hc*s + self.h_rho[:, :]*c)/(self.hc+self.h_rho)

        if z is not None:
            self.z_rho = (z+self.h_rho[:,:,None]) * self.S + z # I think so at least?
        else:
            self.z_rho = self.h_rho[:,:,None] * self.S

        # Make sure to update the u- and v- points in the same go
        self.h_u = (self.h_rho[:,1:]   + self.h_rho[:,:-1])/2
        self.z_u = (self.z_rho[:,1:,:] + self.z_rho[:,:-1,:])/2
        self.h_v = (self.h_rho[1:,:]   + self.h_rho[:-1,:])/2
        self.z_v = (self.z_rho[1:,:,:] + self.z_rho[:-1,:,:])/2

    def crop_rho(self):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_rho >= np.min(self.xbounds)-self.offset, self.x_rho <= np.max(self.xbounds)+self.offset)
        ind2 = np.logical_and(self.y_rho >= np.min(self.ybounds)-self.offset, self.y_rho <= np.max(self.ybounds)+self.offset)
        return np.logical_and(ind1, ind2)

    def crop_u(self):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_u >= np.min(self.xbounds)-self.offset, self.x_u <= np.max(self.xbounds)+self.offset)
        ind2 = np.logical_and(self.y_u >= np.min(self.ybounds)-self.offset, self.y_u <= np.max(self.ybounds)+self.offset)
        return np.logical_and(ind1, ind2)

    def crop_v(self):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_v >= np.min(self.xbounds)-self.offset, self.x_v <= np.max(self.xbounds)+self.offset)
        ind2 = np.logical_and(self.y_v >= np.min(self.ybounds)-self.offset, self.y_v <= np.max(self.ybounds)+self.offset)
        return np.logical_and(ind1, ind2)

# Each ROMS setup will have its own quirks w/respect to finding data, these quirks should be adressed in these subclasses
class HINorKyst(ROMSbase):
    '''
    Routines to check if HI-NorKyst data is available, inherits grid-methods from ROMSbase
    '''
    @cached_property
    def all_local_norkyst_files(self):
        '''
        property holding all local ncfiles
        '''
        self.folders     = ['/cluster/shared/NS9067K/apn_backup/ROMS/NK800_2016-2017',\
                            '/cluster/shared/NS9067K/apn_backup/ROMS/NK800_2017',\
                            '/cluster/shared/NS9067K/apn_backup/ROMS/NK800_2018',\
                            '/cluster/shared/NS9067K/apn_backup/ROMS/NK800_2019',\
                            '/cluster/shared/NS9067K/apn_backup/ROMS/NK800_20194',\
                            '/cluster/shared/NS9067K/apn_backup/ROMS/NK800_2020']
        self._bottom_folders()
        return self._list_ncfiles()

    def test_day(self, date):
        '''
        See if the local file exists that day, and has enough data
        '''
        file = self.get_norkyst_local(date)
        self.test_ncfile(file)
        self.path = file
        return file

    def test_ncfile(self, file):
        try:
            with Dataset(file, 'r') as d:
                if len(d.variables['ocean_time'][:])<24:
                    raise NoAvailableData(f'{file} does not have a complete timeseries')
        except:
            raise NoAvailableData

    def get_norkyst_local(self, date):
        '''
        Looks for NorKyst data in the predefined folders.
        '''
        return self._connect_date_to_file(self.all_local_norkyst_files, date)

    def _connect_date_to_file(self, all_ncfiles, date):
        '''
        check which date to start with
        '''
        # Identify the files using their names (ie. not a filelist approach?)
        # ----
        year        = str(date.year)
        month       = '{:02d}'.format(date.month)
        day         = '{:02d}'.format(date.day)

        files       = [files for files in all_ncfiles if year+month+day in files]

        # I want the file that starts the same date as my date
        # ----
        for f in files:
            if f'{year}{month}{day}' in f.split('_')[-1].split('-')[0]:
                read_file = f
                break

        return read_file

    def _bottom_folders(self):
        '''
        Returns the folders on the bottom of the pyramid (hence the name)
        mandatory:
        folders   - parent folder(s) to cycle through
        '''
        # ----
        dirs = []
        for folder in self.folders:
            dirs.extend([x[0] for x in os.walk(folder)])

        # remove folders that are not at the top of the tree
        # ----
        leaf_branch = []
        for dr in dirs:
            if dr[-1]=='/':
                continue
            else:
                # This string is at the end of the branch, thus this is where the data is stored
                # ----
                leaf_branch.append(dr)
        self.subfolders = leaf_branch

    def _list_ncfiles(self):
        '''
        returns list of all files in directories (or in one single directory)
        '''
        ncfiles = []
        for dr in self.subfolders:
            stuff   = os.listdir(dr)
            ncfiles.extend([dr+'/'+fil for fil in stuff if '.nc' in fil])
        return ncfiles

class METNorKyst(ROMSbase):
    '''
    Routines to check if MET-NorKyst data is available
    '''
    def test_day(self, date):
        '''
        Check if the file exists that day, and that it has enough data
        '''
        file = self.get_norkyst_url(date)
        self.test_ncfile(file)
        self.path = file
        return file

    def test_ncfile(self, file):
        try:
            with Dataset(file, 'r') as d:
                if len(d.variables['ocean_time'][:])<24:
                    raise NoAvailableData(f'{file} does not have a complete timeseries')
        except:
            raise NoAvailableData

    def get_norkyst_url(self, date):
        '''
        Give it a date, and you will get the corresponding url in return
        '''
        https       = 'https://thredds.met.no/thredds/dodsC/fou-hi/new_norkyst800m/his/ocean_his.an.'
        year        = str(date.year)
        month       = '{:02d}'.format(date.month)
        day         = '{:02d}'.format(date.day)
        return f'{https}{year}{month}{day}.nc'

class NorShelf(ROMSbase):
    '''
    for accessing NorShelf data
    '''
    def __init__(self, avg):
        self.avg  = avg
        self.path = 'https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_avg_an_20210531T00Z.nc'
        self.min_len = 24
        if self.avg:
            self.min_len = 1

    def test_day(self, date):
        '''
        Load norshelf path.
        - norshelf is a forecast model, so we get more dates there than with norkyst.
        '''
        file = self.get_norshelf_day_url(date)

        # Test if the data is good
        # ----
        forecast_nr = 0
        while True:
            try:
                try:
                   d    = Dataset(file, 'r')
                except:
                    raise NoAvailableData

                if len(d.variables['ocean_time'][:])<self.min_len:  # Check to see if empty
                    raise NoAvailableData
                else:
                    break

            except NoAvailableData:
                forecast_nr +=1
                file = self.get_norshelf_fc_url(date-timedelta(days=forecast_nr))

            if forecast_nr > 3:
                raise NoAvailableData

        d.close()
        return file

    def get_norshelf_fc_url(self, date):
        '''
        Give it a date, and you will get the corresponding url in return
        '''
        date = date-timedelta(days=1) # minus since this is a forecast, see *_dat_* file for best data
        if self.avg:
            https = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_avg_fc_"
        else:
            https = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_qck_fc_"
        return https + "{0.year}{0.month:02}{0.day:02}".format(date) + "T00Z.nc"

    def get_norshelf_day_url(self, date):
        '''
        Give it a date, and you will get the corresponding url in return
        '''
        if self.avg:
            https = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_avg_an_"
        else:
            https = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_qck_an_"
        return https+ "{0.year}{0.month:02}{0.day:02}".format(date) + "T00Z.nc"

class RomsDownloader:
    path_old = 'loremipsum'
    verbose = False
    def read_timestep(self, variables = ['salt', 'temp', 'zeta', 'u', 'v']):
        '''
        Reads a timestep
        '''
        unavailable = True
        while unavailable:
            try:
                self._check_nc_handle()
                self._load_roms_data(variables)
                unavailable = False
            except:
                print("\n--------------------\n"\
                    + "The data is unavailable at the moment.\n"\
                    + "We a minute and try again.\n"\
                    + "--------------------\n")
                time_mod.sleep(60)

    def _check_nc_handle(self):
        '''
        Access (and close) netCDF handle
        '''
        if self.path_here != self.path_old:
            if self.verbose: print(f'\nReading data from: {self.path_here}')
            self.path_old = self.path_here
            try:
                self.nc.close()
            except:
                pass
            self.nc = Dataset(self.path_here, 'r')

    def _load_roms_data(self, variables):
        '''
        Dumps roms data from the netcdf file and prepare for interpolation
        - how we access the ROMS data may depend on the interpolation method?
        '''
        supported = ['salt', 'temp', 'zeta', 'u', 'v']
        for var in variables:
            if var not in supported:
                raise InputError(f'The variable: "{var}" is not supported by RomsDownloader')

        if 'salt' in variables:
            self.salt = self.nc['salt'][self.index_here, :, self.ROMS.m_ri:(self.ROMS.x_ri+1), self.ROMS.m_rj:(self.ROMS.x_rj+1)]

        if 'temp' in variables:
            self.temp = self.nc['temp'][self.index_here, :, self.ROMS.m_ri:(self.ROMS.x_ri+1), self.ROMS.m_rj:(self.ROMS.x_rj+1)]

        if 'zeta' in variables:
            self.zeta = self.nc['zeta'][self.index_here, self.ROMS.m_ri:(self.ROMS.x_ri+1), self.ROMS.m_rj:(self.ROMS.x_rj+1)]

        if 'u' in variables:
            self.u    = self.nc['u'][self.index_here, :, self.ROMS.m_ui:(self.ROMS.x_ui+1), self.ROMS.m_uj:(self.ROMS.x_uj+1)]

        if 'v' in variables:
            self.v    = self.nc['v'][self.index_here, :, self.ROMS.m_vi:(self.ROMS.x_vi+1), self.ROMS.m_vj:(self.ROMS.x_vj+1)]

class NoAvailableData(Exception): pass
class InputError(Exception): pass