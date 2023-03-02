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
from dataclasses import dataclass

def get_roms_grid(mother, projection = None, offset = 7500):
    '''
    Returns a ROMS object for the mother model of question
    - we currently have readers for NorShelf (mother=NS) and the IMR- (HI-NK) and MET (MET-NK) operated NorKyst models.

    NorKyst is a 800 m grid spacing ROMS model intended for Norway-scale fjord studies. NorShelf is a data assimilated
    ROMS model often used for SAR operational forecasting and other outside-of-the-coast applications.
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

    if projection is not None:
        ROMS.Proj = projection

    return ROMS

# Classes that lets us access ROMS data from thredds server, grid information and routines to crop a ROMS domain to cover an FVCOM domain
# ----
class ROMSdepths:
    @property
    def zeta(self):
        '''
        z is the sea surface elevation interpolated from ROMS. Its added to h_rho later, and will be used
        when computing the vertical interpolation coefficients
        '''
        if not hasattr(self, '_zeta'):
            self._zeta = np.zeros(self.lon_rho.shape)
        return self._zeta

    @zeta.setter
    def zeta(self, var):
        '''SSE at rho points'''
        if var.shape != self.lon_rho.shape:
            raise ValueError(f'zeta and h_rho needs to have the same shape')
        self._zeta = var
    
    @property
    def h_rho(self):
        '''depth at rho points'''
        return self._h_rho+self.zeta
    
    @h_rho.setter
    def h_rho(self, var):
        self._h_rho = var

    @property
    def h_u(self):
        return (self.h_rho[:, 1:]   + self.h_rho[:, :-1])/2
    
    @property
    def h_v(self):
        return (self.h_rho[1:, :]   + self.h_rho[:-1, :])/2

    @property
    def S_rho(self):
        '''Depth where scalar-variables are stored'''
        S_rho = np.tile(np.zeros((self.lon_rho.shape))[:,:, None], (1,1,len(self.Cs_r)))
        for i, (s, c) in enumerate(zip(self.s_rho, self.Cs_r)):
            S_rho[:, :, i] = (self.hc*s + self.h_rho[:, :]*c)/(self.hc+self.h_rho)
        return S_rho

    @property
    def S_w(self):
        '''
        Depth at interface between sigma layers (as well as top/bottom). Stored in the same way as internally in ROMS
        '''
        S_w   = np.tile(np.zeros((self.lon_rho.shape))[None, :, :], (len(self.Cs_w),1,1))
        for i, (s, c) in enumerate(zip(self.s_w, self.Cs_w)):
            S_w[i, :, :] = (self.hc*s + self.h_rho[:, :]*c)/(self.hc+self.h_rho)
        return S_w

    @property
    def zw_rho(self):
        '''depth at sigma-interfaces'''
        return self.h_rho[None, :, :] * self.S_w

    @property
    def z_rho(self):
        '''depth at centre of sigma layers'''
        return self.h_rho[:,:,None] * self.S_rho

    @property
    def z_u(self):
        return(self.z_rho[:, 1:, :] + self.z_rho[:, :-1, :])/2

    @property
    def zw_u(self):
        return (self.zw_rho[:, :, 1:] + self.zw_rho[:, :, :-1])/2

    @property
    def z_v(self):
        return (self.z_rho[1:, :, :] + self.z_rho[:-1, :, :])/2

    @property
    def zw_v(self):
        return (self.zw_rho[:, 1:, :] + self.zw_rho[:, :-1, :])/2

    @cached_property
    def dsigma_u(self):
        '''sigma layer thickness at u-points'''
        return np.diff(self.zw_u, axis = 0)/self.h_u[None, :, :]

    @cached_property
    def dsigma_v(self):
        return np.diff(self.zw_v, axis = 0)/self.h_v[None, :, :]

# We always deal with a cropped verison of the ROMS grid in the nesting- restart- and movie routines.
# These classes are used to crop the roms domain to cover xbounds, ybounds (xy-coordinates), and provides metrics (m_ri, x_ri) etc,
# to be used when we crop data on-the-fly while downloading.
class CropRho:
    def crop_rho(self):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_rho >= np.min(self.xbounds)-self.offset, self.x_rho <= np.max(self.xbounds)+self.offset)
        ind2 = np.logical_and(self.y_rho >= np.min(self.ybounds)-self.offset, self.y_rho <= np.max(self.ybounds)+self.offset)
        return np.logical_and(ind1, ind2)

    @property
    def fv_rho_mask(self):
        if not hasattr(self, '_fv_rho_mask'):
            self._fv_rho_mask = self.crop_rho()
        return self._fv_rho_mask

    @fv_rho_mask.setter
    def fv_rho_mask(self, var):
        self._fv_rho_mask = var

    @cached_property
    def cropped_rho_mask(self):
        '''Cropped version of the rho-mask that we use when processing the data downloaded from thredds'''
        return self.fv_rho_mask[self.m_ri:self.x_ri+1, self.m_rj:self.x_rj+1]

    @cached_property
    def m_ri(self):
        '''min i-index for rho-points'''
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return min(rho_i)

    @cached_property
    def x_ri(self):
        '''max i-index for rho points'''
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return max(rho_i)

    @cached_property
    def m_rj(self):
        '''min j-index for rho-points'''
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return min(rho_j)

    @cached_property
    def x_rj(self):
        '''max j-index for rho points'''
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return max(rho_j)

class CropU:
    def crop_u(self):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_u >= np.min(self.xbounds)-self.offset, self.x_u <= np.max(self.xbounds)+self.offset)
        ind2 = np.logical_and(self.y_u >= np.min(self.ybounds)-self.offset, self.y_u <= np.max(self.ybounds)+self.offset)
        return np.logical_and(ind1, ind2)

    @property
    def fv_u_mask(self):
        if not hasattr(self, '_fv_u_mask'):
            self._fv_u_mask = self.crop_u()
        return self._fv_u_mask

    @fv_u_mask.setter
    def fv_u_mask(self, var):
        self._fv_u_mask = var

    @cached_property
    def cropped_u_mask(self):
        return self.fv_u_mask[self.m_ui:self.x_ui+1, self.m_uj:self.x_uj+1]

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

class CropV:
    def crop_v(self):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_v >= np.min(self.xbounds)-self.offset, self.x_v <= np.max(self.xbounds)+self.offset)
        ind2 = np.logical_and(self.y_v >= np.min(self.ybounds)-self.offset, self.y_v <= np.max(self.ybounds)+self.offset)
        return np.logical_and(ind1, ind2)

    @property
    def fv_v_mask(self):
        if not hasattr(self, '_fv_v_mask'):
            self._fv_v_mask = self.crop_v()
        return self._fv_v_mask

    @fv_v_mask.setter
    def fv_v_mask(self, var):
        self._fv_v_mask = var

    @cached_property
    def cropped_v_mask(self):
        return self.fv_v_mask[self.m_vi:self.x_vi+1, self.m_vj:self.x_vj+1]

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

class ROMSCropper(CropU, CropV, CropRho):
    '''
    Class that uses the crop-properties to fit arrays to the desired subdomain
    '''
    @property
    def cropped_dsigma_v(self):
        '''
        sigma layer thickness of each ROMS layer
        '''
        return self.dsigma_v[:, self.m_vi:self.x_vi+1, self.m_vj:self.x_vj+1]
    
    @property
    def cropped_dsigma_u(self):
        '''
        sigma layer thickness of each ROMS layer
        '''
        return self.dsigma_u[:, self.m_ui:self.x_ui+1, self.m_uj:self.x_uj+1]

    # Cropped roms coordinates as (n,) arrays.
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

    # Land-mask
    @property
    def Land_rho(self):
        '''rho mask cropped to fit with the mesh we're interpolating to'''
        return self.rho_mask[self.fv_rho_mask]

    @property
    def Land_u(self):
        '''u mask cropped to fit with the mesh we're interpolating to'''
        return self.u_mask[self.fv_u_mask]

    @property
    def Land_v(self):
        '''v mask cropped to fit with the mesh we're interpolating to'''
        return self.v_mask[self.fv_v_mask]

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

    @property
    def cropped_x_rho_grid(self):
        '''Exclusively used in the ROMS movie maker'''
        return self.x_rho[self.m_ri:(self.x_ri+1), self.m_rj:(self.x_rj+1)]

    @property
    def cropped_y_rho_grid(self):
        '''Exclusively used in the ROMS movie maker'''
        return self.y_rho[self.m_ri:(self.x_ri+1), self.m_rj:(self.x_rj+1)]

class ROMSbase(ROMSdepths, ROMSCropper):
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
        self.load_grid_from_nc()
        self.get_x_y_z()

    @property
    def path(self):
        if not hasattr(self, '_path'):
            try:
                self._path = self.test_day(datetime.now())
            except:
                self._path = self.test_day(datetime.now()-timedelta(days=1))
        return self._path
    
    @path.setter
    def path(self, var):
        self._path = var

    def load_grid_from_nc(self):
        '''
        Load the position data we need to get going
        - positions (of rho, u and v points)
        - depth (at rho points)
        - stretching functions (for rho and w points)
        - angle between XI-axis and east
        '''
        ncdata = Dataset(self.path, 'r')
        load_position_fields = ['lon_rho', 'lat_rho', 'lat_u', 'lat_v', 'lon_u', 'lon_v',
                                'h', 'angle', 'Cs_r', 's_rho', 'Cs_w', 's_w', 'hc']
        for load in load_position_fields:
            setattr(self, load, ncdata.variables.get(load)[:])
        self.h_rho = self.__dict__.pop('h')

        load_mask = ['u', 'v', 'rho']
        for load in load_mask:
            setattr(self, f'{load}_mask', ((ncdata.variables.get(f'mask_{load}')[:]-1)*(-1)).astype(bool))
        ncdata.close()

    def get_x_y_z(self):
        '''
        compute z at each S level for mean sealevel, project roms lon,lat to desired projection
        '''
        self.x_rho, self.y_rho = self.Proj(self.lon_rho, self.lat_rho, inverse=False)
        self.x_u,   self.y_u   = self.Proj(self.lon_u,   self.lat_u,   inverse=False)
        self.x_v,   self.y_v   = self.Proj(self.lon_v,   self.lat_v,   inverse=False)

# Each ROMS setup will have its own quirks w/respect to finding data, these quirks should be adressed in these subclasses
# ----
class HINorKyst(ROMSbase):
    '''
    Routines to check if HI-NorKyst data is available, inherits grid-methods from ROMSbase
    '''
    def __str__(self):
        return 'Havforskningsinstituttet NorKyst simulations'

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
        year  = str(date.year)
        month = '{:02d}'.format(date.month)
        day   = '{:02d}'.format(date.day)

        files = [files for files in all_ncfiles if year+month+day in files]

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
    def __str__(self):
        return 'Met Norway NorKyst'

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
                    raise NoAvailableData(f'{file} does not have a complete timeseries, discard the date.')
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
    def __str__(self):
        if avg:
            return 'NorShelf daily values'

        else:
            return 'NorShelf hourly values'

    def __init__(self, avg):
        self.avg  = avg
        self.path = 'https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_avg_an_20210531T00Z.nc'
        self.min_len = 24
        if self.avg:
            self.min_len = 1

    def test_day(self, date):
        '''
        Load norshelf path.
        - norshelf is a forecast model, so we sometimes get more dates there than with norkyst
        '''
        file = self.get_norshelf_day_url(date)

        # Test if the data is good
        # ----
        forecast_nr = 0
        while True:
            try:
                try:
                   d  = Dataset(file, 'r')
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

# Methods for downloading data from a ROMS output file and preparing them to be interpolated to FVCOM
# ----
class RomsDownloader:
    '''
    Routine that downloads data. It will automatically stop and wait if the thredds server goes down / the data we need is temporarily unavailable
    '''
    def read_timestep(self, index_here, filepath, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va'], sigma = None):
        '''
        Reads a timestep
        '''
        unavailable = True
        while unavailable:
            try:
                timestep = self._load_roms_data(index_here, filepath, variables, sigma)
                unavailable = False

            except:
                print("\n--------------------------------------\n"\
                      + "The data is unavailable at the moment.\n"\
                      + "We wait thirty seconds and try again.\n"\
                      + "--------------------------------------\n")
                time_mod.sleep(30)
        return timestep

    def _load_roms_data(self, index_here, filepath, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va'], sigma = None):
        '''
        Dumps roms data from the netcdf file and prepare for interpolation
        - how we access the ROMS data may depend on the interpolation method?
        '''
        supported = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']
        timestep = ROMSTimeStep(index_here)
 
        with Dataset(filepath) as nc:
            for var in variables:
                if var not in supported:
                    raise InputError(f'The variable: "{var}" is not supported by RomsDownloader')

            if 'salt' in variables:
                if sigma is None:
                    timestep.salt = nc['salt'][index_here, :, self.N4.m_ri:(self.N4.x_ri+1), self.N4.m_rj:(self.N4.x_rj+1)]
                else:
                    timestep.salt = nc['salt'][index_here, sigma, self.N4.m_ri:(self.N4.x_ri+1), self.N4.m_rj:(self.N4.x_rj+1)]

            if 'temp' in variables:
                if sigma is None:
                   timestep.temp = nc['temp'][index_here, :, self.N4.m_ri:(self.N4.x_ri+1), self.N4.m_rj:(self.N4.x_rj+1)]
                else:
                    timestep.temp = nc['temp'][index_here, sigma, self.N4.m_ri:(self.N4.x_ri+1), self.N4.m_rj:(self.N4.x_rj+1)]
            if 'zeta' in variables:
                timestep.zeta = nc['zeta'][index_here, self.N4.m_ri:(self.N4.x_ri+1), self.N4.m_rj:(self.N4.x_rj+1)]

            if 'u' in variables:
                timestep.u    = nc['u'][index_here, :, self.N4.m_ui:(self.N4.x_ui+1), self.N4.m_uj:(self.N4.x_uj+1)]

            if 'v' in variables:
                timestep.v    = nc['v'][index_here, :, self.N4.m_vi:(self.N4.x_vi+1), self.N4.m_vj:(self.N4.x_vj+1)]

            if 'ua' in variables:
                try:
                    timestep.ua = nc['ubar'][index_here, self.N4.m_ui:(self.N4.x_ui+1), self.N4.m_uj:(self.N4.x_uj+1)]
                except:
                    timestep.ua = np.sum(timestep.u*self.N4.cropped_dsigma_u, axis = 0)

            if 'va' in variables:
                try:
                    timestep.va = nc['vbar'][index_here, self.N4.m_vi:(self.N4.x_vi+1), self.N4.m_vj:(self.N4.x_vj+1)]
                except:
                    timestep.va = np.sum(timestep.v*self.N4.cropped_dsigma_v, axis = 0)

        return timestep

    def crop_and_transpose(self, timestep, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']):
        '''
        Re-shape downloaded field to fit the interpolation method
        '''
        if 'salt' in variables:
            timestep.salt = timestep.salt[:, self.N4.cropped_rho_mask].transpose()
        if 'temp' in variables:
            timestep.temp = timestep.temp[:, self.N4.cropped_rho_mask].transpose()
        if 'zeta' in variables:
            timestep.zeta = timestep.zeta[self.N4.cropped_rho_mask].transpose()
        if 'u' in variables:
            timestep.u    = timestep.u[:, self.N4.cropped_u_mask].transpose()
        if 'v' in variables:
            timestep.v    = timestep.v[:, self.N4.cropped_v_mask].transpose()
        if 'ua' in variables:
            timestep.ua   = timestep.ua[self.N4.cropped_u_mask].transpose()
        if 'va' in variables:
            timestep.va   = timestep.va[self.N4.cropped_v_mask].transpose()
        return timestep

    def adjust_uv(self, timestep):
        '''
        rotate angle of current to match north/south as in forcing model
        '''
        # First rotate to north/east
        timestep.u, timestep.v   = self._rotate_to_latlon(timestep.u, timestep.v, self.N4.angle)
        timestep.ua, timestep.va = self._rotate_to_latlon(timestep.ua, timestep.va, self.N4.angle)

        # Then correct for the meridional convergence in utm coordinates if needbe
        if not self.latlon:
            timestep.u, timestep.v   = self._rotate_from_latlon(timestep.u, timestep.v, self.N4.cell_utm_angle)
            timestep.ua, timestep.va = self._rotate_from_latlon(timestep.ua, timestep.va, self.N4.cell_utm_angle)
        return timestep

    def _rotate_to_latlon(self, u, v, angle):
        '''
        Rotates vectors from (x', y') system to (lon, lat)
        '''
        unew = u*np.cos(angle) - v*np.sin(angle)
        vnew = u*np.sin(angle) + v*np.cos(angle)
        return unew, vnew

    def _rotate_from_latlon(self, u, v, angle):
        '''
        Rotates vectors from (lon, lat) system to (x', y')
        '''
        unew =  u*np.cos(angle) + v*np.sin(angle)
        vnew = -u*np.sin(angle) + v*np.cos(angle)
        return unew, vnew

@dataclass
class ROMSTimeStep:
    '''
    Fields we expect in other routines from this field
    '''
    netcdf_target_index: int
    salt: np.array = np.empty(0)
    temp: np.array = np.empty(0)
    u: np.array = np.empty(0)
    v: np.array = np.empty(0)
    ua: np.array = np.empty(0)
    va: np.array = np.empty(0)

class NoAvailableData(Exception): pass
class InputError(Exception): pass