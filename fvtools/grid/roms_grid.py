# ==================================================================================================================
#                                Routines to access specific ROMS models
# =================================================================================================================
import numpy as np
import time as time_mod

from datetime import datetime, timedelta
from functools import cached_property
from netCDF4 import Dataset
from dataclasses import dataclass, field
from .cropper_roms import ROMSCropper

def get_roms_grid(mother, projection = None):
    '''
    Returns a ROMS object for the mother model of question
    - we currently have readers for NorShelf (mother=NS) and the MET NorKyst models (NKv2 and NKv3).

    NorKyst is a 800 m grid spacing ROMS model intended for Norway-scale fjord studies. NorShelf is a data assimilated
    ROMS model often used for SAR operational forecasting and other outside-of-the-coast applications.
    '''
    if mother == 'D-NS':
        ROMS = NorShelf(avg = True)

    elif mother == 'H-NS':
        ROMS = NorShelf(avg = False)

    elif mother == 'NKv2':
        ROMS = METNorKystV2()
    
    elif mother == 'NKv3':
        ROMS = METNorKystV3()

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
        return self._h_rho + self.zeta
    
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
        '''
        Depth where scalar-variables are stored
        - Will depend on which stretching function the ROMS model was operated with
        - See more details in the ROMS documentation, 
        '''
        S_rho = np.tile(np.zeros((self.lon_rho.shape))[:,:, None], (1,1,len(self.Cs_r)))
        for i, (s, c) in enumerate(zip(self.s_rho, self.Cs_r)):
            S_rho[:, :, i] = (self.hc*s + self.h_rho[:, :]*c)/(self.hc+self.h_rho)
        return S_rho

    @property
    def S_w(self):
        '''
        Depth at interface between sigma layers (as well as top/bottom). Stored in the same way as internally in ROMS
        '''
        S_w   = np.tile(np.zeros((self.lon_rho.shape))[None, :, :], (len(self.Cs_w), 1, 1))
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
        return self.h_rho[:, :, None] * self.S_rho

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
                self._path = self.test_day(datetime(2024,1,1))
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
        load_position_fields = [
            'lon_rho', 'lat_rho', 
            'lat_u', 'lat_v', 
            'lon_u', 'lon_v',
            'h', 'angle', 
            'Cs_r', 's_rho', 
            'Cs_w', 's_w', 
            'hc',
            'Vstretch'
            ]
        with Dataset(self.path, 'r') as ncdata:
            for load in load_position_fields:
                try:
                    setattr(self, load, ncdata.variables.get(load)[:])
                except:
                    if load == 'Vstretch':
                        setattr(self, load, ncdata.variables.get('Vstretching')[:])
                    else:
                        raise TypeError(f'{load} is not available in {self.path}.')

            self.h_rho = self.__dict__.pop('h')

            load_mask = ['u', 'v', 'rho']
            for load in load_mask:
                setattr(self, f'{load}_mask', ((ncdata.variables.get(f'mask_{load}')[:]-1)*(-1)).astype(bool))

    def get_x_y_z(self):
        '''
        compute z at each S level for mean sealevel, project roms lon,lat to desired projection
        '''
        self.x_rho, self.y_rho = self.Proj(self.lon_rho, self.lat_rho, inverse=False)
        self.x_u,   self.y_u   = self.Proj(self.lon_u,   self.lat_u,   inverse=False)
        self.x_v,   self.y_v   = self.Proj(self.lon_v,   self.lat_v,   inverse=False)

# Each ROMS setup will have its own quirks w/respect to finding data, these quirks should be adressed in these subclasses
class METNorKystV2(ROMSbase):
    '''
    Routines to check if MET-NorKyst data is available
    '''
    def __str__(self):
        return 'Met Norway NorKyst version 2'

    def test_day(self, date):
        '''
        Check if the file exists that day, and that it has enough data
        '''
        file = self.get_norkyst_url(date)
        self.test_ncfile(file)
        self.path = file
        return file

    def get_norkyst_url(self, date):
        '''
        Give it a date, and you will get the corresponding url in return
        '''
        https       = 'https://thredds.met.no/thredds/dodsC/fou-hi/new_norkyst800m/his/ocean_his.an.'
        year        = str(date.year)
        month       = '{:02d}'.format(date.month)
        day         = '{:02d}'.format(date.day)
        return f'{https}{year}{month}{day}.nc'

    def test_ncfile(self, file):
        try:
            with Dataset(file, 'r') as d:
                if len(d.variables['ocean_time'][:])<24:
                    raise NoAvailableData(f'{file} does not have a complete timeseries, discard the date.')
        except:
            raise NoAvailableData

class METNorKystV3(ROMSbase):
    '''Routines to check if MET-NorKyst v3 data is available'''
    def __str__(self):
        return 'MET Norway NorKyst version 3'
    
    def test_day(self, date):
        '''
        Check if the file exists that day, and that it has enough data
        '''
        file = self.get_norkyst_url(date)
        self.test_ncfile(file)
        self.path = file
        return file

    def get_norkyst_url(self, date):
        '''
        Give it a date, and you will get the corresponding url in return
        '''
        https       = 'https://thredds.met.no/thredds/dodsC/fou-hi/new_norkyst800m/norkyst_v3_test/his/'
        year        = str(date.year)
        month       = '{:02d}'.format(date.month)
        day         = '{:02d}'.format(date.day)
        return f'{https}/{year}/{month}/{day}/norkyst800_his_sdepth_{year}{month}{day}T00Z_m00_AN.nc'
    
    def test_ncfile(self, file):
        try:
            with Dataset(file, 'r') as d:
                if len(d.variables['ocean_time'][:])<24:
                    raise NoAvailableData(f'{file} does not have a complete timeseries, discard the date.')
        except:
            raise NoAvailableData

class NorShelf(ROMSbase):
    '''
    for accessing NorShelf data
    '''
    def __str__(self):
        if self.avg:
            return 'NorShelf daily values'

        else:
            return 'NorShelf hourly values'

    def __init__(self, avg = False):
        '''
        Access NorShelf grid data, will by default try to load hourly values. Set avg to True if you want daily averages.
        '''
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
    salt: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    temp: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    u: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    v: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    ua: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    va: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))

class NoAvailableData(Exception): pass
class InputError(Exception): pass