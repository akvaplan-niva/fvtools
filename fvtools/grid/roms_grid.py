# ==================================================================================================================
#                                Routines to access specific ROMS models
# =================================================================================================================
import numpy as np
import time as time_mod
import os

from datetime import datetime, timedelta
from functools import cached_property
from netCDF4 import Dataset
from dataclasses import dataclass, field
from .cropper_roms import ROMSCropper

def get_roms_grid(mother, projection = None):
    '''
    Returns a ROMS object for the mother model of question
    - we currently have readers for NorShelf (mother=NS) and the MET NorKyst models (NKv2 and NKv3, and the
      NorKyst v3 hindcast: NKv3-hindcast).

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

    elif mother == 'NKv3-hindcast':
        ROMS = METNorKystV3Hindcast()

    elif mother == 'IMR-NF':
        ROMS = IMRNorFjords()

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

    # u and v points: halfway between rho points on a staggered (Arakawa C) grid, the rho points themselves
    # when the mother model's velocities are given at the rho points (staggered = False)
    @property
    def h_u(self):
        if not getattr(self, 'staggered', True):
            return self.h_rho
        return (self.h_rho[:, 1:]   + self.h_rho[:, :-1])/2
    
    @property
    def h_v(self):
        if not getattr(self, 'staggered', True):
            return self.h_rho
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
        if not getattr(self, 'staggered', True):
            return self.z_rho
        return(self.z_rho[:, 1:, :] + self.z_rho[:, :-1, :])/2

    @property
    def zw_u(self):
        if not getattr(self, 'staggered', True):
            return self.zw_rho
        return (self.zw_rho[:, :, 1:] + self.zw_rho[:, :, :-1])/2

    @property
    def z_v(self):
        if not getattr(self, 'staggered', True):
            return self.z_rho
        return (self.z_rho[1:, :, :] + self.z_rho[:-1, :, :])/2

    @property
    def zw_v(self):
        if not getattr(self, 'staggered', True):
            return self.zw_rho
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
    # How the mother model's files name things. ROMS output by default; a reader for post-processed files
    # (e.g. METNorKystV3Hindcast) sets its own. The downloader reads the names through N4.varnames.
    time_name = 'ocean_time'
    varnames  = {'salt': 'salt', 'temp': 'temp', 'zeta': 'zeta', 'u': 'u', 'v': 'v', 'ua': 'ubar', 'va': 'vbar'}
    staggered = True   # u, v on their own grids and along the grid axes (rotated to east/north by angle)

    def load_grid(self, xbounds=None, ybounds=None, offset=7500, date = None):
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

        if date is not None:
            try:
                path = self.test_day(date)
            except:
                raise ValueError(f'Could not find a valid {self} file for {date}')

        else:
            try:
                path = self.test_day(datetime.now())
            except:
                raise ValueError(f'Could not find a valid {self} file for {datetime.now()}')

        # Load grid positions
        self.load_grid_from_nc(path)
        self.get_x_y_z()


    def load_grid_from_nc(self, path):
        '''
        Load the position data we need to get going
        - positions (of rho, u and v points)
        - depth (at rho points)
        - stretching functions (for rho and w points)
        - angle between XI-axis and east
        '''
        # Thredds can refuse a request for a while (e.g. HTTP 429), retry before giving up
        retries, wait = 5, 30
        for attempt in range(retries):
            try:
                self._read_grid_fields(path)
                return
            except (RuntimeError, OSError) as error:
                if attempt == retries - 1:
                    raise
                print(f'  - could not read the grid from {path} ({error!r}), trying again in {wait*(attempt+1)} s')
                time_mod.sleep(wait*(attempt+1))

    def _read_grid_fields(self, path):
        '''
        Read the grid fields listed in load_grid_from_nc
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
        with Dataset(path, 'r') as ncdata:
            for load in load_position_fields:
                var = ncdata.variables.get(load)
                if var is None and load == 'Vstretch':
                    var = ncdata.variables.get('Vstretching')
                if var is None:
                    raise TypeError(f'{load} is not available in {path}.')
                setattr(self, load, var[:])

            self.h_rho = self.__dict__.pop('h')

            # True on land. Round first: files written with lossy compression (e.g. MET's NorKyst v3, ncks --ppc)
            # can store water as 1.0000038 rather than 1
            load_mask = ['u', 'v', 'rho']
            for load in load_mask:
                setattr(self, f'{load}_mask', np.round(np.asarray(ncdata.variables.get(f'mask_{load}')[:])) == 0)

    def test_ncfile(self, file, min_steps = 24, retries = 4, wait = 15):
        '''
        Check that the file opens and holds a complete day. Errors when opening it (e.g. HTTP 429 from thredds)
        are retried before the date is declared missing.
        '''
        for attempt in range(retries):
            try:
                with Dataset(file, 'r') as d:
                    n_steps = len(d.variables[self.time_name][:])
                break
            except FileNotFoundError as error:
                raise NoAvailableData(f'{file} does not exist') from error
            except Exception as error:
                if attempt == retries - 1:
                    raise NoAvailableData(f'could not open {file} ({error!r})') from error
                time_mod.sleep(wait*(attempt+1))

        if n_steps < min_steps:
            raise NoAvailableData(f'{file} does not have a complete timeseries, discard the date.')

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
        return file

    def get_norkyst_url(self, date):
        '''
        Give it a date, and you will get the corresponding url in return
        '''
        https       = 'https://thredds.met.no/thredds/dodsC/fou-hi/new_norkyst800m/norkyst_v3_test/his/'
        year        = str(date.year)
        month       = '{:02d}'.format(date.month)
        day         = '{:02d}'.format(date.day)
        return f'{https}{year}/{month}/{day}/norkyst800_his_sdepth_{year}{month}{day}T00Z_m00_AN.nc'
    
class METNorKystV3Hindcast(ROMSbase):
    '''
    MET Norway's NorKyst v3 hindcast (thredds romshindcast/norkyst_v3/sdepth), one file per day with 24 hourly steps.

    The files are post-processed from the model output, so they differ from the operational NKv3 files:
    - every field is at the rho points; u and v here use the rho grid (staggered = False);
    - the currents are already east/north (u_eastward, v_northward, ubar_eastward, vbar_northward): angle = 0,
      and only the turn from east/north into the FVCOM grid (latlon = False) remains;
    - other names: temperature, salinity, time (seconds since 1970), lon, lat, sea_mask;
    - no angle, u/v grids or Vtransform: the vertical grid is ROMS Vtransform 2 (s_rho is ocean_s_coordinate_g2,
      checked when the grid is read), from hc, s_rho, Cs_r, s_w and Cs_w.
    u_eastward and v_northward are not masked on land (they are near zero there), while ubar_eastward and
    vbar_northward are masked on land and also at sea cells next to it (about 1 % of the sea cells near Tampen,
    the same in every step). So the rho land mask comes from sea_mask, and the u and v land masks add every
    cell where the first step of ubar_eastward or vbar_northward is masked: no velocity is taken from there.
    '''
    time_name = 'time'
    varnames  = {'salt': 'salinity', 'temp': 'temperature', 'zeta': 'zeta', 'u': 'u_eastward', 'v': 'v_northward',
                 'ua': 'ubar_eastward', 'va': 'vbar_northward'}
    staggered = False
    grid_date = datetime(2024, 1, 1)   # a day in the hindcast, to read the grid from when load_grid gets no date

    def __str__(self):
        return 'MET Norway NorKyst version 3 hindcast'

    def test_day(self, date):
        '''
        Check if the file exists that day, and that it has enough data
        '''
        file = self.get_norkyst_url(date)
        self.test_ncfile(file)
        return file

    def get_norkyst_url(self, date):
        '''
        Give it a date, and you will get the corresponding url in return
        '''
        https = 'https://thredds.met.no/thredds/dodsC/romshindcast/norkyst_v3/sdepth/'
        return f'{https}{date.year}/{date.month:02d}/norkyst800-{date.year}{date.month:02d}{date.day:02d}.nc'

    def load_grid(self, xbounds=None, ybounds=None, offset=7500, date = None):
        # the base class looks for today's file when no date is given, and the hindcast has none
        super().load_grid(xbounds, ybounds, offset, date = self.grid_date if date is None else date)

    def _read_grid_fields(self, path):
        '''
        Read the grid; the u and v grids are the rho grid, and the angle is zero (currents are east/north)
        '''
        with Dataset(path, 'r') as ncdata:
            for load in ['lon', 'lat', 'h', 'sea_mask', 's_rho', 'Cs_r', 's_w', 'Cs_w', 'hc']:
                if load not in ncdata.variables:
                    raise TypeError(f'{load} is not available in {path}.')
            kind = getattr(ncdata.variables['s_rho'], 'standard_name', '')
            if kind != 'ocean_s_coordinate_g2':
                raise TypeError(f'{path}: s_rho is {kind!r}, but the depths here assume ROMS Vtransform 2 (ocean_s_coordinate_g2)')

            self.lon_rho = np.asarray(ncdata.variables['lon'][:])
            self.lat_rho = np.asarray(ncdata.variables['lat'][:])
            self.h_rho   = np.asarray(ncdata.variables['h'][:], dtype = float)
            for load in ['Cs_r', 's_rho', 'Cs_w', 's_w', 'hc']:
                setattr(self, load, ncdata.variables[load][:])

            # True on land
            self.rho_mask = np.ma.filled(ncdata.variables['sea_mask'][:], 0) != 1

            # and for the velocities, also where the depth-mean currents are masked (sea cells next to land)
            uv_land = self.rho_mask.copy()
            for load in ['ubar_eastward', 'vbar_northward']:
                if load in ncdata.variables:
                    uv_land |= np.ma.getmaskarray(ncdata.variables[load][0])

        self.u_mask, self.v_mask = uv_land, uv_land
        self.lon_u, self.lat_u   = self.lon_rho, self.lat_rho
        self.lon_v, self.lat_v   = self.lon_rho, self.lat_rho
        self.angle = np.zeros(self.lon_rho.shape)

    # The nearest-4 search looks for the box centre closest to each FVCOM point. With u and v at the rho points,
    # every box is a rho box, and its centre is the mean of its four corners.
    def _psi_from_rho(self, field):
        m = self.fv_rho_mask
        inside = m[1:, 1:] & m[1:, :-1] & m[:-1, 1:] & m[:-1, :-1]
        return ((field[1:, 1:] + field[1:, :-1] + field[:-1, 1:] + field[:-1, :-1])/4)[inside]

    @property
    def cropped_x_psi(self):
        return self._psi_from_rho(self.x_rho)

    @property
    def cropped_y_psi(self):
        return self._psi_from_rho(self.y_rho)

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

class IMRNorFjords(ROMSbase):
    '''
    Routines to check if IMR-NorFjords data is available, inherits grid-methods from ROMSbase
    '''
    def __str__(self):
        return 'Havforskningsinstituttet Norfjords simulations'

    @cached_property
    def all_local_norkyst_files(self):
        '''
        property holding all local ncfiles
        '''
        oliviafolder = '/cluster/work/projects/nn9238k/norfjords/'
        print(f'- Assuming that norfjords data is available under {oliviafolder}')
        self.folders = [oliviafolder]
        self._bottom_folders()
        return self._list_ncfiles()

    def test_day(self, date):
        '''
        See if the local file exists that day, and has enough data
        date: datetime object
        '''
        file = self.get_norkyst_local(date)
        self.test_ncfile(file)
        return file

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
        year  = str(date.year)
        month = '{:02d}'.format(date.month)
        day   = '{:02d}'.format(date.day)

        files = [files for files in all_ncfiles if year+month+day in files]

        if not any(files):
            print(f'- did not find {date}')

        # I want the file that starts the same date as my date
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

        # remove folders that contain folders
        leaf_branch = []
        for dr in dirs:
            if dr[-1]=='/':
                continue
            else:
                # This string is at the end of the branch, thus this is where the data is stored
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
        names = getattr(self.N4, 'varnames', None) or ROMSbase.varnames   # the mother model's names in its files
 
        with Dataset(filepath) as nc:
            for var in variables:
                if var not in supported:
                    raise InputError(f'The variable: "{var}" is not supported by RomsDownloader')

            if 'salt' in variables:
                if sigma is None:
                    timestep.salt = nc[names['salt']][index_here, :, self.N4.m_ri:(self.N4.x_ri+1), self.N4.m_rj:(self.N4.x_rj+1)]
                else:
                    timestep.salt = nc[names['salt']][index_here, sigma, self.N4.m_ri:(self.N4.x_ri+1), self.N4.m_rj:(self.N4.x_rj+1)]

            if 'temp' in variables:
                if sigma is None:
                   timestep.temp = nc[names['temp']][index_here, :, self.N4.m_ri:(self.N4.x_ri+1), self.N4.m_rj:(self.N4.x_rj+1)]
                else:
                    timestep.temp = nc[names['temp']][index_here, sigma, self.N4.m_ri:(self.N4.x_ri+1), self.N4.m_rj:(self.N4.x_rj+1)]
            if 'zeta' in variables:
                timestep.zeta = nc[names['zeta']][index_here, self.N4.m_ri:(self.N4.x_ri+1), self.N4.m_rj:(self.N4.x_rj+1)]

            if 'u' in variables:
                timestep.u    = nc[names['u']][index_here, :, self.N4.m_ui:(self.N4.x_ui+1), self.N4.m_uj:(self.N4.x_uj+1)]

            if 'v' in variables:
                timestep.v    = nc[names['v']][index_here, :, self.N4.m_vi:(self.N4.x_vi+1), self.N4.m_vj:(self.N4.x_vj+1)]

            if 'ua' in variables:
                try:
                    timestep.ua = nc[names['ua']][index_here, self.N4.m_ui:(self.N4.x_ui+1), self.N4.m_uj:(self.N4.x_uj+1)]
                except:
                    timestep.ua = np.sum(timestep.u*self.N4.cropped_dsigma_u, axis = 0)

            if 'va' in variables:
                try:
                    timestep.va = nc[names['va']][index_here, self.N4.m_vi:(self.N4.x_vi+1), self.N4.m_vj:(self.N4.x_vj+1)]
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