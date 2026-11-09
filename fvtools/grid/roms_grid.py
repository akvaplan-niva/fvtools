# ==================================================================================================================
#                                Routines to access specific ROMS models
# =================================================================================================================
from contextlib import contextmanager
import numpy as np
import time as time_mod
import os

from datetime import datetime, timedelta
from functools import cached_property
from netCDF4 import Dataset
from dataclasses import dataclass, field
from .roms.cropper_roms import ROMSCropper
from .roms.source_adapter import NATIVE_ROMS, NORKYST_V3_HINDCAST
from .roms.coastal_mask import probe_hindcast_wet_crop

from fvtools.util.thredds_io import is_retryable_netcdf_error
from fvtools.util.thredds_io import SourcePool
from fvtools.util.thredds_io import source_dataset
from fvtools.util.thredds_io import retry_read

def get_roms_time_variable(nc):
    """Get the time axis of operational or hindcast ROMS NetCDF files.

    Prefer ``ocean_time`` when both names are present, as in the original
    operational products. Hindcast files may use ``time`` instead.
    This only selects a NetCDF variable; it performs no remote data read.
    """
    for name in ('ocean_time', 'time'):
        if name in nc.variables:
            return nc.variables[name]
    raise ValueError("ROMS source has neither 'ocean_time' nor 'time'")


def compute_roms_s_coordinates(nlevels, vstretching):
    """Reproduce the ROMS s-coordinate definitions for Vstretching 1-5.

    The s-coordinates are independent of theta_s, theta_b and hc. The
    corresponding Cs_r/Cs_w stretching curves must still come from the
    source file; they are not computed here.

    Vstretching=5 follows ROMS/Utility/set_scoord.F (Souza coordinates),
    which uses a quadratic distribution rather than equally spaced s.
    Arrays are ordered bottom-to-surface, like ROMS NetCDF coordinates.
    """
    n = int(nlevels)
    if n < 1 or n != nlevels:
        raise ValueError(f'Invalid number of ROMS s_rho levels: {nlevels!r}')
    version = int(np.asarray(vstretching).item())

    k_rho = np.arange(1, n + 1, dtype=np.float64) - 0.5
    k_w = np.arange(n + 1, dtype=np.float64)

    if version in (1, 2, 3, 4):
        s_rho = (k_rho - n) / n
        s_w = (k_w - n) / n
    elif version == 5:
        if n < 2:
            raise ValueError('Vstretching=5 requires at least two ROMS layers')

        # Exact quadratic function in ROMS set_scoord.F, applied to both
        # half-integer rho levels and integer w levels.
        def souza_s(k):
            return (-(k*k - 2.0*k*n + k + n*n - n) / (n*n - n)
                    - 0.01*(k*k - k*n) / (1.0 - n))

        s_rho = souza_s(k_rho)
        s_w = souza_s(k_w)
        # ROMS sets these endpoints explicitly.
        s_w[0] = -1.0
        s_w[-1] = 0.0
    else:
        raise ValueError(f'Unsupported ROMS Vstretching={version}; expected 1-5')

    return s_rho, s_w


def read_roms_s_coordinates(nc):
    """Prefer stored s_rho/s_w; calculate only missing coordinates.

    Reading Cs_r metadata to determine N avoids any extra remote field read.
    Missing coordinates require a Vstretch/Vstretching scalar variable or
    the corresponding global attribute. Present coordinates need no such
    metadata, so existing hindcast files retain their previous behavior.
    """
    nlevels = len(nc.variables['Cs_r'])
    if len(nc.variables['Cs_w']) != nlevels + 1:
        raise ValueError('ROMS Cs_w must contain N+1 interfaces')

    s_rho = nc.variables['s_rho'][:] if 's_rho' in nc.variables else None
    s_w = nc.variables['s_w'][:] if 's_w' in nc.variables else None

    if s_rho is None or s_w is None:
        source = nc.variables.get('Vstretch')
        if source is None:
            source = nc.variables.get('Vstretching')
        if source is not None:
            version = source[()]
        else:
            version = getattr(nc, 'Vstretching', None)
            if version is None:
                version = getattr(nc, 'Vstretch', None)
        if version is None:
            raise ValueError('Cannot calculate missing ROMS s-coordinates: '
                             'Vstretching/Vstretch metadata is absent')
        calculated_rho, calculated_w = compute_roms_s_coordinates(
            nlevels, version)
        if s_rho is None:
            s_rho = calculated_rho
        if s_w is None:
            s_w = calculated_w

    if np.ndim(s_rho) != 1 or len(s_rho) != nlevels:
        raise ValueError(f'ROMS s_rho must have shape ({nlevels},)')
    if np.ndim(s_w) != 1 or len(s_w) != nlevels + 1:
        raise ValueError(f'ROMS s_w must have shape ({nlevels + 1},)')
    return s_rho, s_w


def get_roms_grid(mother, projection = None):
    '''
    Returns a ROMS object for the mother model of question
    - we currently have readers for NorShelf (mother=NS) and the MET NorKyst models (NKv2 and NKv3).

    NorKyst is a 800 m grid spacing ROMS model intended for Norway-scale fjord studies. NorShelf is a data assimilated
    ROMS model often used for SAR operational forecasting and other outside-of-the-coast applications.

    Valid options:
    - D-NS, H-NS - Daily and hourly norshelf data
    - NKv2       - NorKyst version 2 stored at metno thredds
    - NKv3       - operational NorKyst version 3 stored at metno thredds
    - NKv3H      - hincast NorKyst version 3 stored at metno thredds
    - IMR-NKv3   - NorKyst version 3 downloaded from the institute of marine research AWS S3 storage
    - IMR-NF     - NorFjords downloaded from institute of marine research AWS S3 storage
    '''
    if mother == 'D-NS':
        ROMS = NorShelf(avg = True)

    elif mother == 'H-NS':
        ROMS = NorShelf(avg = False)

    elif mother == 'NKv2':
        ROMS = METNorKystV2()
    
    elif mother == 'NKv3':
        ROMS = METNorKystV3()

    elif mother == 'IMR-NKv3':
        ROMS = IMRNorKystv3()

    elif mother in ('NKv3-hindcast', 'NKv3H'):
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
    source_adapter = NATIVE_ROMS

    @property
    def is_collocated(self):
        return self.source_adapter.collocated
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
        load_position_fields = [
            'lon_rho', 'lat_rho',
            'lat_u', 'lat_v',
            'lon_u', 'lon_v',
            'h', 'angle',
            'Cs_r', 'Cs_w',
        ]
        with Dataset(path, 'r') as ncdata:
            for name in load_position_fields:
                if name not in ncdata.variables:
                    raise TypeError(f'{name} is not available in {path}.')
                setattr(self, name, ncdata.variables[name][:])

            self.hc = ncdata.variables['hc'][()]
            source = ncdata.variables.get('Vstretch')
            if source is None:
                source = ncdata.variables.get('Vstretching')
            if source is None:
                raise TypeError(f'Vstretching/Vstretch is not available in {path}.')
            self.Vstretch = np.asarray(source[()]).item()

            self.s_rho, self.s_w = read_roms_s_coordinates(ncdata)
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
                if len(get_roms_time_variable(d)) < 24:
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
        return file

    def get_norkyst_url(self, date):
        '''
        Operational NorKyst v3 on the native staggered ROMS C-grid.
        '''
        https = 'https://thredds.met.no/thredds/dodsC/fou-hi/new_norkyst800m/norkyst_v3_test/his/'
        year = str(date.year)
        month = f'{date.month:02d}'
        day = f'{date.day:02d}'
        return (f'{https}{year}/{month}/{day}/'
                f'norkyst800_his_sdepth_{year}{month}{day}T00Z_m00_AN.nc')
    
    def test_ncfile(self, file):
        try:
            with Dataset(file, 'r') as d:
                if len(get_roms_time_variable(d)) < 24:
                    raise NoAvailableData(f'{file} does not have a complete timeseries, discard the date.')
        except:
            raise NoAvailableData

class METNorKystV3Hindcast(METNorKystV3):
    """Postprocessed sdepth NorKyst v3: collocated, geographic velocities.

    Keeps the native ROMS grid/interpolator API.  The physical source arrays
    have shape (time, s_rho, Y, X) for 3-D fields and (time, Y, X) for 2-D.
    """
    source_adapter = NORKYST_V3_HINDCAST

    def __str__(self):
        return 'MET Norway NorKyst version 3 hindcast (sdepth)'

    def get_norkyst_url(self, date):
        base = 'https://thredds.met.no/thredds/dodsC/romshindcast/norkyst_v3/sdepth'
        return (f'{base}/{date.year}/{date.month:02d}/'
                f'norkyst800-{date.year}{date.month:02d}{date.day:02d}.nc')

    def load_grid_from_nc(self, path):
        """Read the one collocated source grid exactly once per setup."""
        with Dataset(path, 'r') as nc:
            required = ('lon', 'lat', 'h', 'sea_mask', 'Cs_r', 'Cs_w', 'hc')
            absent = [name for name in required if name not in nc.variables]
            if absent:
                raise ValueError(f'NorKyst v3 hindcast is missing {absent}: {path}')
            for name in ('lon', 'lat', 'h', 'sea_mask'):
                if tuple(nc.variables[name].dimensions) != ('Y', 'X'):
                    raise ValueError(f'Hindcast {name} must have (Y, X) dimensions')
            for name in ('s_rho', 'Cs_r'):
                if name in nc.variables and tuple(nc.variables[name].dimensions) != ('s_rho',):
                    raise ValueError(f'Hindcast {name} must have (s_rho,) dimensions')
            for name in ('s_w', 'Cs_w'):
                if name in nc.variables and tuple(nc.variables[name].dimensions) != ('s_w',):
                    raise ValueError(f'Hindcast {name} must have (s_w,) dimensions')
            self.lon_rho = nc.variables['lon'][:]
            self.lat_rho = nc.variables['lat'][:]
            self.h_rho = nc.variables['h'][:]
            self.rho_mask = np.asarray(nc.variables['sea_mask'][:]) != 1
            for name in ('Cs_r', 'Cs_w'):
                setattr(self, name, nc.variables[name][:])
            self.s_rho, self.s_w = read_roms_s_coordinates(nc)
            # hc is a scalar NetCDF variable: use [()], not [:].
            self.hc = nc.variables['hc'][()]

        self._hindcast_grid_path = path
        # Aliases, not copies; ROMSCropper and N4ROMS already know these names.
        self.lon_u = self.lon_v = self.lon_rho
        self.lat_u = self.lat_v = self.lat_rho
        self.u_mask = self.v_mask = self.rho_mask

    def load_grid(self, *args, **kwargs):
        """Build the effective wet mask before interpolation weights or MatchTopo.

        A hindcast field can be masked even where static sea_mask == 1.
        Open the already selected source file once more and only read the
        requested spatial crop at the first timestep; no full-domain reads.
        """
        super().load_grid(*args, **kwargs)
        window = (slice(self.m_ri, self.x_ri + 1),
                  slice(self.m_rj, self.x_rj + 1))
        with Dataset(self._hindcast_grid_path, 'r') as nc:
            self._hindcast_wet_crop = probe_hindcast_wet_crop(
                nc, window, self.rho_mask[window], self.h_rho[window]
            )
        n_static = int((~self.rho_mask[window]).sum())
        n_valid = int(self._hindcast_wet_crop.sum())
        print(f'  - Hindcast effective wet mask: {n_valid} / '
              f'{n_static} static ocean source cells are valid for all fields')

    def get_x_y_z(self):
        # One projection call rather than three, because all points coincide.
        self.x_rho, self.y_rho = self.Proj(self.lon_rho, self.lat_rho, inverse=False)
        self.x_u = self.x_v = self.x_rho
        self.y_u = self.y_v = self.y_rho

    @property
    def h_u(self):
        return self.h_rho

    @property
    def h_v(self):
        return self.h_rho

    @property
    def z_u(self):
        return self.z_rho

    @property
    def z_v(self):
        return self.z_rho

    @cached_property
    def _cropped_sigma_thickness(self):
        """Sigma thicknesses in the crop only; avoid full-domain 41-layer arrays."""
        depth = np.asarray(self.h_rho[
            self.m_ri:self.x_ri + 1, self.m_rj:self.x_rj + 1])
        # Land may have zero or missing depth; never divide by it. Those
        # values are excluded later by the existing source land mask.
        wet = np.isfinite(depth) & (depth > 0)
        safe_depth = np.where(wet, depth, 1.)
        s = np.asarray(self.s_w)[:, None, None]
        cs = np.asarray(self.Cs_w)[:, None, None]
        sw = ((self.hc * s + safe_depth[None, :, :] * cs)
              / (self.hc + safe_depth[None, :, :]))
        interfaces = safe_depth[None, :, :] * sw
        weights = np.diff(interfaces, axis=0) / safe_depth[None, :, :]
        return np.where(wet[None, :, :], weights, 0.)

    @property
    def cropped_dsigma_u(self):
        return self._cropped_sigma_thickness

    @property
    def cropped_dsigma_v(self):
        return self._cropped_sigma_thickness

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

                if len(get_roms_time_variable(d)) < self.min_len:  # Check to see if empty
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

# For data stored on norstore
class LocalData:
    def test_ncfile(self, file):
        try:
            with Dataset(file, 'r') as d:
                if len(get_roms_time_variable(d)) < 24:
                    raise NoAvailableData(f'{file} does not have a complete timeseries')
        except:
            raise NoAvailableData

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

    @cached_property
    def all_local_files(self):
        '''
        property holding all local ncfiles
        '''
        print(f'- Assuming that data is available under {self.local_folder}')
        self.folders = [self.local_folder]
        self._bottom_folders()
        return self._list_ncfiles()
    
    def test_day(self, date):
        '''
        See if the local file exists that day, and has enough data
        date: datetime object
        '''
        file = self.get_local(date)
        self.test_ncfile(file)
        return file

    def get_local(self, date):
        '''
        Looks for NorKyst data in the predefined folders.
        '''
        return self._connect_date_to_file(self.all_local_files, date)

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
        if dirs == self.folders:
            self.subfolders = self.folders

        else:
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

class IMRNorFjords(ROMSbase, LocalData):
    '''
    Routines to check if IMR-NorFjords data is available, inherits grid-methods from ROMSbase
    '''
    @property
    def local_folder(self):
        return '/cluster/work/projects/nn9238k/norfjords/'

    def __str__(self):
        return 'Havforskningsinstituttet Norfjords simulations'

class IMRNorKystv3(ROMSbase, LocalData):
    '''
    Routines to check if IMR-NorKyst data is available, inherits grid-methods from ROMSbase
    '''
    @property
    def local_folder(self):
        return '/cluster/work/projects/nn9238k/norkyst/'
    
    def __str__(self):
        return 'Havforskningsinstituttet Norkyst v3 simulations'
\

# Methods for downloading data from a ROMS output file and preparing them to be interpolated to FVCOM
class RomsDownloader:
    '''
    Routine that downloads data. It will automatically stop and wait if the thredds server goes down / the data we need is temporarily unavailable
    '''
    # Keep the historical entry point: downstream code and tests call this.
    @staticmethod
    def _is_retryable_roms_io_error(exc):
        """Same bounded classification as MetCoOp: DAP/curl are transient."""
        return is_retryable_netcdf_error(exc)

    @contextmanager
    def reuse_roms_datasets(self):
        """Keep at most one ROMS source file open during a serial pass.

        No global cache: parallel workers and standalone reads remain isolated.
        All users of RomsDownloader (nesting and restart) share this behavior.
        """
        if getattr(self, '_roms_reader_cache', None) is not None:
            raise RuntimeError('Nested ROMS Dataset reuse scopes are not supported')
        pool = SourcePool(Dataset, max_open=1, mode=None)
        self._roms_reader_cache = pool
        try:
            yield self
        finally:
            self._roms_reader_cache = None
            pool.close()

    @contextmanager
    def _roms_dataset(self, filepath):
        """Use the shared connection context, with damage-aware eviction."""
        pool = getattr(self, '_roms_reader_cache', None)
        with source_dataset(filepath, Dataset, pool=pool,
                            classifier=self._is_retryable_roms_io_error) as nc:
            yield nc

    def read_timestep(self, index_here, filepath, variables=None, sigma=None,
                      *, max_retries=5, retry_delay=30):
        """Read a single ROMS timestep with shared bounded NetCDF/DAP retry."""
        if variables is None:
            variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']
        return retry_read(
            lambda: self._load_roms_data(index_here, filepath, variables, sigma),
            max_retries=max_retries, retry_delay=retry_delay,
            classifier=self._is_retryable_roms_io_error,
            sleep=time_mod.sleep,
            failure_message=(f'Could not read ROMS timestep {index_here} '
                             f'from {filepath!s}'),
            log_prefix=f'  - ROMS I/O error for {filepath}',
        )

    def _load_roms_data(self, index_here, filepath, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va'], sigma = None):
        '''
        Dumps roms data from the netcdf file and prepare for interpolation
        - how we access the ROMS data may depend on the interpolation method?
        '''
        supported = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']
        timestep = ROMSTimeStep(index_here)
        adapter = getattr(self.N4, 'source_adapter', None)
        # Only the file names change; downstream code still uses salt/temp/u/v.
        def field(logical):
            return adapter.field(logical) if adapter is not None else logical

        with self._roms_dataset(filepath) as nc:
            for var in variables:
                if var not in supported:
                    raise InputError(f'The variable: "{var}" is not supported by RomsDownloader')

            rho = (slice(self.N4.m_ri, self.N4.x_ri + 1),
                   slice(self.N4.m_rj, self.N4.x_rj + 1))
            u_slice = (slice(self.N4.m_ui, self.N4.x_ui + 1),
                       slice(self.N4.m_uj, self.N4.x_uj + 1))
            v_slice = (slice(self.N4.m_vi, self.N4.x_vi + 1),
                       slice(self.N4.m_vj, self.N4.x_vj + 1))
            level = slice(None) if sigma is None else sigma

            if 'salt' in variables:
                timestep.salt = nc[field('salt')][(index_here, level, *rho)]
            if 'temp' in variables:
                timestep.temp = nc[field('temp')][(index_here, level, *rho)]
            if 'zeta' in variables:
                timestep.zeta = nc[field('zeta')][(index_here, *rho)]
            if 'u' in variables:
                timestep.u = nc[field('u')][(index_here, slice(None), *u_slice)]
            if 'v' in variables:
                timestep.v = nc[field('v')][(index_here, slice(None), *v_slice)]

            if 'ua' in variables:
                if field('ubar') in nc.variables:
                    timestep.ua = nc[field('ubar')][(index_here, *u_slice)]
                else:
                    timestep.ua = np.sum(timestep.u * self.N4.cropped_dsigma_u, axis=0)
            if 'va' in variables:
                if field('vbar') in nc.variables:
                    timestep.va = nc[field('vbar')][(index_here, *v_slice)]
                else:
                    timestep.va = np.sum(timestep.v * self.N4.cropped_dsigma_v, axis=0)

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
        Convert source velocities to FVCOM's lat/lon or projected axes.
        '''
        adapter = getattr(self.N4, 'source_adapter', None)
        frame = adapter.velocity_frame if adapter is not None else 'grid'
        if frame == 'grid':
            # Native ROMS xi/eta components need rotation to east/north.
            timestep.u, timestep.v = self._rotate_to_latlon(timestep.u, timestep.v, self.N4.angle)
            timestep.ua, timestep.va = self._rotate_to_latlon(timestep.ua, timestep.va, self.N4.angle)
        elif frame != 'east_north':
            raise ValueError(f'Unsupported ROMS velocity frame: {frame}')

        # The hindcast already has geographic components. Both formats still
        # need geographic-to-UTM rotation when FVCOM is in a projected CRS.
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
