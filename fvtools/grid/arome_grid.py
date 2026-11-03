# Object we use to load an AROME grid, and subsequently interpolate data from the AROME grid to the FVCOM mesh
import numpy as np
from datetime import datetime
from functools import cached_property
from netCDF4 import Dataset
from abc import ABC, abstractmethod
import requests
import time as time_mod

from fvtools.util.thredds_io import retry_read, is_retryable_netcdf_error
from .common.base import BaseGrid


def _opendap_http_status(path):
    """Only probe HTTP *after* NetCDF fails; no extra requests on success.

    An OPeNDAP .dds request retrieves metadata, not numerical fields.
    Unavailable HTTP diagnostics must never be mistaken for HTTP 404.
    """
    try:
        with requests.get(f'{path}.dds', stream=True, timeout=(5, 15)) as response:
            return response.status_code
    except requests.RequestException:
        return None

def get_arome_grids(startdate, stopdate, Proj):
    # The arome grid changed feb. 5 2020
    OldAROME = None; NewAROME = None
    if startdate < datetime(2020, 2, 5):
        OldAROME = OldAromeGrid(Proj)  
        OldAROME.prepare_grid()      

    if stopdate >= datetime(2020, 2, 5):
        NewAROME = NewAromeGrid(Proj)
        NewAROME.prepare_grid()
    return OldAROME, NewAROME

class AROMEversion(ABC):
    @abstractmethod
    def test_day(self, day):
        '''
        method that returns a path to the AROME file of the day, or a NoAvailableData exception
        '''
        pass

    @abstractmethod
    def prepare_grid(self):
        '''
        Method that loads the grids and projects them to a UTM coordinate
        '''
        pass

class AROMEbase(BaseGrid):
    '''
    When dealing with old AROME data, we need to access data stored on several grids.
    '''
    @property
    def thredds_folders(self):
        '''
        Folders containing thredds data
        '''
        recent_thredds_folder   = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive'
        old_data_thredds_folder = 'https://thredds.met.no/thredds/dodsC/mepsoldarchive'
        return [recent_thredds_folder, old_data_thredds_folder]
    
    @property
    def x_center(self):
        return ((self.x[1:,1:]+self.x[1:,:-1])/2 + (self.x[:-1,1:]+self.x[:-1,:-1])/2)/2

    @property
    def y_center(self):
        return ((self.y[1:,1:]+self.y[:-1,1:])/2 + (self.y[1:,:-1]+self.y[:-1,:-1])/2)/2

    def crop_extended(self, xlim, ylim):
        """ 
        The use of this will _only_ be correct if the the pp- and extended files share the same grid, but
        one of them are a cropped version of the other!
        """
        ind1 = np.logical_and(self.xex >= xlim[0], self.xex <= xlim[1])
        ind2 = np.logical_and(self.yex >= ylim[0], self.yex <= ylim[1])
        return np.logical_and(ind1, ind2)

    def infer_day_month_year(self, date):
        self._date = date
        self.year  = str(date.year)
        self.month = '{:02d}'.format(date.month)
        self.day   = '{:02d}'.format(date.day)

    def files_in_folders(self, folders, files):
        '''
        combine list with all combinations of folders and files
        '''
        checkout_files = []
        for folder in folders:
            for file in files:
                checkout_files.append(f'{folder}{file}')
    def _test_netCDF(self, path, *, capture_time=False):
        """Open one candidate once and optionally retain its time coordinate.

        HTTP status is checked only if NetCDF fails. A confirmed 404/410 is
        missing data; access denial must never turn into NoAvailableData.
        """
        if capture_time:
            self._verified_time = None
        try:
            with Dataset(path, 'r') as ds:
                if capture_time:
                    clock = ds.variables['time']
                    self._verified_time = (
                        path, np.ma.asarray(clock[:]).copy(), clock.units)
        except OSError as exc:
            status = _opendap_http_status(path)
            if status in (404, 410):
                raise NoAvailableData(f'MetCoOp source does not exist (HTTP {status}): {path}') from exc
            if status == 403:
                raise PermissionError(f'MetCoOp THREDDS access denied (HTTP 403): {path}') from exc
            if status == 429:
                raise OSError(f'MetCoOp THREDDS rate limit (HTTP 429): {path}') from exc
            raise
        return path

    def _test_paths(self, paths, *, capture_time=False):
        """Try alternate archive files only for confirmed missing sources."""
        for path in paths:
            try:
                return retry_read(
                    lambda: self._test_netCDF(path, capture_time=capture_time),
                    max_retries=3, retry_delay=10,
                    classifier=lambda exc: (
                        not isinstance(exc, PermissionError)
                        and is_retryable_netcdf_error(exc)),
                    sleep=time_mod.sleep,
                    failure_message=f'Could not access MetCoOp source {path}',
                    log_prefix='  - MetCoOp file-list I/O error',
                )
            except NoAvailableData:
                continue
        raise NoAvailableData('No MetCoOp archive files exist for this date')

class OldAromeGrid(AROMEbase, AROMEversion):
    '''
    Object containing grid information about AROME atmospheric model grid.
    TBD: Still need to set pathToPP and pathToEX
    '''

    def __init__(self, Proj):
        """
        AROME grid for the older (messier) file system
        """
        self.Proj = Proj

    def test_day(self, day):
        '''
        Check what files are available at this moment
        '''
        # Convert datetime to day, month and year numbers
        self.infer_day_month_year(day)

        # Get path to file for non-radiation-, radiation- and accumulated fields
        file     = self._test_paths(self.files_in_folders(self.thredds_folders, [self.pp_file, self.extracted_file, self.subset_file]), capture_time=True)
        file_acc = self._test_paths(self.files_in_folders(self.thredds_folders, [self.extracted_file, self.full_backup_file]))
        file_rad = self._test_paths(self.files_in_folders(self.thredds_folders, [self.extracted_file, self.subset_file]))
        return file, file_acc, file_rad

    # Standard values for pathToPP and pathToEX will be overwritten if they are need, so this is strictly speaking not necessary
    # The set dates will be valid for some years.
    # ----
    @property
    def pathToPP(self):
        if not hasattr(self, '_pathToPP'):
            self.year = '2020'
            self.month = '01'
            self.day = '30'
            self._pathToPP = self._test_paths(self.files_in_folders(self.thredds_folders, [self.pp_file]))

        return self._pathToPP

    @pathToPP.setter
    def pathToPP(self, filepath):
        self._pathToPP = filepath

    @property
    def pathToEX(self):
        if not hasattr(self, '_pathToEX'):
            self.year = '2020'
            self.month = '01'
            self.day = '30'
            self._pathToEX = self._test_paths(self.files_in_folders(self.thredds_folders, [self.extracted_backup_file]))
        return self._pathToEX

    @pathToEX.setter
    def pathToEX(self, filepath):
        self._pathToEX = filepath

    # Standard paths
    # ----
    @property
    def pp_file(self):
        return f'/{self.year}/{self.month}/{self.day}/meps_mbr0_pp_2_5km_{self.year}{self.month}{self.day}T00Z.nc'

    @property
    def extracted_file(self):
        return f'/{self.year}/{self.month}/{self.day}/meps_mbr0_extracted_2_5km_{self.year}{self.month}{self.day}T00Z.nc'

    @property
    def extracted_backup_file(self):
        return f'/{self.year}/{self.month}/{self.day}/meps_mbr0_extracted_backup_2_5km_{self.year}{self.month}{self.day}T00Z.nc'

    @property
    def subset_file(self):
        return f'/{self.year}/{self.month}/{self.day}/meps_subset_2_5km_{self.year}{self.month}{self.day}T00Z.nc'

    @property
    def full_backup_file(self):
        return f'/{self.year}/{self.month}/{self.day}/meps_mbr0_full_backup_2_5km_{self.year}{self.month}{self.day}T00Z.nc'

    def prepare_grid(self):
        '''
        Load grid variables that we need later to prepare interpolation coefficients
        '''
        with Dataset(self.pathToPP, 'r') as ncdata:
            self.lonpp = ncdata.variables.get('longitude')[:]
            self.latpp = ncdata.variables.get('latitude')[:]

        with Dataset(self.pathToEX, 'r') as ncdata:
            self.lonex = ncdata.variables.get('longitude')[:]
            self.latex = ncdata.variables.get('latitude')[:]

        # We require that we have access to both pp and extracted files.
        self.x,   self.y   = self.Proj(self.lonpp, self.latpp, inverse=False)
        self.xex, self.yex = self.Proj(self.lonex, self.latex, inverse=False)

    @cached_property
    def land_mask(self):
        '''
        MetCoOp fractional landmask. Let "1" indicate ocean. We only use this to avoid
        land radiation to the ocean model. Should be avoided and replaced with an albedo adjustment at
        for shortwave. Not sure if longwave behaves similarly.
        '''
        with Dataset(self.pathToPP, 'r') as nc:
            _landfraction = (nc.variables.get('land_area_fraction')[:]-1.0)*-1.0
        return _landfraction

class NewAromeGrid(AROMEbase, AROMEversion):
    """
    Arome included new partners in February 2020. The computational domain was extended in the process.
    This class reads grids from that period.
    """
    def __init__(self, Proj):
        """
        Read grid coordinates from nc-files.
        A bit of a mess since pp files uses a snipped version of the extracted grid.
        """
        self.Proj = Proj

    def prepare_grid(self):
        '''
        read grid locations from netCDF file
        '''
        self.name = self.basepath.split('/')[-1].split('.')[0]
        self.na   = self.basepath.split('/')[-1].split('.')[0]
        with Dataset(self.basepath, 'r') as nc:
            self.lon, self.lat = nc.variables.get('longitude')[:], nc.variables.get('latitude')[:]
        self.x, self.y = self.Proj(self.lon, self.lat, inverse=False)

    @property
    def basepath(self):
        if not hasattr(self, '_basepath'):
            self._basepath = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive/2020/10/30/meps_det_2_5km_20201030T00Z.nc'
        return self._basepath

    @property
    def meps_file(self):
        if self._date < datetime(2023, 9, 30):
            return f'/{self.year}/{self.month}/{self.day}/meps_det_2_5km_{self.year}{self.month}{self.day}T00Z.nc'
        else:
            return f'/{self.year}/{self.month}/{self.day}/meps_det_sfc_{self.year}{self.month}{self.day}T00Z.ncml'
    
    @property
    def xex(self):
        return self.x

    @property
    def yex(self):
        return self.y

    @cached_property
    def land_mask(self):
        with Dataset(self.basepath) as nc:
            _landfraction = (nc.variables.get('land_area_fraction')[0, 0, :] - 1.0) * -1.0
        return _landfraction

    def test_day(self, day):
        '''
        Check what files are available at this moment
        '''
        # Convert datetime to day, month and year numbers
        self.infer_day_month_year(day)
        file = self._test_paths(self.files_in_folders(self.thredds_folders, [self.meps_file]), capture_time=True)
        return file, file, file

class NoAvailableData(Exception): pass