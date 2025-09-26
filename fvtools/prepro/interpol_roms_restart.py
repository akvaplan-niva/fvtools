# ----------------------------------------------------------------------------------
#              Dump data from a ROMS run to a FVCOM restart file
# ----------------------------------------------------------------------------------
import os
import pyproj
import fvtools.nesting.roms_nesting_fg as rn
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar as pb
import fvtools.nesting.vertical_interpolation as vi

from datetime import datetime, timedelta
from time import gmtime, strftime
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.roms_grid import get_roms_grid
from scipy.spatial import cKDTree as KDTree
from fvtools.grid.roms_grid import RomsDownloader
from fvtools.interpolators.roms_interpolators import N4ROMS, LinearInterpolation

import warnings
warnings.filterwarnings("ignore")

def main(restartfile, mother, uv=False, proj='epsg:32633', latlon = False):
    '''
    restartfile  - restart file formatted for FVCOM
    mother       - 'HI-NK' or 'MET-NK' for NorKyst-800
                   'H-NS' for hourly- or 'D-NS' for daily averaged NorShelf 2.4km files
    uv           - set True if you want to interpolate velocity fields to the mesh
    proj         - set projection, default: epsg:32633 (UTM33)
    '''
    # FVCOM grid object
    # ----
    ROMS = get_roms_grid(mother)

    print(f'\nInterpolate data from {ROMS} to {restartfile}\n---')
    print('- Load FVCOM restart file')
    M         = FVCOM_grid(restartfile, reference=proj)
    ROMS.Proj = M.Proj

    # Fields we will interpolate to the restart file
    # ----
    if uv:
        coords = ['rho','u','v']
        variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']

    else:
        coords    = ['rho']
        variables = ['salt', 'temp', 'zeta']

    # Load a part of the ROMS grid covering the FVCOM domain
    # ----
    ROMS.load_grid(M.x, M.y)

    # Interpolation coefficients
    # ----
    print('\nCompute interpolation coefficients')
    N4 = N4ROMSRESTART(ROMS, 
                        x = M.x, y = M.y,
                        tri = M.tri,
                        uv = uv,
                        land_check = False,
                        latlon = latlon,
                       )
    N4.nearest4()

    # Land correction (we can't use ROMS land points)
    # ---
    print('    - Adjust interpolation coefficients to remove land')
    N4.rho_index, N4.rho_coef = N4.correct_land(N4.rho_index, N4.rho_coef, ROMS.cropped_x_rho, ROMS.cropped_y_rho, M.x, M.y, ROMS.Land_rho)
    if uv:
        N4.u_index, N4.u_coef = N4.correct_land(N4.u_index, N4.u_coef, ROMS.cropped_x_u, ROMS.cropped_y_u, M.xc, M.yc, ROMS.Land_u, zero = True)
        N4.v_index, N4.v_coef = N4.correct_land(N4.v_index, N4.v_coef, ROMS.cropped_x_v, ROMS.cropped_y_v, M.xc, M.yc, ROMS.Land_v, zero = True)

    # Update FVCOM depth and ROMS depth with sea surface perturbation (i.e. tides ++) before finding vertical interpolation weights
    # ---
    N4.FV     = M
    N4.ROMS   = ROMS
    
    print('\nUpdate zeta in domain (needed for vertical interpolation)')
    Restarter = Roms2FVCOMRestart(restartfile, N4, M.tri, variables, latlon, verbose=False)
    ROMS.zeta = Restarter.z # Add actual depth to ROMS
    timestep  = Restarter.download(variables = variables)

    # Add the FVCOM grid and depth to the interpolator
    # ---
    M.zeta = np.sum(timestep.zeta[N4.rho_index] * N4.rho_coef, axis = 1)
    N4.h  = M.d  # Add sea surface perturbation to FVCOM, FVCOM_grid will now say that M.d = M.h + M.zeta
    N4.FV = M
    N4.ROMS.z = timestep.zeta

    print('\nVertical interpolation coefficients')
    N4 = vi.add_vertical_interpolation2N4(N4, coords = coords)

    print('\nInterpolate and store ROMS hydrography to the FVCOM restart file')
    Restarter = Roms2FVCOMRestart(restartfile, N4, M.tri, variables, latlon, verbose=True) #re-initializing the interpolator just to make sure that everything is ready...
    timestep = Restarter.download(variables = variables)
    Restarter.dump(timestep, variables = variables)
    print('\n- Fin.')

class N4DEPTH:
    '''
    The FVCOM depth is "not negotiable" in FVCOM restart files, thus we set it to the FVCOM depth
    '''
    @property
    def fvcom_rho_dpt(self):
        return self.h
    
    @property
    def fvcom_u_dpt(self):
        return self.hc

    @property
    def fvcom_v_dpt(self): 
        return self.hc

class N4ROMSRESTART(N4ROMS, N4DEPTH):
    def __init__(self, ROMS, x = None, y = None, tri = None, h = None, uv = False, land_check=False, latlon = False):
        '''
        Initialize empty attributes
        - h is necessary by the time this object is passed to vertical_interpolation.py
        '''
        self.x, self.y, self.tri, self.h = x, y, tri, h
        self.uv = uv
        self.latlon = latlon
        self.land_check = land_check
        self.ROMS = ROMS

    @property
    def h(self):
        '''
        fvcom watercolumn depth at nodes
        '''
        if not hasattr(self, '_h'):
            raise ValueError('You must specify h')
        return self._h
    
    @h.setter
    def h(self, var):
        '''
        fvcom watercolumn depth at cells
        '''
        self._h = var

    @property
    def hc(self):
        return np.mean(self.h[self.tri], axis=1)

    def correct_land(self, index, coef, x_source, y_source, fvcom_x, fvcom_y, LandMask, zero = False):
        '''
        Remove points completely covered by land.
        - points partially covered by land will use those values instead
        '''
        _coef, _index, _land = self._nullify_land_points(coef, index, LandMask)
        nearest_ocean = self._find_nearest_ocean_neighbor(_index, _coef, x_source, y_source, fvcom_x, fvcom_y, LandMask)

        # Overwrite indices at points completely covered by arome land
        _index[_land, :] = nearest_ocean[_land][:, None]

        # Set land weight to something
        if not zero:
            _coef[_land, :] = 0.25
        else:
            _coef[_land, :] = 0.0 # used for velocities when we deal with ROMS data in restart files

        return _index, _coef

    @staticmethod
    def _nullify_land_points(coef, index, LandMask):
        '''
        Identify land points, these must be removed for the radiation field interpolation
        '''
        # Copy to not mess up anything inadvertibly :)
        _index = np.copy(index)
        _coef  = np.copy(coef)

        # Set weight of land points to zero and re-normalize
        landbool = LandMask[_index]
        _coef[landbool] = 0
        _coef = _coef/np.sum(_coef,axis=1)[:,None]

        # Identify points completely covered by land
        _land = np.where(np.isnan(_coef[:,0]))[0]

        return _coef, _index, _land

    @staticmethod
    def _find_nearest_ocean_neighbor(_index, _coef, x_source, y_source, fvcom_x, fvcom_y, LandMask):
        '''
        replace land point with nearest ocean neighbor
        '''
        # Create a tree referencing ocean points
        ocean_tree  = KDTree(np.array([x_source[LandMask==False], y_source[LandMask==False]]).transpose())
        source_tree = KDTree(np.array([x_source, y_source]).T)

        # Nearest ocean point to all FVCOM points
        _, _nearest_ocean = ocean_tree.query(np.array([fvcom_x, fvcom_y]).transpose())
        nearest_ocean_x = x_source[LandMask==False][_nearest_ocean]
        nearest_ocean_y = y_source[LandMask==False][_nearest_ocean]

        # With same indexing as the rest of source
        _, nearest_ocean = source_tree.query(np.array([nearest_ocean_x, nearest_ocean_y]).transpose())
        return nearest_ocean

class Roms2FVCOMRestart(RomsDownloader, LinearInterpolation):
    '''
    Class writing data to restart-file.
    - Downloading ROMS data
    - Interpolating to FVCOM
    '''
    def __init__(self, restartfile, N4, tri, variables, latlon, verbose = False):
        '''
        outfile: Name of file to write to
        time:    list of timesteps to write to
        path:    list of path to file to read timestep from
        index:   list of indices to read from file for each timestep
        N4:      Class with Nearest4 interpolation coefficients
        '''
        self.restartfile = restartfile
        self.variables = variables
        self.latlon = latlon
        self.N4 = N4.dump()
        self.ROMS = N4.ROMS

        with netCDF4.Dataset(restartfile) as nc:
            self.restart_time = netCDF4.num2date(nc['time'][0], units = nc['time'].units, only_use_cftime_datetimes=False, only_use_python_datetimes=False)
        self.find_roms_to_start_from(verbose)

    @property
    def z(self):
        '''
        downloads zeta in same shape as ROMS grid
        '''
        with netCDF4.Dataset(self.path) as nc:
            z = nc['zeta'][self.index_here,:,:]
        return z

    def download(self, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']):
        '''
        Download timestep thredds
        '''
        timestep = self.read_timestep(self.index_here, self.path, variables = variables)
        timestep = self.crop_and_transpose(timestep, variables = variables)
        return timestep

    def dump(self, timestep, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']):
        '''
        Download all required fields from the thredds server
        '''
        timestep = self.horizontal_interpolation(timestep, variables = variables)
        timestep = self.vertical_interpolation(timestep, variables = variables)
        if 'u' in variables or 'v' in variables or 'ua' in variables or 'va' in variables:
            timestep = self.adjust_uv(timestep) # adjust from (xi, eta) to (lon, lat) - and then to (x,y) in utm if self.latlon=False
        self.write_to_restart(timestep, variables = variables)

    def write_to_restart(self, timestep, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']):
        '''
        Dump the fields to the restart file
        '''
        with netCDF4.Dataset(self.restartfile, 'r+') as nc:
            if 'zeta' in variables:
                nc.variables['zeta'][0,:]       = timestep.zeta
                print('  - Updated zeta')

            if 'temp' in variables:
                nc.variables['temp'][0,:,:]     = timestep.temp
                print('  - Updated temp')

            if 'salt' in variables:
                nc.variables['salinity'][0,:,:] = timestep.salt
                print('  - Updated salinity')

            if 'u' in variables:
                nc.variables['u'][0, :, :]      = timestep.u
                print('  - Updated u')

            if 'v' in variables:
                nc.variables['v'][0, :, :]      = timestep.v
                print('  - Updated v')

            if 'ua' in variables:
                nc.variables['ua'][0,:]         = timestep.ua
                print('  - Updated ua')

            if 'va' in variables:
                nc.variables['va'][0,:]         = timestep.va
                print('  - Updated va')

    def find_roms_to_start_from(self, verbose):
        '''
        Search for the relevant time and index over more than one file, in case we use a forecast- or a HI-NorKyst file
        '''
        with netCDF4.Dataset(self.restartfile) as nc:
            self._prepare_search()
            time, self.path, self.indices = rn.make_fileList(self.start_time, self.stop_time, self.ROMS)
            ind = np.where(time==nc['time'][0])

        if ind[0].size == 0:
            raise CouldNotFindRestartTime("We were not able to find a suitable restart time")

        self.path  = self.path[ind[0][0]]
        self.index_here = self.indices[ind[0][0]]

        if verbose:
            with netCDF4.Dataset(self.path) as nc:
                print(f"    - Found data for {netCDF4.num2date(nc['ocean_time'][self.index_here], units = nc['ocean_time'].units)}")

    def _prepare_search(self, days = 2):
        '''
        Search over a some days to make sure that you will find the correct restart file
        '''
        start = self.restart_time-timedelta(days = days)
        self.start_time = "{0.year}-{0.month:02}-{0.day:02}".format(start)
        self.stop_time  = "{0.year}-{0.month:02}-{0.day:02}".format(self.restart_time)

class CouldNotFindRestartTime(Exception): pass

def smooth_fields(restartfile, fields = ['ua', 'va', 'zeta', 'u', 'v'], n = 10):
    '''
    Smooth fields in the restartfile
    '''
    M = FVCOM_grid(restartfile)
    with netCDF4.Dataset(restartfile, 'r+') as restart:
        for field in fields:
            print(f'\n-{field=}')
            if len(restart[field].shape) == 2:
                restart[field][-1, :] = M.smooth(restart[field][-1, :], n = n)

            elif len(restart[field].shape) == 3:
                for s in range(restart.dimensions['siglay'].size):
                    restart[field][-1, s, :] = M.smooth(restart[field][-1, s, :], n = n)

            else:
                raise ValueError('what?')