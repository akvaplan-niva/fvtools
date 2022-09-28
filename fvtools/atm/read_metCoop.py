import sys
import os 
import pickle
import numpy as np
import netCDF4
import pandas as pd
import pyproj
import progressbar as pb
import matplotlib.pyplot as plt
import time as time_mod
import warnings
warnings.filterwarnings("ignore")

from fvtools.grid.fvcom_grd import FVCOM_grid # objects that load what we need to know about the FVCOM grid
from fvtools.grid.arome_grid import AROME_grid, newAROME_grid # objects that load arome grid data
from fvtools.interpolators.nearest4 import N4 # base-class for nearest 4 interpolation

from time import gmtime, strftime
from datetime import datetime, timedelta
from netCDF4 import Dataset
from functools import cached_property
from numba import njit
from scipy.spatial import cKDTree as KDTree

def main(grd_file, outfile, start_time, stop_time, nearest4=None):
    '''
    Create AROME atmospheric forcing file for FVCOM 

    Parameters:
    ----
    grd_file:   'M.npy' or equivalent
    outfile:     name of netcdf output
    start_time: 'yyyy-mm-dd-hh'
    stop_time:  'yyyy-mm-dd-hh'
    nearest4:   'nearest4arome.npy' file if you have already made an _atm file for this mesh (=None by default)
    '''
    print(f'Create an atmospheric forcing file')

    # initialize
    Arome_grid = None; N4A    = None
    new_Arome  = None; newN4A = None
    
    print(f'- Load {grd_file}')
    Mobj       = FVCOM_grid(grd_file)
    startnum   = start_time.split('-')
    stopnum    = stop_time.split('-')
    startdate  = datetime(int(startnum[0]), int(startnum[1]), int(startnum[2]))
    stopdate   = datetime(int(stopnum[0]), int(stopnum[1]), int(stopnum[2]))

    # The arome grid changed feb. 5 2020, we need to know both if the model period crosses feb 5th
    print('- Load AROME grid')
    if startdate < datetime(2020, 2, 5):
        Arome_grid = AROME_grid()        

    if stopdate >= datetime(2020, 2, 5):
        new_Arome  = newAROME_grid()

    print('- Create an empty output file.')
    create_nc_forcing_file(outfile, Mobj)

    print('- Make a filelist')
    time, path, path_acc, path_rad, index = metcoop_make_fileList(start_time, stop_time)

    print('\nCompute nearest4 interpolation coefficients.')
    if Arome_grid is not None:
        N4A = N4AROME(Mobj, Arome_grid)

        if nearest4 is not None:
            N4A.load_nearest4(nearest4)

        else:
            N4A.compute_nearest4()

    if new_Arome is not None:
        newN4A = N4AROME(Mobj, new_Arome)
        if nearest4 is not None:
            newN4A.load_nearest4(nearest4)
        else:
            newN4A.compute_nearest4()

    print(f'\nRead data from cloud, interpolate to FVOM mesh, prepare units in FVCOM readable format and dump data to {outfile}')
    arome2fvcom(outfile, N4A, newN4A, time, path, path_acc, path_rad, index, stopdate)

def metcoop_make_fileList(start_time, stop_time):
    '''
    Go through MetCoop thredds server and link points in time to files.
    format: yyyy-mm-dd-hh
    '''
    # AROME grid and this routine should be modified in the same way as ROMS_grid and roms_nesting_fg if we continue to use this script
    start     = datetime(int(start_time.split('-')[0]), int(start_time.split('-')[1]), int(start_time.split('-')[2]))
    stop      = datetime(int(stop_time.split('-')[0]), int(stop_time.split('-')[1]), int(stop_time.split('-')[2]))
    dates     = pd.date_range(start, stop)
    
    thredds_folder = 'https://thredds.met.no/thredds/dodsC/'
    
    # non-accumulated values
    time      = np.empty(0)
    path      = []
    path_acc  = []
    path_rad  = []
    index     = []

    # Temperature, wind and pressure
    # ----------------------------------------------------------------------------------
    accdelay  = 2    # Since we can't read the last hour in the day before, and since the first value is masked in full_backup files.

    # Number of missing dates (keeping track of this since the forecast completely covers 3 days and part of the fourth)
    missing   = 0
    for date in dates:
        year     = str(date.year)
        month    = '{:02d}'.format(date.month)
        day      = '{:02d}'.format(date.day)

        # Check that files for accumulated values are there
        # --------------------------------------------------
        if date < datetime(2020,2,5):
            try:
                file = f'{thredds_folder}meps25epsarchive/{year}/{month}/{day}/meps_mbr0_pp_2_5km_{year}{month}{day}T00Z.nc'
                d    = Dataset(file, 'r')
            
            except:
                try:
                    file = f'{thredds_folder}meps25epsarchive/{year}/{month}/{day}/meps_mbr0_extracted_2_5km_{year}{month}{day}T00Z.nc'
                    d    = Dataset(file, 'r')
                    missing = 0

                except:
                    try:
                        file = f'{thredds_folder}meps25epsarchive/{year}/{month}/{day}/meps_subset_2_5km_{year}{month}{day}T00Z.nc'
                        d    = Dataset(file, 'r')
                        missing = 0

                    except:
                        missing += 1
                        if missing == 1:
                            pass
                            
                        elif missing == 2:
                            pass
                    
                        elif missing == 3:
                            print(f'  - {date} - will not be completely covered in the forcing file!')
                        else:
                            print(f'  - {date} - is unavailable')

                        continue
    
            # check that files for non-accumulated values are there. We prefer to use extracted files, but will use full_backup files if needbe
            # ----------------------------------------------------------------------------------------------------------------------------------
            try:
                file_acc = f'{thredds_folder}meps25epsarchive/{year}{month}/{day}/meps_mbr0_extracted_2_5km_{year}{month}{day}T00Z.nc'
                d        = Dataset(file_acc, 'r')
                file_rad = file_acc
            
            except:
                try:
                    file_acc = f'{thredds_folder}meps25epsarchive/{year}/{month}/{day}/meps_mbr0_full_backup_2_5km_{year}{month}{day}T00Z.nc'
                    d        = Dataset(file_acc, 'r')

                    file_rad = f'{thredds_folder}meps25epsarchive/{year}/{month}/{day}/meps_subset_2_5km_{year}{month}{day}T00Z.nc'
                    d        = Dataset(file_rad, 'r')

                except:
                    print(f'  - {date} - not available')
                    continue

        else:
            try:
                file_rad = file_acc = file = f'{thredds_folder}meps25epsarchive/{year}/{month}/{day}/meps_det_2_5km_{year}{month}{day}T00Z.nc'
                d    = Dataset(file)

            except:
                print(f'  - {date} - not available')
                continue

        # We can't use the first two indexes due to accumulated values and NaNs at the beginning
        # of some of the files.
        # ----
        t = d.variables['time'][accdelay:-2]
        if len(d.variables['time'][:]) == 0: # Check to see if empty
            print(f'  - {date} - not available')
            continue

        time     = np.append(time, t)
        path     = path + [file]*len(t)
        path_acc = path_acc + [file_acc]*len(t)
        path_rad = path_rad + [file_rad]*len(t)
        index.extend(list(range(accdelay, len(t)+accdelay)))
        print(f'  - {date}')

    # Remove overlap
    # ----
    time_no_overlap     = [time[-1]]
    path_no_overlap     = [path[-1]]
    path_acc_no_overlap = [path_acc[-1]]
    path_rad_no_overlap = [path_rad[-1]]
    index_no_overlap    = [index[-1]]

    for n in range(len(time)-1, 0, -1):
        if time[n-1] < time_no_overlap[0]:
            time_no_overlap.insert(0, time[n-1])
            path_no_overlap.insert(0, path[n-1])
            path_acc_no_overlap.insert(0, path_acc[n-1])
            path_rad_no_overlap.insert(0, path_rad[n-1])
            index_no_overlap.insert(0, index[n-1])

    return np.array(time_no_overlap), path_no_overlap, path_acc_no_overlap, path_rad_no_overlap, index_no_overlap


class N4AROME(N4):
    '''
    Object with indices and coefficients for AROME to FVCOM interpolation
    - note: radiation fields will only use ocean points, if land in radiation square, we will mask it.
            squares with full-land coverage will use nearest ocean neighbor
    '''
    def __init__(self, FVCOM_grd, AROME_grd):
        self.FVCOM_grd = FVCOM_grd
        self.AROME_grd = AROME_grd
        self.ncoef  = np.empty([len(self.FVCOM_grd.x), 4])
        self.nindex = np.empty([len(self.FVCOM_grd.x), 4])
        self.ccoef  = np.empty([len(self.FVCOM_grd.xc), 4])
        self.cindex = np.empty([len(self.FVCOM_grd.xc), 4])

    def load_nearest4(self, infile):
        '''
        Load an already computed nearest4.
        '''
        nearest4 = np.load(infile, allow_pickle=True).item()
        for field in ['nindex', 'ncoef', 'cindex', 'ccoef', 'nindex_rad', 'ncoef_rad', 'cindex_rad', 'ccoef_rad']:
            setattr(self, field, nearest4[field])

    def compute_nearest4(self, nearest4filename = 'nearest4arome.npy'):
        '''
        Create nearest four indices and weights for all of the fields
        '''
        assert all(self.x_arome == self.AROME_grd.xex[self.fv_domain_mask_ex]), f"(parts) of your FVCOM grid seems to be outside of the AROME domain"

        print('  - Find cluster of 4 surrounding')
        _4node_inds, _4cell_inds = self._find_clusters_of_4_points()      # find the nearest 4 points to this FVCOM node

        print('  - Compute interpolation coefficients')
        self.nindex, self.ncoef = self._compute_interpolation_coefficients(
                                                self.FVCOM_grd.x, self.FVCOM_grd.y, 
                                                _4node_inds,
                                                self.arome_tree.data, 
                                                self.nindex, self.ncoef,
                                                widget_title = 'node'
                                                )

        self.cindex, self.ccoef = self._compute_interpolation_coefficients(
                                                self.FVCOM_grd.xc, self.FVCOM_grd.yc, 
                                                _4cell_inds,
                                                self.arome_tree.data, 
                                                self.cindex, self.ccoef,
                                                widget_title = 'cell'
                                                )

        print('  - Adjust radiation interpolation to avoid using land points')
        self.nindex_rad, self.ncoef_rad = self._remove_land_points(self.nindex, self.ncoef)
        self.cindex_rad, self.ccoef_rad = self._remove_land_points(self.cindex, self.ccoef)

        print(f'  - Save coefficients and indices for later as {nearest4filename}')
        out = {}
        for field in ['nindex', 'ncoef', 'cindex', 'ccoef', 'nindex_rad', 'ncoef_rad', 'cindex_rad', 'ccoef_rad']:
            out[field] = getattr(self, field)
        np.save(nearest4filename, out)

    @cached_property
    def fv_domain_mask(self):
        '''
        Reduces the number of AROME points we take in to consideration when computing the interpolation coefficients
        '''
        return self.AROME_grd.crop_grid(xlim=[self.FVCOM_grd.x.min() - 3000, self.FVCOM_grd.x.max() + 3000],
                                        ylim=[self.FVCOM_grd.y.min() - 3000, self.FVCOM_grd.y.max() + 3000])

    @cached_property
    def fv_domain_mask_ex(self):
        '''
        crops the ex-domain to the smaller arome domain found in other files
        '''
        return self.AROME_grd.crop_extended(xlim=[self.FVCOM_grd.x.min() - 3000, self.FVCOM_grd.x.max() + 3000],
                                            ylim=[self.FVCOM_grd.y.min() - 3000, self.FVCOM_grd.y.max() + 3000])

    @cached_property
    def xy_center_mask(self):
        '''
        mask centre-poitns (used to find nearest4) to the fv_domain_mask
        '''
        return np.logical_and(self.fv_domain_mask[1:, 1:], self.fv_domain_mask[:-1, :-1])

    @property
    def LandMask(self):
        '''
        1 = ocean
        0 = land
        '''
        return self.AROME_grd.landmask[self.fv_domain_mask].astype(int)
    
    @property
    def x_arome(self):
        return self.AROME_grd.x[self.fv_domain_mask]
    
    @property
    def y_arome(self):
        return self.AROME_grd.y[self.fv_domain_mask]

    @property
    def x_arome_center(self):
        return self.AROME_grd.x_center[self.xy_center_mask]
    
    @property
    def y_arome_center(self):
        return self.AROME_grd.y_center[self.xy_center_mask]

    @property
    def fv_nodes(self):
        return np.array([self.FVCOM_grd.x, self.FVCOM_grd.y]).transpose()

    @property
    def fv_cells(self):
        return np.array([self.FVCOM_grd.xc, self.FVCOM_grd.yc]).transpose()

    @property
    def ball_radius(self):
        '''
        We make use of the staggering of the source grid to find clusters of 4 indices covering each FVCOM point
        '''
        dst = np.sqrt((self.x_arome - self.x_arome_center[0])**2 + (self.y_arome - self.y_arome_center[0])**2)
        return 1.3*dst[dst.argsort()[0]] # 1.3 was arbitrary, but turns ot to make sure that we only find the points we need

    @property
    def arome_tree(self):
        return KDTree(np.array([self.x_arome, self.y_arome]).transpose())

    @property
    def arome_centre_tree(self):
        return KDTree(np.array([self.x_arome_center, self.y_arome_center]).transpose())

    @property
    def fv2arome_node(self):
        '''
        Figure out which square each FVCOM point should be connected to (centre points is at the centre of squares...)
        '''
        _, fv2arome_node = self.arome_centre_tree.query(self.fv_nodes)
        return fv2arome_node

    @property
    def fv2arome_cell(self):
        _, _fv2arome_cell = self.arome_centre_tree.query(self.fv_cells)
        return _fv2arome_cell

    def _find_clusters_of_4_points(self):
        '''
        Find the 4 source points around each centre-point
        '''
        _arome_node_inds = self.arome_tree.query_ball_point(self.arome_centre_tree.data[self.fv2arome_node], r = self.ball_radius)
        _arome_cell_inds = self.arome_tree.query_ball_point(self.arome_centre_tree.data[self.fv2arome_cell], r = self.ball_radius)
        return _arome_node_inds, _arome_cell_inds

    def _remove_land_points(self, index, coef):
        '''
        Identify where we need to remove land
        '''
        newindex = np.copy(index)
        newcoef  = np.copy(coef)

        # 1. set weight of land points to zero
        landbool = self.LandMask[newindex]==0
        newcoef[landbool] = 0

        # 2. re-normalize
        newcoef = newcoef/np.sum(newcoef,axis=1)[:,None]

        # 3. find the nearest neighbour where all points are landpoints
        points_on_arome_land = np.where(np.isnan(newcoef[:,0]))[0]  # points completely covered by arome land
        nearest_arome_ocean  = self._find_nearest_ocean_neighbor(index, coef)

        # 4. Overwrite indices at points completely covered by arome land
        newindex[points_on_arome_land, :] = nearest_arome_ocean[points_on_arome_land][:,None]
        newcoef[points_on_arome_land, :]  = 0.25 # just weight the same point = 0.25 for 4 times

        return newindex, newcoef

    def _find_nearest_ocean_neighbor(self, index, coef):
        '''
        replace land point with nearest ocean neighbor
        '''
        # Points we need to change
        fvcom_x = np.sum(self.x_arome[index]*coef, axis=1)
        fvcom_y = np.sum(self.y_arome[index]*coef, axis=1)

        # Create a tree referencing ocean points
        ocean_tree = KDTree(np.array([self.x_arome[self.LandMask==1], self.y_arome[self.LandMask==1]]).transpose())

        # Nearest ocean point
        _, _nearest_ocean = ocean_tree.query(np.array([fvcom_x, fvcom_y]).transpose())
        nearest_ocean_x = self.x_arome[self.LandMask==1][_nearest_ocean]
        nearest_ocean_y = self.y_arome[self.LandMask==1][_nearest_ocean]

        # With same indexing as the rest of AROME
        _, nearest_arome_ocean = self.arome_tree.query(np.array([nearest_ocean_x, nearest_ocean_y]).transpose())
        return nearest_arome_ocean

def arome2fvcom(outfile, oldNearest4, newNearest4, arome_time, path, path_acc, path_rad, index, stop): 
    '''
    Read AROME-data, interpolate to FVCOM-grid and export to nc-file.

    Due to the messy nature of MetCoOp data storage, we have to do this in two steps.
    The first loop handles data that do not have NaNs in their first column, while
    the second handles data where we use the forecast 2 hours into the next day to be
    able to give a good estimate of the average based on the accumulated value.
    '''
    print('- Interpolating MetCoOp data: ')
    dates        = netCDF4.num2date(arome_time, 'seconds since 1970-01-01 00:00:00')
    out          = Dataset(outfile, 'r+', format = 'NETCDF4')

    # Loop through file list and add data to nc-file
    first_time   = 1
    already_read = ' '
    # -----------------------------------------------------------------------------------------
    for counter, (time, dtime, path, path_acc, path_rad, index) in enumerate(zip(arome_time, dates, path, path_acc, path_rad, index)):
        if dtime > stop:
            print('Finished.')
            break

        if path.find('_det_') > 0:
            Nearest4 = newNearest4

        else:
            Nearest4 = oldNearest4
    
        if first_time:
            out.variables['lat'][:]  = Nearest4.FVCOM_grd.y
            out.variables['lon'][:]  = Nearest4.FVCOM_grd.x
            out.variables['x'][:]    = Nearest4.FVCOM_grd.x
            out.variables['y'][:]    = Nearest4.FVCOM_grd.y
            out.variables['latc'][:] = Nearest4.FVCOM_grd.yc
            out.variables['lonc'][:] = Nearest4.FVCOM_grd.xc
            out.variables['yc'][:]   = Nearest4.FVCOM_grd.yc
            out.variables['xc'][:]   = Nearest4.FVCOM_grd.xc
            tris                     = Nearest4.FVCOM_grd.tri+1
            out.variables['nv'][:]   = tris.transpose()

        # Data at thredds is sometimes unavailable. We wait for it to come back online.
        # ----
        unavailable = True
        while unavailable:
            try:
                if first_time:
                    fvcom_time, SwD, andLw, evap, precip, caf, relative_humidity, spechum, Pair, Tair, Vwind, Uwind, nc, nc_a, nc_r, already_read =\
                    read_data(Nearest4, time, dtime, path, path_acc, path_rad, index, already_read)
                    first_time  = 0
                    unavailable = False

                else:
                    fvcom_time, SwD, andLw, evap, precip, caf, relative_humidity, spechum, Pair, Tair, Vwind, Uwind, nc, nc_a, nc_r, already_read =\
                    read_data(Nearest4, time, dtime, path, path_acc, path_rad, index, already_read, nc = nc, nc_a = nc_a, nc_r = nc_r)
                    unavailable = False

            except:
                print('\n--------------------------------------\n '+\
                      'The data is unavailable at the moment.\n '+\
                      'We wait five minutes and try again.'+\
                      '\n--------------------------------------\n')
                time_mod.sleep(300)
                already_read     = ' '
                first_time       = 1

        # Interpolation to FVCOM grid after data is read
        # ----
        caf               = np.sum(caf[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        Pair              = np.sum(Pair[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        relative_humidity = np.sum(relative_humidity[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        spechum           = np.sum(spechum[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        Tair              = np.sum(Tair[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        Vwind             = np.sum(Vwind[Nearest4.cindex] * Nearest4.ccoef, axis=1)
        Uwind             = np.sum(Uwind[Nearest4.cindex] * Nearest4.ccoef, axis=1)
        evap              = np.sum(evap[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        precip            = np.sum(precip[Nearest4.nindex] * Nearest4.ncoef, axis=1)

        # Radiation is gathered from closest ocean points, hence own interpolation method
        andLw             = np.sum(andLw[Nearest4.nindex_rad] * Nearest4.ncoef_rad, axis=1)
        SwD               = np.sum(SwD[Nearest4.nindex_rad] * Nearest4.ncoef_rad, axis=1)

        # Store the data
        # - Consult Qin: Do we need the duplicates of temperature and wind speeds anymore? FVCOM3/4 and FABM changes...
        out.variables['time'][counter]                 = fvcom_time
        out.variables['Itime'][counter]                = np.floor(fvcom_time)
        out.variables['Itime2'][counter]               = (fvcom_time - np.floor(fvcom_time)) * 24 * 60 * 60 * 1000
        out.variables['short_wave'][counter, :]        = SwD
        out.variables['long_wave'][counter, :]         = andLw
        out.variables['evap'][counter, :]              = evap
        out.variables['precip'][counter, :]            = precip
        out.variables['cloud_cover'][counter, :]       = caf
        out.variables['relative_humidity'][counter, :] = relative_humidity*100.0 # from kg/kg fraction to percentage
        out.variables['air_pressure'][counter, :]      = Pair
        out.variables['SAT'][counter, :]               = Tair-273.15 # Kelvin to C
        out.variables['air_temperature'][counter, :]   = Tair-273.15 # Kelvin to C
        out.variables['SPQ'][counter, :]               = spechum   # decimal fraction kg/kg
        out.variables['V10'][counter, :]               = Vwind
        out.variables['vwind_speed'][counter, :]       = Vwind
        out.variables['U10'][counter, :]               = Uwind
        out.variables['uwind_speed'][counter, :]       = Uwind

    # Close netCDF handles
    out.close()
    try:
        nc.close()
        nc_r.close()
        nc_a.close()
    except:
        pass

def read_data(Nearest4, time, dtime, path, path_acc, path_rad, index, already_read, nc = ' ', nc_a = ' ', nc_r = ' '):
    '''
    Routine that reads raw meps data covering the the model domain.
    '''
    if path != already_read:
        print(' ')
        print('  Reading data from:')
        print(f'   - {path}')
        try:
            nc.close()
        except:
            pass
        nc   = Dataset(path,'r')
        
        # No need to download the same thing twice
        # ----
        if path.find('_det_')==-1:
            if path != path_acc:
                try:
                    nc_a.close()
                except:
                    pass
                print(f'   - {path_acc}')
                nc_a = Dataset(path_acc,'r')
            else:
                nc_a = nc

            if path != path_rad:
                print(f'   - {path_rad}')
                try:
                    nc_r.close()
                except:
                    pass
                nc_r = Dataset(path_rad,'r')
            else:
                nc_r = nc

        else:
            nc_a = ' '
            nc_r = ' '

    print(f'  timestep: {dtime}')

    already_read          = path
    fvcom_time            = netCDF4.date2num(dtime, units = 'days since 1858-11-17 00:00:00')

    # Indices to be read by accumulated value files
    read_inds             = [index-1,index]

    if path.find('_pp_') >= 0:
        Uwind             = nc.variables.get('x_wind_10m')[index, :, :][Nearest4.fv_domain_mask]
        Vwind             = nc.variables.get('y_wind_10m')[index, :, :][Nearest4.fv_domain_mask]
        Tair              = nc.variables.get('air_temperature_2m')[index, 0, :, :][Nearest4.fv_domain_mask]
        Pair              = nc.variables.get('air_pressure_at_sea_level')[index, 0, :, :][Nearest4.fv_domain_mask]
        relative_humidity = nc.variables.get('relative_humidity_2m')[index, 0, :, :][Nearest4.fv_domain_mask]
        caf               = nc.variables.get('cloud_area_fraction')[index, :, :][Nearest4.fv_domain_mask]*0.01 # convert from percent to fraction
        precip            = nc.variables.get('precipitation_amount_acc')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask]

    elif path.find('extracted') >= 0:
        Uwind             = nc.variables.get('x_wind_10m')[index, 0, :, :][Nearest4.fv_domain_mask_ex]
        Vwind             = nc.variables.get('y_wind_10m')[index, 0, :, :][Nearest4.fv_domain_mask_ex]
        Tair              = nc.variables.get('air_temperature_2m')[index, 0, :, :][Nearest4.fv_domain_mask_ex]
        Pair              = nc.variables.get('air_pressure_at_sea_level')[index, 0, :, :][Nearest4.fv_domain_mask_ex]
        relative_humidity = nc.variables.get('relative_humidity_2m')[index, 0, :, :][Nearest4.fv_domain_mask_ex]
        caf               = nc.variables.get('cloud_area_fraction')[index, 0, :, :][Nearest4.fv_domain_mask_ex]
        precip            = nc.variables.get('precipitation_amount_acc')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask_ex]

    elif path.find('subset') >= 0:
        Uwind             = nc.variables.get('x_wind_10m')[index, 0, 0, :, :][Nearest4.fv_domain_mask_ex]
        Vwind             = nc.variables.get('y_wind_10m')[index, 0, 0, :, :][Nearest4.fv_domain_mask_ex]
        Tair              = nc.variables.get('air_temperature_2m')[index, 0, 0, :, :][Nearest4.fv_domain_mask_ex]
        Pair              = nc.variables.get('air_pressure_at_sea_level')[index, 0, 0, :, :][Nearest4.fv_domain_mask_ex]
        relative_humidity = nc.variables.get('relative_humidity_2m')[index, 0, 0, :, :][Nearest4.fv_domain_mask_ex]
        caf               = nc.variables.get('cloud_area_fraction')[index, 0, 0, :, :][Nearest4.fv_domain_mask_ex]
        precip            = nc.variables.get('precipitation_amount_acc')[read_inds, 0, 0, :, :][:,Nearest4.fv_domain_mask_ex]

    elif path.find('_det_') >= 0:
        Uwind             = nc.variables.get('x_wind_10m')[index, 0, :, :][Nearest4.fv_domain_mask]
        Vwind             = nc.variables.get('y_wind_10m')[index, 0, :, :][Nearest4.fv_domain_mask]
        Tair              = nc.variables.get('air_temperature_2m')[index, 0, :, :][Nearest4.fv_domain_mask]
        Pair              = nc.variables.get('air_pressure_at_sea_level')[index, 0, :, :][Nearest4.fv_domain_mask]
        relative_humidity = nc.variables.get('relative_humidity_2m')[index, 0, :, :][Nearest4.fv_domain_mask]
        caf               = nc.variables.get('cloud_area_fraction')[index, 0, :, :][Nearest4.fv_domain_mask]
        precip            = nc.variables.get('precipitation_amount_acc')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask]

    # Switching to _full_backup_ and _subset_ if extended is not available
    if path_acc.find('_full_') >= 0:
        # Load data centered above this timestep
        evap    = nc_a.variables.get('water_evaporation_amount')[read_inds, 0, 0, :, :][:, Nearest4.fv_domain_mask_ex]
        spechum = nc_a.variables.get('specific_humidity_2m')[index, 0, 0, :, :][Nearest4.fv_domain_mask_ex]
        SwD     = nc_r.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, 0, :, :][:,Nearest4.fv_domain_mask_ex]
        andLw   = nc_r.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, 0, :, :][:,Nearest4.fv_domain_mask_ex]

    elif path_acc.find('_det_') >= 0:
        evap    = nc.variables.get('water_evaporation_amount')[read_inds, 0, :, :][:, Nearest4.fv_domain_mask]
        spechum = nc.variables.get('specific_humidity_2m')[index, 0, :, :][Nearest4.fv_domain_mask]
        SwD     = nc.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask]
        andLw   = nc.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask]

    else:
        # Load data centered above this timestep
        # full_backup
        evap    = nc_a.variables.get('water_evaporation_amount')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask_ex]
        spechum = nc_a.variables.get('specific_humidity_2m')[index, 0, :, :][Nearest4.fv_domain_mask_ex]

        if path_rad.find('_extracted_') >= 0:
            SwD     = nc_r.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask_ex]
            andLw   = nc_r.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask_ex]
        else:
            SwD     = nc_r.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, 0, :, :][:,Nearest4.fv_domain_mask_ex]
            andLw   = nc_r.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, 0, :, :][:,Nearest4.fv_domain_mask_ex]

    # "remove accumulation"
    SwD     = (SwD[1,:]    - SwD[0,:])/(3600.0)
    andLw   = (andLw[1,:]  - andLw[0,:])/(3600.0)

    evap    = (evap[1,:] - evap[0,:])/(1000.0*3600.0)     # from (kg m-2) accumulated over 1h to m s-1
    precip  = (precip[1,:] - precip[0,:])/(1000.0*3600.0) # from (kg m-2) accumulated over 1h to m s-1

    return fvcom_time, SwD, andLw, evap, precip, caf, relative_humidity, spechum, Pair, Tair, Vwind, Uwind, nc, nc_a, nc_r, already_read

def create_nc_forcing_file(name, FVCOM_grd):
    '''
    Creates empty nc file formatted to fit fvcom atm forcing
    '''
    nc = Dataset(name, 'w', format='NETCDF4')

    # Write global attributes
    nc.title       = 'FVCOM Forcing File'
    nc.institution = 'Akvaplan-niva AS'
    nc.source      = 'FVCOM grid (unstructured) surface forcing'
    nc.created     = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # Create dimensions
    nc.createDimension('time', 0)
    nc.createDimension('node', len(FVCOM_grd.x))
    nc.createDimension('nele', len(FVCOM_grd.xc))
    nc.createDimension('three', 3)

    # Create variables and variable attributes
    # ----------------------------------------------------------
    time               = nc.createVariable('time', 'single', ('time',))
    time.units         = 'days since 1858-11-17 00:00:00'
    time.format        = 'modified julian day (MJD)'
    time.time_zone     = 'UTC'

    Itime              = nc.createVariable('Itime', 'int32', ('time',))
    Itime.units        = 'days since 1858-11-17 00:00:00'
    Itime.format       = 'modified julian day (MJD)'
    Itime.time_zone    = 'UTC'

    Itime2             = nc.createVariable('Itime2', 'int32', ('time',))
    Itime2.units       = 'msec since 00:00:00'
    Itime2.time_zone   = 'UTC'

    lat                = nc.createVariable('lat', 'single', ('node',))
    lat.long_name      = 'nodal latitude'
    lat.units          = 'degrees'

    latc               = nc.createVariable('latc', 'single', ('nele',))
    latc.long_name     = 'elemental latitude'
    latc.units         = 'degrees'

    lon                = nc.createVariable('lon', 'single', ('node',))
    lon.long_name      = 'nodal longitude'
    lon.units          = 'degrees'

    lonc               = nc.createVariable('lonc', 'single', ('nele',))
    lonc.long_name     = 'elemental longitude'
    lonc.units         = 'degrees'

    x                  = nc.createVariable('x', 'single', ('node',))
    x.long_name        = 'nodal x-position'
    x.units            = 'm'
    
    y                  = nc.createVariable('y', 'single', ('node',))
    y.long_name        = 'nodal x-position'
    y.units            = 'm'

    xc                 = nc.createVariable('xc', 'single', ('nele',))
    xc.long_name       = 'nodal x-position'
    xc.units           = 'm'
    
    yc                  = nc.createVariable('yc', 'single', ('nele',))
    yc.long_name       = 'nodal x-position'
    yc.units           = 'm'

    nv                 = nc.createVariable('nv', 'int32', ('three', 'nele',))
    nv.long_name       = 'nodes surrounding elements'

    # ---------------------------------------------------------------
    precip             = nc.createVariable('precip', 'single', ('time', 'node',))
    precip.long_name   = 'Precipitation'
    precip.description = 'Precipitation, ocean lose water if negative'
    precip.units       = 'm s-1'
    precip.grid        = 'fvcom_grid'
    precip.coordinates = ''
    precip.type        = 'data'

    evap               = nc.createVariable('evap', 'single', ('time', 'node',))
    evap.long_name     = 'Evaporation'
    evap.description   = 'Evaporation, ocean lose water is negative'
    evap.units         = 'm s-1'
    evap.grid          = 'fvcom_grid'
    evap.coordinates   = ''
    evap.type          = 'data'

    relative_humidity  = nc.createVariable('relative_humidity', 'single', ('time', 'node',))
    relative_humidity.long_name   = 'Relative Humidity'
    relative_humidity.units       = '%'
    relative_humidity.grid        = 'fvcom_grid'
    relative_humidity.coordinates = ''
    relative_humidity.type        = 'data'

    specific_humidity  = nc.createVariable('SPQ', 'single', ('time', 'node',))
    specific_humidity.long_name   = 'Specific Humidity'
    specific_humidity.units       = 'kg/kg'
    specific_humidity.grid        = 'fvcom_grid'
    specific_humidity.coordinates = ''
    specific_humidity.type        = 'data'

    long_wave              = nc.createVariable('long_wave', 'single', ('time', 'node',))
    long_wave.long_name    = 'Long Wave Radiation'
    long_wave.units        = 'W m-2'
    long_wave.grid         = 'fvcom_grid'
    long_wave.coordinates  = ''
    long_wave.type         = 'data'

    short_wave             = nc.createVariable('short_wave', 'single', ('time', 'node',))
    short_wave.long_name   = 'Short Wave Radiation'
    short_wave.units       = 'W m-2'
    short_wave.grid        = 'fvcom_grid'
    short_wave.coordinates = ''
    short_wave.type        = 'data'

    cloud_area_fraction    = nc.createVariable('cloud_cover', 'single', ('time', 'node',))
    cloud_area_fraction.long_name   = 'Cloud Area Fraction'
    cloud_area_fraction.units       = 'cloud covered fraction of sky [0,1]'
    cloud_area_fraction.grid        = 'fvcom_grid'
    cloud_area_fraction.coordinates = ''
    cloud_area_fraction.type        = 'data'

    air_pressure = nc.createVariable('air_pressure', 'single', ('time', 'node',))
    air_pressure.long_name   = 'Surface Air Pressure'
    air_pressure.units       = 'Pa'
    air_pressure.grid        = 'fvcom_grid'
    air_pressure.coordinates = ''
    air_pressure.type        = 'data'

    SAT = nc.createVariable('SAT', 'single', ('time', 'node',))
    SAT.long_name            = 'Sea surface air temperature'
    SAT.units                = 'Degree (C)'
    SAT.grid                 = 'fvcom_grid'
    SAT.coordinates          = ''
    SAT.type                 = 'data'

    air_temp = nc.createVariable('air_temperature', 'single', ('time', 'node',))
    air_temp.long_name       = 'Sea surface air temperature'
    air_temp.units           = 'Degree (C)'
    air_temp.grid            = 'fvcom_grid'
    air_temp.coordinates     = ''
    air_temp.type            = 'data'

    U10 = nc.createVariable('U10', 'single', ('time', 'nele',))
    U10.long_name            = 'Eastward Wind Speed'
    U10.units                = 'm/s'
    U10.grid                 = 'fvcom_grid'
    U10.coordinates          = ''
    U10.type                 = 'data'

    uwind_speed = nc.createVariable('uwind_speed', 'single', ('time', 'nele',))
    uwind_speed.long_name    = 'Eastward Wind Speed'
    uwind_speed.units        = 'm/s'
    uwind_speed.grid         = 'fvcom_grid'
    uwind_speed.coordinates  = ''
    uwind_speed.type         = 'data'

    V10 = nc.createVariable('V10', 'single', ('time', 'nele',))
    V10.long_name            = 'Northward Wind Speed'
    V10.units                = 'm/s'
    V10.grid                 = 'fvcom_grid'
    V10.coordinates          = ''
    V10.type                 = 'data'

    vwind_speed = nc.createVariable('vwind_speed', 'single', ('time', 'nele',))
    vwind_speed.long_name    = 'Northward Wind Speed'
    vwind_speed.units        = 'm/s'
    vwind_speed.grid         = 'fvcom_grid'
    vwind_speed.coordinates  = ''
    vwind_speed.type         = 'data'

    nc.close()