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

from fvcom_pytools.grid.fvcom_grd import FVCOM_grid
from time import gmtime, strftime
from datetime import datetime, timedelta
from netCDF4 import Dataset

# Routines for extracting AROME atmospheric data from the met OPeNDAP server, extracting
# the part of the data covering the model domain, interpolating these data to the FVCOM
# grid and finally storing it in a FVCOM friendly netCDF forcing file.

# See the MetCoOp Forcing manual for instructions on how to use this routine.

def main(grd_file, outfile, start_time, stop_time):
    '''
    Create AROME atmospheric forcing file for FVCOM 

    Parameters:
    ----
    grd_file:   'M.npy' or equivalent
    outfile:     name of netcdf output
    start_time: 'yyyy-mm-dd-hh'
    stop_time:  'yyyy-mm-dd-hh'
    '''
    # initialize
    Arome_grid = None; N4    = None
    new_Arome  = None; newN4 = None
    
    Mobj           = FVCOM_grid(grd_file)
    startnum       = start_time.split('-')
    stopnum        = stop_time.split('-')
    startdate      = datetime(int(startnum[0]), int(startnum[1]), int(startnum[2]))
    stopdate       = datetime(int(stopnum[0]), int(stopnum[1]), int(stopnum[2]))
    if startdate < datetime(2020, 2, 5):
        Arome_grid = AROME_grid()        

    if stopdate >= datetime(2020, 2, 5):
        new_Arome  = new_AROME_grid()

    print('\nCreating empty output file.')
    create_nc_forcing_file(outfile, Mobj)

    print('\nCreating file-list')
    time, path, path_acc, path_rad, index = metcoop_make_fileList(start_time, stop_time)

    print('\nCreating nearest4 interpolation coefficients.')
    if Arome_grid is not None:
        if new_Arome is not None:
            print('Old Arome grid')
        N4 = nearest4(Mobj, Arome_grid)

    if new_Arome is not None:
        if Arome_grid is not None:
            print('New Arome grid')
        newN4 = nearest4(Mobj, new_Arome)

    print('\nRead data from cloud and dump non-accumulated data to file\n')
    arome2fvcom(outfile, N4, newN4, time, path, path_acc, path_rad, index, stopdate)

class AROME_grid():
    '''
    Object containing grid information about AROME atmospheric model grid.
    '''

    def __init__(self):
        """
        Read grid coordinates from nc-files.
        A bit of a mess since pp files uses a snipped version of the extracted grid.
        """
        # Assuming that the grid is unchanged, will probably lead to crashes in the future - but those are
        # problems we need to think about anyways, and in such we welcome them...
        # --------------
        year     = '2020'#str(date.year)
        month    = '01'#'{:02d}'.format(date.month)
        day      = '30'#'{:02d}'.format(date.day)
        pathToEX = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive/' + year + '/' + month + '/' +\
                   day + '/meps_mbr0_extracted_backup_2_5km_' + year + month + day + 'T00Z.nc'

        

        pathToPP = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive/' + year + '/' + month + '/' +\
                   day + '/meps_mbr0_pp_2_5km_' + year + month + day + 'T00Z.nc'

        self.ncpp  = pathToPP
        self.ncex  = pathToEX
        self.name  = pathToPP.split('/')[-1].split('.')[0]
        self.naex  = pathToEX.split('/')[-1].split('.')[0]

        ncdata     = Dataset(pathToPP, 'r')
        self.lonpp = ncdata.variables.get('longitude')[:]
        self.latpp = ncdata.variables.get('latitude')[:]

        # MetCoOp fractional landmask (from pp file). Let "1" indicate ocean. We only use this to avoid
        # land radiation to the ocean model. Should be avoided and replaced with an albedo adjustment at
        # for shortwave. Not sure if longwave behaves similarly.
        self.landmask = (ncdata.variables.get('land_area_fraction')[:]-1.0)*-1.0

        ncdata.close()

        ncdata     = Dataset(pathToEX, 'r')
        self.lonex = ncdata.variables.get('longitude')[:]
        self.latex = ncdata.variables.get('latitude')[:]
        ncdata.close()

        UTM33W = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
        self.x, self.y = UTM33W(self.lonpp, self.latpp, inverse=False)
        self.xex, self.yex = UTM33W(self.lonex, self.latex, inverse=False)

    def crop_grid(self, xlim, ylim):
        """Find indices of grid points inside specified domain"""

        ind1 = np.logical_and(self.x >= xlim[0], self.x <= xlim[1])
        ind2 = np.logical_and(self.y >= ylim[0], self.y <= ylim[1])
        return np.logical_and(ind1, ind2)

    def crop_extended(self,xlim,ylim):
        """ 
        The use of this will _only_ be correct if the the pp- and extended files share the same grid, but
        one of them are a cropped version of the other!
        """
        ind1 = np.logical_and(self.xex >= xlim[0], self.xex <= xlim[1])
        ind2 = np.logical_and(self.yex >= ylim[0], self.yex <= ylim[1])
        return np.logical_and(ind1, ind2)

class new_AROME_grid():
    """
    Arome included new partners in February 2020. The computational domain was extended in the process.
    This class reads grids from that period.
    """
    def __init__(self):
        """
        Read grid coordinates from nc-files.
        A bit of a mess since pp files uses a snipped version of the extracted grid.
        """
        # Assuming that the grid is unchanged, will probably lead to crashes in the future - but those are
        # problems we need to think about anyways, and in such we welcome them...
        # --------------

        self.nc    = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive/2020/10/30/meps_det_2_5km_20201030T00Z.nc'
        self.name  = self.nc.split('/')[-1].split('.')[0]
        self.na    = self.nc.split('/')[-1].split('.')[0]

        ncdata     = Dataset(self.nc, 'r')

        # MetCoOp fractional landmask. Let "1" indicate ocean. We only use this to avoid
        # land radiation to the ocean model. Should be avoided and replaced with an albedo adjustment at
        # for shortwave. Not sure if longwave behaves similarly.
        self.landmask = (ncdata.variables.get('land_area_fraction')[0,0,:]-1.0)*-1.0

        ncdata   = Dataset(self.nc, 'r')
        self.lon = ncdata.variables.get('longitude')[:]
        self.lat = ncdata.variables.get('latitude')[:]
        ncdata.close()

        UTM33W = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
        self.x, self.y     = UTM33W(self.lon, self.lat, inverse=False)

        # To work together with the rest of the routine
        self.xex = np.copy(self.x)
        self.yex = np.copy(self.y)

    def crop_grid(self, xlim, ylim):
        """
        Find indices of grid points inside specified domain
        """

        ind1 = np.logical_and(self.x >= xlim[0], self.x <= xlim[1])
        ind2 = np.logical_and(self.y >= ylim[0], self.y <= ylim[1])
        return np.logical_and(ind1, ind2)

    def crop_extended(self,xlim,ylim):
        """ 
        The use of this will _only_ be correct if the the pp- and extended files share the same grid, but
        one of them are a cropped version of the other!
        """
        ind1 = np.logical_and(self.xex >= xlim[0], self.xex <= xlim[1])
        ind2 = np.logical_and(self.yex >= ylim[0], self.yex <= ylim[1])
        return np.logical_and(ind1, ind2)

def metcoop_make_fileList(start_time, stop_time):
    '''
    Go through MetCoop thredds server and link points in time to files.
    format: yyyy-mm-dd-hh
    '''
    
    start     = datetime(int(start_time.split('-')[0]), int(start_time.split('-')[1]), int(start_time.split('-')[2]))
    stop      = datetime(int(stop_time.split('-')[0]), int(stop_time.split('-')[1]), int(stop_time.split('-')[2]))
    dates     = pd.date_range(start, stop)
    
    thredds_folder = 'https://thredds.met.no/thredds/dodsC/'
    k         = 0
    
    # non-accumulated values
    time      = np.empty(0)
    path      = []
    path_acc  = []
    path_rad  = []
    index     = []
    missing_flag = []

    # Temperature, wind and pressure
    # ----------------------------------------------------------------------------------
    accdelay  = 2    # Since we can't read the last hour in the day before, and since the first value is masked sometimes. That is only true
                     # in the case of to full_backup files, but I am in too much of a hurry to make it more general this time around.
    missing   = 0
    for date in dates:
        year     = str(date.year)
        month    = '{:02d}'.format(date.month)
        day      = '{:02d}'.format(date.day)

        # Check that files for accumulated values are there
        # --------------------------------------------------
        if date < datetime(2020,2,5):
            try:
                file = thredds_folder + 'meps25epsarchive/' + year + '/' + month + '/' +\
                       day + '/meps_mbr0_pp_2_5km_' + year + month + day + 'T00Z.nc'
                d    = Dataset(file, 'r')
            
            except:
                try:
                    file = thredds_folder + 'meps25epsarchive/' + year + '/' + month + '/' +\
                           day + '/meps_mbr0_extracted_2_5km_' + year + month + day + 'T00Z.nc'
                    d    = Dataset(file, 'r')
                    missing = 0

                except:
                    try:
                        file = thredds_folder + 'meps25epsarchive/' + year + '/' + month + '/' +\
                               day + '/meps_subset_2_5km_' + year + month + day + 'T00Z.nc'
                        d    = Dataset(file, 'r')
                        missing = 0

                    except:
                        missing += 1
                        if missing == 1:
                            print('- ' + str(date) + ' - not available.')
                            
                        elif missing == 2:
                            print('- ' + str(date) + ' - not available.')
                    
                        elif missing == 3:
                            print('- ' + str(date) + ' - not available, and will not be completely covered in the forcing file!')
                        else:
                            print('- ' + str(date) + ' - not available, and not enough forecast to be in the forcing')

                        continue
    
            # check that files for non-accumulated values are there. We prefer to use extracted files, but will use full_backup files if needbe
            # ----------------------------------------------------------------------------------------------------------------------------------
            try:
                file_acc = thredds_folder + 'meps25epsarchive/' + year + '/' + month + '/' +\
                           day + '/meps_mbr0_extracted_2_5km_' + year + month + day + 'T00Z.nc'
                d        = Dataset(file_acc, 'r')
                file_rad = file_acc
            
            except:
                try:
                    file_acc = thredds_folder + 'meps25epsarchive/' + year + '/' + month + '/' +\
                               day + '/meps_mbr0_full_backup_2_5km_' + year + month + day + 'T00Z.nc'
                    d        = Dataset(file_acc, 'r')

                    file_rad = thredds_folder + 'meps25epsarchive/' + year + '/' + month + '/' +\
                               day + '/meps_subset_2_5km_' + year + month + day + 'T00Z.nc'
                    
                    d        = Dataset(file_rad, 'r')
                except:
                    print('- '+str(date)+' - not available')
                    continue

        else:
            try:
                file = thredds_folder + 'meps25epsarchive/' + year + '/' + month + '/' +\
                       day + '/meps_det_2_5km_'+year+month+day+'T00Z.nc'
                file_rad = file
                file_acc = file
                d    = Dataset(file)

            except:
                print('- '+str(date)+' - not available')
                continue

        # We can't use the first two indexes due to accumulated values and NaNs at the beginning
        # of some of the files.
        # ----
        t         = d.variables['time'][accdelay:-2]
        if len(d.variables['time'][:])==0: # Check to see if empty
            print('- '+str(date) + ' - not available')
            continue

        time     = np.append(time, t)
        path     = path + [file]*len(t)
        path_acc = path_acc + [file_acc]*len(t)
        path_rad = path_rad + [file_rad]*len(t)
        index.extend(list(range(accdelay, len(t)+accdelay)))
        print('- '+str(date))

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


def nearest4(FVCOM_grd, AROME_grd):
    '''
    Create nearest four indices and weights.
    '''
    widget   = ['Finding nearest 4 and weights: ', pb.Percentage(), pb.BouncingBar()]
    bar      = pb.ProgressBar(widgets=widget, maxval=len(FVCOM_grd.x)+len(FVCOM_grd.xc))
    N4       = N4AROME(FVCOM_grd, AROME_grd)

    # Find mask for extracing AROME-points limited by FVCOM grid domain
    N4.fv_domain_mask    = AROME_grd.crop_grid(xlim=[FVCOM_grd.x.min() - 800, FVCOM_grd.x.max() + 800],
                                               ylim=[FVCOM_grd.y.min() - 800, FVCOM_grd.y.max() + 800])

    N4.fv_domain_mask_ex = AROME_grd.crop_extended(xlim=[FVCOM_grd.x.min() - 800, FVCOM_grd.x.max() + 800],
                                                   ylim=[FVCOM_grd.y.min() - 800, FVCOM_grd.y.max() + 800])

    LandMask = AROME_grd.landmask[N4.fv_domain_mask]
    x_arome  = AROME_grd.x[N4.fv_domain_mask]
    y_arome  = AROME_grd.y[N4.fv_domain_mask]

    widget   = ['Finding nearest 4 and weights: ', pb.Percentage(), pb.BouncingBar()]
    bar      = pb.ProgressBar(widgets=widget, maxval=len(FVCOM_grd.x)+len(FVCOM_grd.xc))

    # Nodes
    bcnt = 0
    bar.start()
    for i, (x, y) in enumerate(zip(FVCOM_grd.x, FVCOM_grd.y)):
        bar.update(bcnt)
        distance = np.sqrt((x_arome - x)**2 + (y_arome - y)**2)
        
        # For wind, temperature and precipitation fields
        # ----
        indices_sorted_according_to_distance = distance.argsort()[0:4]
        nearest_distances = distance[indices_sorted_according_to_distance]
        nearest_distances = 1.0/nearest_distances
        sum_of_distances  = np.sum(nearest_distances)
        N4.nindex[i,:]    = indices_sorted_according_to_distance
        N4.ncoef[i,:]     = nearest_distances/sum_of_distances
        
        # For radiation fields
        # ------------------------------------------------------------------
        land_value        = LandMask[indices_sorted_according_to_distance] # Local (nearest 4) landmask
        distance[np.where(LandMask<1.0)] = 1000000000.0 # We don't want to use areas influenced by land, hence we "move" the land far away
        closest_ocean     = distance.argsort()[0:4]
            
        # check if land has snuck into our nearest 4, remove it if so
        if land_value.max()==1.0:
            nearest_distances[np.where(land_value!=1.0)] = 0.0

        elif land_value.max()<1.0:
            # Use closest_ocean, and fill the rest of the vector with dummies
            nearest_distances[:] = 0.0
            nearest_distances[0:4] = distance[closest_ocean]
            indices_sorted_according_to_distance[0:4] = closest_ocean
            
        sum_of_distances   = np.sum(nearest_distances)
        N4.nindex_rad[i,:] = indices_sorted_according_to_distance
        N4.ncoef_rad[i,:]  = nearest_distances/sum_of_distances

        bcnt             += 1

    # Elements
    for i, (x, y) in enumerate(zip(FVCOM_grd.xc, FVCOM_grd.yc)):
        bar.update(bcnt)
        distance = np.sqrt((x_arome - x)**2 + (y_arome - y)**2)
        indices_sorted_according_to_distance = distance.argsort()[0:4]
        nearest_distances = distance[indices_sorted_according_to_distance]
        nearest_distances = 1.0/nearest_distances
        sum_of_distances  = nearest_distances.sum()

        N4.cindex[i,:]    = indices_sorted_according_to_distance
        N4.ccoef[i,:]     = nearest_distances/sum_of_distances
        bcnt += 1

    bar.finish()

    N4.nindex     = N4.nindex.astype(int)
    N4.cindex     = N4.cindex.astype(int)
    N4.nindex_rad = N4.nindex_rad.astype(int)
    N4.FVCOM_grd  = FVCOM_grd
    N4.AROME_grd  = AROME_grd
    
    return N4

class N4AROME():
    '''
    Object with indices and coefficients for AROME to FVCOM interpolation
    '''

    def __init__(self, FVCOM_grd, AROME_grd):
        '''Initialize empty attributes'''
        # For all fields except shortwave radiation
        self.ncoef         = np.empty([len(FVCOM_grd.x),  4])
        self.nindex        = np.empty([len(FVCOM_grd.x),  4])
        self.ccoef         = np.empty([len(FVCOM_grd.xc), 4])
        self.cindex        = np.empty([len(FVCOM_grd.xc), 4])
        self.cdistance     = np.empty([len(FVCOM_grd.xc), 4])

        # For shortwave radiation
        self.ncoef_rad     = np.empty([len(FVCOM_grd.x),  4])
        self.nindex_rad    = np.empty([len(FVCOM_grd.x),  4])
        self.ccoef_rad     = np.empty([len(FVCOM_grd.xc), 4])
        self.cindex_rad    = np.empty([len(FVCOM_grd.xc), 4])
        self.cdistance_rad = np.empty([len(FVCOM_grd.xc), 4])

    def save(self, name="Nearest4"):
        '''Save object to file.'''
        pickle.dump(self, open( name + ".p", "wb" ) )

def arome2fvcom(outfile, oldNearest4, newNearest4, arome_time, path, path_acc, path_rad, index, stop): 
    '''
    Read AROME-data, interpolate to FVCOM-grid and export to nc-file.

    Due to the messy nature of MetCoOp data storage, we have to do this in two steps.
    The first loop handles data that do not have NaNs in their first column, while
    the second handles data where we use the forecast 2 hours into the next day to be
    able to give a good estimate of the average based on the accumulated value.
    '''
    print('Interpolating MetCoOp data: ')
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

        # Data at thredds is sometimes unavailable. We wait for it to come back online in such situations.
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
        out.variables['time'][counter]                 = fvcom_time
        out.variables['Itime'][counter]                = fvcom_time # np.floor(fvcom_time)?
        out.variables['Itime2'][counter]               = (fvcom_time - np.floor(fvcom_time)) * 24 * 60 * 60 * 1000
        out.variables['short_wave'][counter, :]        = SwD
        out.variables['long_wave'][counter, :]         = andLw
        out.variables['evap'][counter, :]              = evap
        out.variables['precip'][counter, :]            = precip
        out.variables['cloud_cover'][counter, :]       = caf
        out.variables['relative_humidity'][counter, :] = relative_humidity*100.0 # from kg/kg fraction to percentage
        out.variables['air_pressure'][counter, :]      = Pair
        out.variables['SAT'][counter, :]               = Tair-273.15
        out.variables['air_temperature'][counter, :]   = Tair-273.15
        out.variables['SPQ'][counter, :]               = spechum                 # decimal fraction kg/kg
        out.variables['V10'][counter, :]               = Vwind
        out.variables['vwind_speed'][counter, :]       = Vwind
        out.variables['U10'][counter, :]               = Uwind
        out.variables['uwind_speed'][counter, :]       = Uwind

    out.close()

def read_data(Nearest4, time, dtime, path, path_acc, path_rad, index, already_read, nc = ' ', nc_a = ' ', nc_r = ' '):
    '''
    Routine that reads raw meps data covering the the model domain.
    '''
    if path != already_read:
        print(' ')
        print('Reading data from:')
        print(path)
        nc   = Dataset(path,'r')
        
        # No need to download the same thing twice
        # ----
        if path.find('_det_')==-1:
            if path != path_acc:
                print(path_acc)
                nc_a = Dataset(path_acc,'r')
            else:
                nc_a = nc

            if path != path_rad:
                print(path_rad)
                nc_r = Dataset(path_rad,'r')
            else:
                nc_r = nc

        else:
            nc_a = ' '
            nc_r = ' '

    print('- timestep: '+str(dtime))

    already_read          = path
    fvcom_time            = netCDF4.date2num(dtime, units = 'days since 1858-11-17 00:00:00')

    # Indices to be read by accumulated value files
    read_inds             = [index-1,index+1]

    if path.find('_pp_')>=0:
        Uwind             = nc.variables.get('x_wind_10m')[index, :, :][Nearest4.fv_domain_mask]
        Vwind             = nc.variables.get('y_wind_10m')[index, :, :][Nearest4.fv_domain_mask]
        Tair              = nc.variables.get('air_temperature_2m')[index, 0, :, :][Nearest4.fv_domain_mask]
        Pair              = nc.variables.get('air_pressure_at_sea_level')[index, 0, :, :][Nearest4.fv_domain_mask]
        relative_humidity = nc.variables.get('relative_humidity_2m')[index, 0, :, :][Nearest4.fv_domain_mask]
        caf               = nc.variables.get('cloud_area_fraction')[index, :, :][Nearest4.fv_domain_mask]*0.01 # due to stupid inconsistency
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
    if path_acc.find('_full_')>=0:
        # Load data centered above this timestep
        evap    = nc_a.variables.get('water_evaporation_amount')[read_inds, 0, 0, :, :][:, Nearest4.fv_domain_mask_ex]
        spechum = nc_a.variables.get('specific_humidity_2m')[index, 0, 0, :, :][Nearest4.fv_domain_mask_ex]
        SwD     = nc_r.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, 0, :, :][:,Nearest4.fv_domain_mask_ex]
        andLw   = nc_r.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, 0, :, :][:,Nearest4.fv_domain_mask_ex]

    elif path_acc.find('_det_')>=0:
        evap    = nc.variables.get('water_evaporation_amount')[read_inds, 0, :, :][:, Nearest4.fv_domain_mask]
        spechum = nc.variables.get('specific_humidity_2m')[index, 0, :, :][Nearest4.fv_domain_mask]
        SwD     = nc.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask]
        andLw   = nc.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask]

    else:
        # Load data centered above this timestep
        # full_backup
        evap    = nc_a.variables.get('water_evaporation_amount')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask_ex]
        spechum = nc_a.variables.get('specific_humidity_2m')[index, 0, :, :][Nearest4.fv_domain_mask_ex]

        if path_rad.find('_extracted_')>=0:
            SwD     = nc_r.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask_ex]
            andLw   = nc_r.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, :, :][:,Nearest4.fv_domain_mask_ex]
        else:
            SwD     = nc_r.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, 0, :, :][:,Nearest4.fv_domain_mask_ex]
            andLw   = nc_r.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, 0, :, :][:,Nearest4.fv_domain_mask_ex]

    # interpolate to this timestep
    SwD     = (SwD[1,:]    - SwD[0,:])/(2.0*3600.0)
    andLw   = (andLw[1,:]  - andLw[0,:])/(2.0*3600.0)

    evap    = (evap[1,:] - evap[0,:])/(1000.0*2.0*3600.0)     # from 2h*(kg m-2) to m s-1
    precip  = (precip[1,:] - precip[0,:])/(1000.0*2.0*3600.0) # from 2h*(kg m-2) to m s-1

    return fvcom_time, SwD, andLw, evap, precip, caf, relative_humidity, spechum, Pair, Tair, Vwind, Uwind, nc, nc_a, nc_r, already_read
