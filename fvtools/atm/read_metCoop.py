import netCDF4
import numpy as np
import pandas as pd
import time as time_mod
import multiprocessing as mp
import progressbar as pb
import warnings
warnings.filterwarnings("ignore")

from fvtools.grid.fvcom_grd import FVCOM_grid  # objects that load what we need to know about the FVCOM grid
from fvtools.grid.arome_grid import get_arome_grids, NoAvailableData # objects that load arome grid data
from fvtools.interpolators.arome_interpolators import N4AROME

from time import gmtime, strftime
from datetime import datetime, timedelta
from functools import cached_property
from dataclasses import dataclass

'''
This script downloads AROME data, interpolates it to the FVCOM mesh (using a bilinear interpolation algorithm) and dumps the
interpolated data to a netCDF forcing file.
'''

def main(grd_file, outfile, start_time, stop_time, nearest4=None, latlon=False):
    '''
    Create AROME atmospheric forcing file for FVCOM 

    Parameters:
    ----
    grd_file:   'M.npy' or equivalent
    outfile:     name of netcdf output
    start_time: 'yyyy-mm-dd-hh'
    stop_time:  'yyyy-mm-dd-hh'
    nearest4:   'nearest4arome.npy' file if you have already made an _atm file for this mesh (=None by default)
    latlon:     Set true if you are making a latlon model (will otherwise rotate to adjust for UTM north/east distortion)
    '''
    print(f'\nCreate {outfile} - unstuctured grid atmospheric forcing file')
    print(f'---')
    print(f'- Load {grd_file}')
    M = FVCOM_grid(grd_file)
    startdate, stopdate = get_start_and_stop_as_datetime(start_time, stop_time)

    print('- Load AROME grid')
    OldAROME, NewAROME = get_arome_grids(startdate, stopdate, M.Proj)

    print('- Make a filelist')
    time, path, path_acc, path_rad, index = metcoop_make_fileList(startdate, stopdate, OldAROME = OldAROME, NewAROME = NewAROME)

    if nearest4 is None:
        print('\nCompute nearest4 interpolation coefficients.')
    else:
        print('\nLoad nearest4 interpolation coefficients')

    # Two different grids, with the option to load pre-loaded interpolation coefficients
    oldN4, newN4 = get_nearest4(OldAROME, NewAROME, nearest4, M)

    # Dump to outfile
    print('\nDump to outfile\n---')
    create_nc_forcing_file(outfile, M, time, latlon, M.reference)
    Downloader = AROMEDownloader(newN4, oldN4, time, path, path_rad, path_acc, index, outfile, latlon)
    Downloader.dump_single()

# Routines to parse input
def get_start_and_stop_as_datetime(start_time, stop_time):
    '''
    return start and stop as datetime objects
    '''
    startnum   = start_time.split('-')
    stopnum    = stop_time.split('-')
    startdate  = datetime(int(startnum[0]), int(startnum[1]), int(startnum[2]))
    stopdate   = datetime(int(stopnum[0]), int(stopnum[1]), int(stopnum[2]))
    return startdate, stopdate

def get_nearest4(OldAROME, NewAROME, nearest4, M):
    '''
    The data on met/thredds is stored on two AROME/MetCoOp main grids - the "original" running to 5. feb. 2020, and the extended
    version running (over a bigger domain since they now work together with the baltic states) running ever since.

    We seem to need two different interpolation algorithms, since they changed the grid (not sure if the norwegian
    grid positions are the same as before)
    '''
    N4A = None; newN4A = None
    if OldAROME is None and NewAROME is None:
        nearest4 = None
    if OldAROME is not None:
        N4A = N4AROME(M, OldAROME)

        if nearest4 is not None:
            N4A.load_nearest4(nearest4)

        else:
            N4A.compute_nearest4()

    if NewAROME is not None:
        newN4A = N4AROME(M, NewAROME)
        if nearest4 is not None:
            newN4A.load_nearest4(nearest4)
        else:
            newN4A.compute_nearest4()

    return N4A, newN4A

# =====================================================================================================================================
def metcoop_make_fileList(start_time, stop_time, OldAROME = None, NewAROME = None):
    '''
    Go through MetCoop thredds server and link points in time to files.
    format: yyyy-mm-dd-hh

    The filelist is used to decide what files to use when forecasts are available, and implicitly
    to "remember" what files are available in case of server downtime during the later interpolation step.
    '''
    dates     = pd.date_range(start_time, stop_time)
    
    # non-accumulated values
    # ----
    time     = np.empty(0)
    path     = []
    path_acc = []
    path_rad = []
    index    = []

    # Since we can't read the last hour in the day before, and since the first value is masked in full_backup files.
    # ----
    accdelay = 2
    missing  = 0 # To track missing days

    # Loop over all relevant dates
    # ----
    for date in dates:
        # Check if files for a given date are available on thredds
        # --------------------------------------------------------
        try:
            if date < datetime(2020,2,5):
                file, file_acc, file_rad = OldAROME.test_day(date)
            else:
                file, file_acc, file_rad = NewAROME.test_day(date)

        except NoAvailableData:
            missing+=1
            if missing < 2:
                print(f'  - {date} - reads forecast from {date-timedelta(days=missing)}')
            else:
                print(f'  - {date} - is unavailable')
            continue

        # Just to check that the file is not empty
        # --------------------------------------------------------
        with netCDF4.Dataset(file, 'r') as d:
            if len(d.variables['time'][:]) == 0: # Check to see if empty
                print(f'  - file for {date} is empty')
                continue
            missing = 0

            # We can't use the first two indexes due to accumulated values and NaNs at the beginning of some of the files.
            # ----
            t_arome  = netCDF4.num2date(d.variables['time'][accdelay:-2], units = d['time'].units, only_use_cftime_datetimes=False)
            t_fvcom  = netCDF4.date2num(t_arome, units = 'days since 1858-11-17 00:00:00')
            time     = np.append(time, t_fvcom)
            path     = path + [file]*len(t_fvcom)
            path_acc = path_acc + [file_acc]*len(t_fvcom)
            path_rad = path_rad + [file_rad]*len(t_fvcom)
            index.extend(list(range(accdelay, len(t_fvcom)+accdelay)))
        print(f'  - found data for {date}')

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

class AROMEDownloader:
    def __init__(self, newN4, oldN4, time, path, path_rad, path_acc, index, outfile, latlon):
        '''
        Class downloading AROME data
        '''
        self.outfile = outfile
        self.normal_paths = path
        self.radiation_paths = path_rad
        self.accumulation_paths = path_acc
        self.indices = index
        self.times = time
        try:
            self.N4old = oldN4.dump()
        except:
            self.N4old = None

        try:
            self.N4new = newN4.dump()
        except:
            self.N4new = None
        self.latlon = latlon

    def dump_single(self):
        '''
        Load and dump using a single processor
        '''
        widget = [f'  Downloading arome timesteps: ', pb.Percentage(), pb.BouncingBar(), pb.ETA()]
        bar = pb.ProgressBar(widgets=widget, maxval=len(self.indices))
        bar.start()
        with netCDF4.Dataset(self.outfile, 'r+') as self.out:
            for counter, (path, path_acc, path_rad, index) in enumerate(zip(self.normal_paths, self.accumulation_paths, self.radiation_paths, self.indices)):
                bar.update(counter)
                if path.find('_det_') > 0:
                    N4 = self.N4new
                else:
                    N4 = self.N4old
                timestep = self.read_data(N4, path, path_acc, path_rad, index)
                timestep.counter = counter 
                timestep = self.interpolate_arome_data(timestep, N4)
                if not self.latlon:
                    timestep = self.rotate_arome_vectors(timestep, N4)
                self.write_timestep(timestep)
        bar.finish()

    def dump(self, nprocs = None): 
        '''
        Read AROME-data, interpolate to FVCOM-grid and export to nc-file.

        The first loop handles data that do not have NaNs in their first column, while
        the second handles data where we use the forecast 2 hours into the next day to be
        able to give a good estimate of the average based on the accumulated value.

        --> Add the worker - listener workflow
        '''
        # Prepare the multiprocessing team
        # ----
        manager = mp.Manager() # We need a manager since all processes (minus one) will communicate with the writer
        q = manager.Queue()    # q is convey messages from the workers to the listener
        if nprocs is None:
            pool = mp.Pool(mp.cpu_count()+2)
        else:
            pool = mp.Pool(nprocs+2) # +2 to make sure that we always have room for workers after listener and the watcher are added to the list

        # Initialize the listener
        watcher = pool.apply_async(self._listener, (q,))

        # Send data to workers
        jobs = []
        for counter, (path, path_acc, path_rad, index) in enumerate(zip(self.normal_paths, self.accumulation_paths, self.radiation_paths, self.indices)):   
            if path.find('_det_') > 0:
                N4 = self.N4new
            else:
                N4 = self.N4old
            job = pool.apply_async(self._worker, (N4, counter, path, path_acc, path_rad, index, q))
            jobs.append(job)

        # Collect results from the workers through the pool result queue
        for job in jobs: 
            job.get()

        #now we are done, kill the listener
        q.put('kill')
        pool.close()
        pool.join()

    def _worker(self, N4, counter, path, path_acc, path_rad, index, q):
        '''
        The worker downloads data
        - does not interpolate, since that will increase the memory use of the worker quite alot
        '''
        thredds_unavailable = True
        while thredds_unavailable:
            try:
                timestep = self.read_data(N4, path, path_acc, path_rad, index)
                timestep.counter = counter
                timestep.N4 = N4
                thredds_unavailable = False

            except RuntimeError:
                print('\n---------------------------------------------------------------------------------\n '+\
                        '   The data is unavailable at the moment. We wait thirty seconds and try again.'+\
                      '\n---------------------------------------------------------------------------------\n')
                time_mod.sleep(30)
                
        q.put(timestep)
        return ''

    def _listener(self, q):
        '''
        Write AROME data to the object
        '''
        widget = [f'  Downloading arome timesteps: ', pb.Percentage(), pb.BouncingBar(), pb.ETA()]
        bar = pb.ProgressBar(widgets=widget, maxval=len(self.indices))
        i = 0
        bar.start()
        with netCDF4.Dataset(self.outfile, 'r+') as self.out:
            while True:
                timestep = q.get()
                if timestep == 'kill':
                    bar.finish()
                    break

                if type(timestep) == AROMETimestep:
                    i+=1
                    bar.update(i)
                    timestep = self.interpolate_arome_data(timestep, timestep.N4)
                    if not self.latlon:
                        timestep = self.rotate_arome_vectors(timestep, timestep.N4)
                    self.write_timestep(timestep)

    def write_timestep(self, timestep):
        '''
        dump timestep to out
        '''
        self.out.variables['short_wave'][timestep.counter, :]        = timestep.SwD
        self.out.variables['long_wave'][timestep.counter, :]         = timestep.andLw
        self.out.variables['evap'][timestep.counter, :]              = timestep.evap
        self.out.variables['precip'][timestep.counter, :]            = timestep.precip
        self.out.variables['cloud_cover'][timestep.counter, :]       = timestep.caf
        self.out.variables['relative_humidity'][timestep.counter, :] = timestep.relative_humidity*100.0 # from kg/kg fraction to percentage
        self.out.variables['air_pressure'][timestep.counter, :]      = timestep.Pair
        self.out.variables['SAT'][timestep.counter, :]               = timestep.Tair-273.15 # Kelvin to C
        self.out.variables['air_temperature'][timestep.counter, :]   = timestep.Tair-273.15 # Kelvin to C
        self.out.variables['SPQ'][timestep.counter, :]               = timestep.spechum     # decimal fraction kg/kg
        self.out.variables['V10'][timestep.counter, :]               = timestep.Vwind
        self.out.variables['vwind_speed'][timestep.counter, :]       = timestep.Vwind
        self.out.variables['U10'][timestep.counter, :]               = timestep.Uwind
        self.out.variables['uwind_speed'][timestep.counter, :]       = timestep.Uwind

    def interpolate_arome_data(self, timestep, N4):
        '''
        Interpolate AROME data to the FVCOM grid
        '''
        # Fields where we use AROME land points
        timestep.caf               = np.sum(timestep.caf[N4.nindex] * N4.ncoef, axis=1)
        timestep.Pair              = np.sum(timestep.Pair[N4.nindex] * N4.ncoef, axis=1)
        timestep.relative_humidity = np.sum(timestep.relative_humidity[N4.nindex] * N4.ncoef, axis=1)
        timestep.spechum           = np.sum(timestep.spechum[N4.nindex] * N4.ncoef, axis=1)
        timestep.Tair              = np.sum(timestep.Tair[N4.nindex] * N4.ncoef, axis=1)
        timestep.Vwind             = np.sum(timestep.Vwind[N4.cindex] * N4.ccoef, axis=1)
        timestep.Uwind             = np.sum(timestep.Uwind[N4.cindex] * N4.ccoef, axis=1)
        timestep.evap              = np.sum(timestep.evap[N4.nindex] * N4.ncoef, axis=1)
        timestep.precip            = np.sum(timestep.precip[N4.nindex] * N4.ncoef, axis=1)

        # Radiation is gathered from nearest ocean points, thus they have their own interpolation method near to land
        timestep.andLw             = np.sum(timestep.andLw[N4.nindex_rad] * N4.ncoef_rad, axis=1)
        timestep.SwD               = np.sum(timestep.SwD[N4.nindex_rad] * N4.ncoef_rad, axis=1)

        return timestep

    def rotate_arome_vectors(self, timestep, N4):
        '''
        When running FVCOM in UTM mode, we need to rotate so that we account for true north
        '''
        timestep.Uwind, timestep.Vwind = self._rotate_vector(timestep.Uwind, timestep.Vwind, N4.cell_utm_angle)
        return timestep

    def _rotate_vector(self, u, v, angle):
        '''
        Rotates vectors from (x', y') system to (x,y)

        (u, v) velocity components in (x', y') system
        angle is the angle between (x') and (xnew)
        returns unew, vnew
        '''
        unew =  u*np.cos(angle) + v*np.sin(angle)
        vnew = -u*np.sin(angle) + v*np.cos(angle)
        return unew, vnew

    @staticmethod
    def read_data(N4, path, path_acc, path_rad, index):
        '''
        Routine that reads raw meps data covering the the model domain.
        '''
        # Initialize output datastructure
        timestep = AROMETimestep()

        # Indices to be read by accumulated value files
        read_inds = [index-1, index]

        # It seems a bit unecessary to load the entire grid every time, we can look into adapting the same type of cropping as is done with ROMS data later
        with netCDF4.Dataset(path,'r') as nc:
            # Standard fields (all fields for new AROME)
            if path.find('_pp_') >= 0:
                timestep.Uwind             = nc.variables.get('x_wind_10m')[index, :, :][N4.fv_domain_mask]
                timestep.Vwind             = nc.variables.get('y_wind_10m')[index, :, :][N4.fv_domain_mask]
                timestep.Tair              = nc.variables.get('air_temperature_2m')[index, 0, :, :][N4.fv_domain_mask]
                timestep.Pair              = nc.variables.get('air_pressure_at_sea_level')[index, 0, :, :][N4.fv_domain_mask]
                timestep.relative_humidity = nc.variables.get('relative_humidity_2m')[index, 0, :, :][N4.fv_domain_mask]
                timestep.caf               = nc.variables.get('cloud_area_fraction')[index, :, :][N4.fv_domain_mask]*0.01 # convert from percent to fraction
                timestep.precip            = nc.variables.get('precipitation_amount_acc')[read_inds, 0, :, :][:, N4.fv_domain_mask]

            elif path.find('extracted') >= 0:
                timestep.Uwind             = nc.variables.get('x_wind_10m')[index, 0, :, :][N4.fv_domain_mask_ex]
                timestep.Vwind             = nc.variables.get('y_wind_10m')[index, 0, :, :][N4.fv_domain_mask_ex]
                timestep.Tair              = nc.variables.get('air_temperature_2m')[index, 0, :, :][N4.fv_domain_mask_ex]
                timestep.Pair              = nc.variables.get('air_pressure_at_sea_level')[index, 0, :, :][N4.fv_domain_mask_ex]
                timestep.relative_humidity = nc.variables.get('relative_humidity_2m')[index, 0, :, :][N4.fv_domain_mask_ex]
                timestep.caf               = nc.variables.get('cloud_area_fraction')[index, 0, :, :][N4.fv_domain_mask_ex]
                timestep.precip            = nc.variables.get('precipitation_amount_acc')[read_inds, 0, :, :][:, N4.fv_domain_mask_ex]

            elif path.find('subset') >= 0:
                timestep.Uwind             = nc.variables.get('x_wind_10m')[index, 0, 0, :, :][N4.fv_domain_mask_ex]
                timestep.Vwind             = nc.variables.get('y_wind_10m')[index, 0, 0, :, :][N4.fv_domain_mask_ex]
                timestep.Tair              = nc.variables.get('air_temperature_2m')[index, 0, 0, :, :][N4.fv_domain_mask_ex]
                timestep.Pair              = nc.variables.get('air_pressure_at_sea_level')[index, 0, 0, :, :][N4.fv_domain_mask_ex]
                timestep.relative_humidity = nc.variables.get('relative_humidity_2m')[index, 0, 0, :, :][N4.fv_domain_mask_ex]
                timestep.caf               = nc.variables.get('cloud_area_fraction')[index, 0, 0, :, :][N4.fv_domain_mask_ex]
                timestep.precip            = nc.variables.get('precipitation_amount_acc')[read_inds, 0, 0, :, :][:, N4.fv_domain_mask_ex]

            elif path.find('_det_') >= 0:
                timestep.Uwind             = nc.variables.get('x_wind_10m')[index, 0, :, :][N4.fv_domain_mask]
                timestep.Vwind             = nc.variables.get('y_wind_10m')[index, 0, :, :][N4.fv_domain_mask]
                timestep.Tair              = nc.variables.get('air_temperature_2m')[index, 0, :, :][N4.fv_domain_mask]
                timestep.Pair              = nc.variables.get('air_pressure_at_sea_level')[index, 0, :, :][N4.fv_domain_mask]
                timestep.relative_humidity = nc.variables.get('relative_humidity_2m')[index, 0, :, :][N4.fv_domain_mask]
                timestep.caf               = nc.variables.get('cloud_area_fraction')[index, 0, :, :][N4.fv_domain_mask]
                timestep.precip            = nc.variables.get('precipitation_amount_acc')[read_inds, 0, :, :][:, N4.fv_domain_mask]
                timestep.evap              = nc.variables.get('water_evaporation_amount')[read_inds, 0, :, :][:, N4.fv_domain_mask]
                timestep.spechum           = nc.variables.get('specific_humidity_2m')[index, 0, :, :][N4.fv_domain_mask]
                timestep.SwD               = nc.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, :, :][:, N4.fv_domain_mask]
                timestep.andLw             = nc.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, :, :][:, N4.fv_domain_mask]

        # Accumulated fields
        if path_acc.find('_full_') >= 0:
            with netCDF4.Dataset(path_acc,'r') as nc_a:
                timestep.evap    = nc_a.variables.get('water_evaporation_amount')[read_inds, 0, 0, :, :][:, N4.fv_domain_mask_ex]
                timestep.spechum = nc_a.variables.get('specific_humidity_2m')[index, 0, 0, :, :][N4.fv_domain_mask_ex]

            with netCDF4.Dataset(path_rad,'r') as nc_r:
                timestep.SwD     = nc_r.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, 0, :, :][:, N4.fv_domain_mask_ex]
                timestep.andLw   = nc_r.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, 0, :, :][:, N4.fv_domain_mask_ex]

        elif path.find('_det_') >= 0:
            pass # no need for special treatment of radiation fields

        else:
            with netCDF4.Dataset(path_acc,'r') as nc_a:
                timestep.evap    = nc_a.variables.get('water_evaporation_amount')[read_inds, 0, :, :][:, N4.fv_domain_mask_ex]
                timestep.spechum = nc_a.variables.get('specific_humidity_2m')[index, 0, :, :][N4.fv_domain_mask_ex]

            with netCDF4.Dataset(path_rad,'r') as nc_r:
                if path_rad.find('_extracted_') >= 0:
                    timestep.SwD     = nc_r.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, :, :][:, N4.fv_domain_mask_ex]
                    timestep.andLw   = nc_r.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, :, :][:, N4.fv_domain_mask_ex]
                else:
                    timestep.SwD     = nc_r.variables.get('integral_of_surface_net_downward_shortwave_flux_wrt_time')[read_inds, 0, 0, :, :][:, N4.fv_domain_mask_ex]
                    timestep.andLw   = nc_r.variables.get('integral_of_surface_downwelling_longwave_flux_in_air_wrt_time')[read_inds, 0, 0, :, :][:, N4.fv_domain_mask_ex]

        # de-accumulate :)
        # ----
        timestep.SwD     = (timestep.SwD[1,:] - timestep.SwD[0,:])/(3600.0)
        timestep.andLw   = (timestep.andLw[1,:] - timestep.andLw[0,:])/(3600.0)

        timestep.evap    = (timestep.evap[1,:]   - timestep.evap[0,:])/(1000.0*3600.0)     # converting from (kg m-2) accumulated over 1 h to m s-1, assuming water density = 1000 kg / m^3
        timestep.precip  = (timestep.precip[1,:] - timestep.precip[0,:])/(1000.0*3600.0)   # converting from (kg m-2) accumulated over 1 h to m s-1, assuming water density = 1000 kg / m^3

        return timestep

def create_nc_forcing_file(name, FVCOM_grd, times, latlon, epsg):
    '''
    Creates empty nc file formatted to fit fvcom atm forcing
    '''
    with netCDF4.Dataset(name, 'w', format='NETCDF4') as nc:
        # Write global attributes
        nc.title       = 'FVCOM Forcing File'
        nc.institution = 'Akvaplan-niva AS'
        nc.source      = 'FVCOM grid (unstructured) surface forcing'
        nc.created     = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        if latlon:
            nc.interpolation_projection = 'degrees'
        else:
            nc.interpolation_projection = epsg

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
        lat.units          = 'degrees_north'

        lon                = nc.createVariable('lon', 'single', ('node',))
        lon.long_name      = 'nodal longitude'
        lon.units          = 'degrees_east'

        latc               = nc.createVariable('latc', 'single', ('nele',))
        latc.long_name     = 'zonal latitude'
        latc.units         = 'degrees_north'

        lonc               = nc.createVariable('lonc', 'single', ('nele',))
        lonc.long_name     = 'zonal longitude'
        lonc.units         = 'degrees_east'

        x                  = nc.createVariable('x', 'single', ('node',))
        x.long_name        = 'nodal x-coordinate'
        x.units            = 'meters'
        
        y                  = nc.createVariable('y', 'single', ('node',))
        y.long_name        = 'nodal y-coordinate'
        y.units            = 'meters'

        xc                 = nc.createVariable('xc', 'single', ('nele',))
        xc.long_name       = 'zonal x-coordinate'
        xc.units           = 'meters'
        
        yc                  = nc.createVariable('yc', 'single', ('nele',))
        yc.long_name       = 'zonal x-coordinate'
        yc.units           = 'meters'

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


        nc.variables['x'][:]    = FVCOM_grd.x
        nc.variables['y'][:]    = FVCOM_grd.y
        nc.variables['yc'][:]   = FVCOM_grd.yc
        nc.variables['xc'][:]   = FVCOM_grd.xc

        nc.variables['lat'][:]  = FVCOM_grd.lat
        nc.variables['lon'][:]  = FVCOM_grd.lon
        nc.variables['latc'][:] = FVCOM_grd.latc
        nc.variables['lonc'][:] = FVCOM_grd.lonc

        tris                    = FVCOM_grd.tri+1
        nc.variables['nv'][:]   = tris.transpose()

        for counter, fvcom_time in enumerate(times):
            nc.variables['time'][counter]   = fvcom_time
            nc.variables['Itime'][counter]  = np.floor(fvcom_time)
            nc.variables['Itime2'][counter] = (fvcom_time - np.floor(fvcom_time)) * 24 * 60 * 60 * 1000

@dataclass
class AROMETimestep:
    '''
    container for the variables we need to download every timestep
    - fill with values it should expect in the future...
    '''
    pass