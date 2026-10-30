import netCDF4
import numpy as np
import pandas as pd
import time as time_mod
import multiprocessing as mp
import progressbar as pb

from fvtools.grid.fvcom_grd import FVCOM_grid  # objects that load what we need to know about the FVCOM grid
from fvtools.grid.arome_grid import get_arome_grids, NoAvailableData # objects that load arome grid data
#from fvtools.interpolators.arome_interpolators import N4AROME
from fvtools.interpolators.horizontal_interpolator import N4Coefficients

from time import gmtime, strftime
from datetime import datetime, timedelta
from dataclasses import dataclass
from ._metcoop_io import SourcePool, read_masked, is_retryable_error
from fvtools.util.thredds_io import retry_read

'''
This script downloads AROME data, interpolates it to the FVCOM mesh (using a bilinear interpolation algorithm) and dumps the
interpolated data to a netCDF forcing file.
'''

def main(grd_file, outfile, start_time, stop_time, latlon=False):
    '''
    Create AROME atmospheric forcing file for FVCOM 

    Parameters:
    ----
    grd_file:   'M.npy' or equivalent
    outfile:     name of netcdf output
    start_time: 'yyyy-mm-dd-hh'
    stop_time:  'yyyy-mm-dd-hh'
    latlon:     Set true if you are making a latlon model (will otherwise rotate to adjust for UTM north/east distortion)
    '''
    print(f'\nCreate {outfile} - unstuctured grid atmospheric forcing file')
    print(f'---')
    print(f'- Load {grd_file}')
    M = FVCOM_grid(grd_file)
    startdate, stopdate = get_start_and_stop_as_datetime(start_time, stop_time)

    if startdate < datetime(2017,9,7):
        raise ValueError('We have not added support for AROME prior to 7. september 2017, when the MEPS domain was extended to include Finland and Estonia')

    print('- Load AROME grid')
    OldAROME, NewAROME = get_arome_grids(startdate, stopdate, M.Proj)

    print('- Make a filelist')
    time, path, path_acc, path_rad, index = metcoop_make_fileList(startdate, stopdate, OldAROME = OldAROME, NewAROME = NewAROME)

    # Two different grids, with the option to load pre-loaded interpolation coefficients
    oldN4, newN4 = get_nearest4(OldAROME, NewAROME, M)

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

def get_nearest4(OldAROME, NewAROME, M):
    '''
    The data on met/thredds is stored on two AROME/MetCoOp main grids - the "original" running to 5. feb. 2020, and the extended
    version running (over a bigger domain since they now work together with the baltic states) running ever since.

    We seem to need two different interpolation algorithms, since they changed the grid 
    (not sure if the grid points covering Norway are the same as before)
    '''
    N4A = None
    newN4A = None

    if OldAROME is not None:
        N4A = N4Coefficients(M, OldAROME)
        N4A.compute_nearest4()

    if NewAROME is not None:
        newN4A = N4Coefficients(M, NewAROME)
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

    # Used to keep track of what's available
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
            continue

        if missing == 1:
            print(f'  - {date} - reads forecast from {date-timedelta(days=missing)}')
        elif missing > 2:
            print(f'  - {date} - is unavailable')
            continue

        # Just to check that the file is not empty
        # --------------------------------------------------------
        with netCDF4.Dataset(file, 'r') as d:
            if len(d.variables['time']) == 0: # Check to see if empty
                print(f'  - file for {date} is empty')
                continue
            missing = 0

            # We can't use the first two indexes due to accumulated values and NaNs at the beginning of some of the files.
            # ----
            t_arome  = netCDF4.num2date(d.variables['time'][accdelay:-2], units = d['time'].units, only_use_cftime_datetimes = False)
            t_fvcom  = netCDF4.date2num(t_arome, units = 'days since 1858-11-17 00:00:00')
            time     = np.append(time, t_fvcom)
            path     = path + [file]*len(t_fvcom)
            path_acc = path_acc + [file_acc]*len(t_fvcom)
            path_rad = path_rad + [file_rad]*len(t_fvcom)
            index.extend(list(range(accdelay, len(t_fvcom)+accdelay)))
        print(f'  - found data for {date}')

    if len(time) == 0:
        raise NoAvailableData('No usable MetCoOp timesteps in requested period')

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
        self.N4old = oldN4.dump() if oldN4 is not None else None
        self.N4new = newN4.dump() if newN4 is not None else None
        self.latlon = latlon

    def dump_single(self, *, max_retries=5, retry_delay=30):
        """Serial source reads, with at most three open OPeNDAP datasets.

        Only transient source I/O is retried. Interpolation and output errors
        propagate immediately, so programming errors cannot loop forever.
        """
        widget = ['  Downloading arome timesteps: ', pb.Percentage(), pb.BouncingBar(), pb.ETA()]
        bar = pb.ProgressBar(widgets=widget, maxval=len(self.indices))
        bar.start()
        try:
            with SourcePool(netCDF4.Dataset) as sources:
                previous_paths = None
                with netCDF4.Dataset(self.outfile, 'r+') as self.out:
                    for counter, (path, path_acc, path_rad, index) in enumerate(
                        zip(self.normal_paths, self.accumulation_paths,
                            self.radiation_paths, self.indices)
                    ):
                        paths = (path, path_acc, path_rad)
                        if paths != previous_paths:
                            sources.close()
                            previous_paths = paths
                        N4 = self.N4new if '_det_' in path else self.N4old
                        timestep = self._read_with_retry(
                            N4, path, path_acc, path_rad, index,
                            datasets=sources, max_retries=max_retries,
                            retry_delay=retry_delay,
                        )
                        timestep.counter = counter
                        timestep = self.interpolate_arome_data(timestep, N4)
                        if not self.latlon:
                            timestep = self.rotate_arome_vectors(timestep, N4)
                        self.write_timestep(timestep)
                        bar.update(counter + 1)
        finally:
            bar.finish()

    @staticmethod
    def _read_with_retry(N4, path, path_acc, path_rad, index, *,
                         datasets=None, max_retries=5, retry_delay=30):
        """Shared bounded NetCDF/DAP retry, scoped to source reads only."""
        return retry_read(
            lambda: AROMEDownloader.read_data(
                N4, path, path_acc, path_rad, index, datasets=datasets,
            ),
            max_retries=max_retries, retry_delay=retry_delay,
            classifier=is_retryable_error,
            sleep=time_mod.sleep,
            on_transient=(getattr(datasets, 'invalidate', datasets.close)
                          if datasets is not None else None),
            failure_message=f'Could not read MetCoOp timestep {index} from {path}',
            log_prefix='  - MetCoOp I/O error',
        )

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
            pool = mp.Pool(mp.cpu_count() + 2)
        else:
            pool = mp.Pool(nprocs + 2) # +2 to make sure that we always have room for workers after listener and the watcher are added to the list

        # Initialize the listener
        _ = pool.apply_async(self._listener, (q,))

        # Send data to workers
        jobs = []
        for counter, (path, path_acc, path_rad, index) in enumerate(zip(self.normal_paths, self.accumulation_paths, self.radiation_paths, self.indices)):   
            if '_det_' in path:
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
        """Legacy parallel worker: use bounded retries without shared handles."""
        timestep = self._read_with_retry(N4, path, path_acc, path_rad, index)
        timestep.counter = counter
        timestep.N4 = N4
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
        timestep.andLw             = np.sum(timestep.andLw[N4.nindex_ocean] * N4.ncoef_ocean, axis=1)
        timestep.SwD               = np.sum(timestep.SwD[N4.nindex_ocean] * N4.ncoef_ocean, axis=1)

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
    def read_data(N4, path, path_acc, path_rad, index, *, datasets=None):
        """Read only the AROME source rectangle containing FVCOM interpolation points.

        Keep the historic 2-D masks and each file format's dimensional layout.
        ``datasets`` is a SourcePool for serial runs; other callers keep the
        original one-call/one-close lifetime using a short local pool.
        """
        if datasets is None:
            with SourcePool(netCDF4.Dataset) as sources:
                return AROMEDownloader.read_data(
                    N4, path, path_acc, path_rad, index, datasets=sources,
                )

        timestep = AROMETimestep()
        nc = datasets.get(path)
        if '_pp_' in path:
            mask = N4.fv_domain_mask
            wind = (index,)
            standard = (index, 0)
            cloud = (index,)
            accumulated = (slice(index - 1, index + 1), 0)
        elif 'subset' in path:
            mask = N4.fv_domain_mask_ex
            wind = standard = cloud = (index, 0, 0)
            accumulated = (slice(index - 1, index + 1), 0, 0)
        elif 'extracted' in path or '_det_' in path:
            mask = N4.fv_domain_mask_ex if 'extracted' in path else N4.fv_domain_mask
            wind = standard = cloud = (index, 0)
            accumulated = (slice(index - 1, index + 1), 0)
        else:
            raise ValueError(f'Unrecognized MetCoOp source format: {path}')

        def read(source, name, prefix, spatial_mask):
            return read_masked(source, name, prefix, spatial_mask)

        timestep.Uwind = read(nc, 'x_wind_10m', wind, mask)
        timestep.Vwind = read(nc, 'y_wind_10m', wind, mask)
        timestep.Tair = read(nc, 'air_temperature_2m', standard, mask)
        timestep.Pair = read(nc, 'air_pressure_at_sea_level', standard, mask)
        timestep.relative_humidity = read(nc, 'relative_humidity_2m', standard, mask)
        timestep.caf = read(nc, 'cloud_area_fraction', cloud, mask)
        if '_pp_' in path:
            timestep.caf = timestep.caf * 0.01
        timestep.precip = read(nc, 'precipitation_amount_acc', accumulated, mask)

        if '_det_' in path:
            # New AROME: all fields are in the same file and use the same grid.
            timestep.evap = read(nc, 'water_evaporation_amount', accumulated, mask)
            timestep.spechum = read(nc, 'specific_humidity_2m', standard, mask)
            timestep.SwD = read(nc, 'integral_of_surface_net_downward_shortwave_flux_wrt_time', accumulated, mask)
            timestep.andLw = read(nc, 'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time', accumulated, mask)
        else:
            extended_mask = N4.fv_domain_mask_ex
            acc = datasets.get(path_acc)
            acc_prefix = (index, 0, 0) if '_full_' in path_acc else (index, 0)
            acc_accumulated = (slice(index - 1, index + 1), 0, 0) if '_full_' in path_acc else (slice(index - 1, index + 1), 0)
            timestep.evap = read(acc, 'water_evaporation_amount', acc_accumulated, extended_mask)
            timestep.spechum = read(acc, 'specific_humidity_2m', acc_prefix, extended_mask)

            radiation = datasets.get(path_rad)
            rad_accumulated = (
                (slice(index - 1, index + 1), 0)
                if '_extracted_' in path_rad and '_full_' not in path_acc
                else (slice(index - 1, index + 1), 0, 0)
            )
            timestep.SwD = read(radiation, 'integral_of_surface_net_downward_shortwave_flux_wrt_time', rad_accumulated, extended_mask)
            timestep.andLw = read(radiation, 'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time', rad_accumulated, extended_mask)

        # Preserve the original de-accumulation and unit conversions exactly.
        timestep.SwD = (timestep.SwD[1, :] - timestep.SwD[0, :]) / 3600.0
        timestep.andLw = (timestep.andLw[1, :] - timestep.andLw[0, :]) / 3600.0
        timestep.evap = (timestep.evap[1, :] - timestep.evap[0, :]) / (1000.0 * 3600.0)
        timestep.precip = (timestep.precip[1, :] - timestep.precip[0, :]) / (1000.0 * 3600.0)
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
        nc.createDimension('time', None)
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

        tris                    = FVCOM_grd.tri + 1
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