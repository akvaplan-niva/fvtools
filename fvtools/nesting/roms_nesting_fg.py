# -----------------------------------------------------------------------------------------
#                  Interpolate data from ROMS to the FVCOM nesting zone
# -----------------------------------------------------------------------------------------
import time as time_module
import netCDF4
import numpy as np
import pandas as pd
import progressbar as pb
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from uuid import uuid4

from time import gmtime, strftime

from datetime import datetime

# fvtools
from fvtools.nesting import vertical_interpolation as vi
from fvtools.grid.fvcom_grd import FVCOM_grid, NEST_grid
from fvtools.interpolators.roms_interpolators import N4ROMSNESTING, LinearInterpolation
from fvtools.nesting.matchtopo import MatchTopo
from fvtools.grid.roms_grid import get_roms_grid, NoAvailableData, RomsDownloader, ROMSTimeStep

def main(fvcom_grd,
         nest_grd,
         outfile,
         start_time,
         stop_time,
         latlon  = True,
         mother  = None,
         weights = [2.5e-4, 2.5e-5],
         nprocs = 1):
    '''
    Nest from NorKyst-800 or NorShelf-2.4km

    mandatory input:
    ----
    fvcom_grd  - FVCOM grid file         (M.mat, M.npy)
    nest_grd   - FVCOM nesting grid file (ngrd.mat, ngrd.npy)
    outfile    - name of the output file
    start_time - yyyy-mm-dd
    stop_time  - yyyy-mm-dd
    latlon     - True/False (latlon = True triggers adjustment of currents for utm-coordinate meridional convergence)

    optional:
    ----
    mother     - ROMS model to nest into
                 - 'HI-NK'    for NorKyst-800 version 1
                 - 'NKv2' for NorKyst-800 version 2, with data from 2017-02-20 to 2025-10-17
                 - 'NKv3' for NorKyst-800 version 3, with data from 2023-05-07 to present
                 - 'H-NS' or 'D-NS' for hourly values or daily averages from NorShelf-2.4km
                 - 'IMR-NF'   for NorFjords-150 data from the IMR. Assu
    weights    - tuple giving the weight interval for the nest nodes and cells. By default, weights = [2.5e-4, 2.5e-5].
    nprocs     - Number of processes to use when downloading (set equal to None to use all available cores)
                - set to 1 by default, since thredds should not be accessed by multiple workers. It can be helpfull to use
                  more processes if the files you're accessing are stored locally
    '''
    ROMS = get_roms_grid(mother)

    start = datetime(int(start_time.split('-')[0]), int(start_time.split('-')[1]), int(start_time.split('-')[2]))
    stop  = datetime(int(stop_time.split('-')[0]), int(stop_time.split('-')[1]), int(stop_time.split('-')[2]))

    print(f'\nInterpolate data from {ROMS} to {outfile}\n---')
    print('- Load mesh info')
    M    = FVCOM_grid(fvcom_grd)
    NEST = NEST_grid(nest_grd, M, proj = M.reference)

    # Add a projection, load the part of the full ROMS grid which covers the FVCOM nesting zone
    ROMS.Proj = M.Proj
    ROMS.load_grid(NEST.x, NEST.y, offset = NEST.R*6, date = start)

    print('- Compute FVCOM nesting nudging coefficients as function of distance from OBC')
    NEST.calcWeights(M, w1 = max(weights), w2 = min(weights))
    
    print('\nMake the filelist\n---')
    time, path, index = make_fileList(start, stop, ROMS)

    N4R, M, NEST = _prepare_nesting_interpolator(ROMS, M, NEST, latlon)
    _publish_nesting_file(outfile, NEST, mother, time, latlon,
                          M.reference, path, index, N4R, nprocs)

    print('\n--> Fin.')

# Phase 4: shared ROMS timestep and output orchestration.
def _prepare_nesting_interpolator(ROMS, M, NEST, latlon):
    """Prepare horizontal, bathymetry and static sigma weights in legacy order."""
    print('\nFind the nearest 4 interpolation coefficients for the nestingzone\n---')
    N4R = N4ROMSNESTING(ROMS, x = NEST.x, y = NEST.y, tri = NEST.tri, uv = True)
    N4R.nearest4(M)

    print('\nSet the nestingzone bathymetry equal to ROMS and add a smooth transition zone.\n---')
    # Will only be done once for each model, there is a flag in M.npy which skips this step if it already has been done once.
    Match   = MatchTopo(ROMS, M, NEST, latlon)
    M, NEST = Match.add_ROMS_bathymetry_to_FVCOM_and_NEST()

    print('\nAdd vertical interpolation method\n---')
    N4R.FV  = NEST # to let vertical_interpolation know which FVCOM grid to find vertical interpolation coeficients for
    N4R     = vi.add_vertical_interpolation2N4(N4R)
    return N4R, M, NEST


def _publish_nesting_file(outfile, NEST, mother, time, latlon, epsg,
                          path, index, N4R, nprocs):
    """Write a complete forcing file atomically, retaining the previous file on failure."""
    # Write to a sibling temporary file, publishing only a complete product.
    # The preexisting nesting file remains intact if an input fails.
    target = Path(outfile)
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.{uuid4().hex}.partial')
    try:
        create_nc_forcing_file(str(temporary), NEST, mother, time, latlon, epsg)
        print('\nInterpolate data from ROMS to the nesting zone\n---')
        R2F = Roms2FVCOMNest(str(temporary), path, index, N4R, latlon)
        if nprocs == 1:
            print('  - Running in single mode')
            R2F.dump_single()
        else:
            R2F.dump(nprocs=nprocs)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)



# ===============================================================================================
#                                        fileList
# ===============================================================================================
def make_fileList(start, stop, ROMS):
    '''
    Link points in time to files.
    input format: datetime

    The filelist is used to decide what files to use when forecasts are available, and implicitly
    to "remember" what files are available in case of server downtime during the later interpolation step.
    '''
    # List of dates to look for files
    # ----
    dates = pd.date_range(start, stop)

    # Initialize some arrays we will be appending to
    # ----
    time  = np.empty(0); path  = []; index = []; file = []

    # See where the files are available, create a list referencing files and indices to timestamps
    # ----
    for date in dates:
        # Pause two second to avoid being stopped
        time_module.sleep(2)
        try:
            file = ROMS.test_day(date)

        except NoAvailableData:
            print(f'    - warning: {date} not found - your forcing file will have a gap')
            continue

        print(f'  - checking: {file}')
        with netCDF4.Dataset(file) as d:
            ocean_time = d.variables['ocean_time']
            t_roms = netCDF4.num2date(ocean_time[:], units=ocean_time.units,
                                       only_use_cftime_datetimes=False)
            t_fvcom = np.atleast_1d(netCDF4.date2num(
                t_roms, units='days since 1858-11-17 00:00:00'))

        # Append the timesteps, paths and indices to a fileList
        # ----
        time     = np.append(time, t_fvcom)
        path     = path + [file]*len(t_fvcom)
        index.extend(list(range(len(t_fvcom))))

    if len(index)==0:
        raise NoAvailableData('We did not find data for your period of interest')

    # Files may overlap *and* contain out-of-order forecasts. The reverse
    # scan used previously discarded valid intermediate timesteps after overlap.
    # Keep the last file occurrence for duplicate timestamps, as before.
    if not np.isfinite(time).all():
        raise ValueError('ROMS file list contains non-finite timestamps')
    latest = {float(t): (p, int(k)) for t, p, k in zip(time, path, index)}
    ordered = sorted(latest)
    return (np.asarray(ordered, dtype=np.float64),
            [latest[t][0] for t in ordered],
            [latest[t][1] for t in ordered])

# --------------------------------------------------------------------------------------
#                             Grid and interpolation objects
# --------------------------------------------------------------------------------------
# Downloading and interpolating to nesting-file
class Roms2FVCOMNest(RomsDownloader, LinearInterpolation):
    '''
    Class writing data to nesting-file.
    - Downloading ROMS data
    - Interpolating to FVCOM
    '''
    def __init__(self, outfile, path, index, N4, latlon = False):
        '''
        outfile: Name of file to write to
        time:    list of timesteps to write to
        path:    list of path to file to read timestep from
        index:   list of indices to read from file for each timestep
        N4:      Class with Nearest4 interpolation coefficients
        latlon:  True when running FVCOM in spherical mode
        '''
        self.path = path; 
        self.index = index
        self.outfile = outfile
        self.N4 = N4.dump()
        self.latlon = latlon

    def _prepare_timestep(self, counter, path_here, index_here):
        """Run exactly the same interpolation stages in serial and parallel mode."""
        timestep = self.read_timestep(index_here, path_here)
        timestep = self.crop_and_transpose(timestep)
        timestep = self.horizontal_interpolation(timestep)
        timestep = self.vertical_interpolation(timestep)
        timestep = self.adjust_uv(timestep)
        timestep.counter = counter    # Referencing where in the nestingfile we will store this timestep
        return timestep

    def dump_single(self):
        '''
        Mostly for debug/illustration of the downloading/interpolation/writing process
        '''
        widget = [f'  - Interpolating and dumping timesteps to nest: ', pb.Percentage(), pb.BouncingBar(), pb.ETA()]
        bar = pb.ProgressBar(widgets=widget, maxval=len(self.path))
        bar.start()
        # Phase 6A: the serial reader keeps only the current ROMS file open.
        with self.reuse_roms_datasets(), netCDF4.Dataset(self.outfile, 'r+') as self.out:
            for counter, (path_here, index_here) in enumerate(zip(self.path, self.index)):
                bar.update(counter)
                timestep = self._prepare_timestep(counter, path_here, index_here)
                self.dump_timestep_to_nest(timestep)
        bar.finish()

    def dump(self, nprocs=None):
        """Process steps in parallel, but write all NetCDF data in this process.

        Only up to 2*workers future results are outstanding; workers do not
        open the output file. Exceptions propagate instead of leaving a
        separate writer waiting forever on a queue.
        """
        if nprocs is None:
            # Respect CPU affinity where supported rather than spawning 256
            # workers from a constrained Jupyter/Slurm process.
            workers = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                       else mp.cpu_count())
        else:
            workers = int(nprocs)
            if workers != nprocs:
                raise ValueError('nprocs must be an integer')
        if workers < 1:
            raise ValueError('nprocs must be at least 1')
        bar = pb.ProgressBar(
            widgets=['  - Interpolating and dumping timesteps to nest: ',
                     pb.Percentage(), pb.BouncingBar(), pb.ETA()],
            maxval=len(self.path))
        bar.start()
        # Spawn is safer after NetCDF/HDF5 access, and works on Windows.
        ctx = mp.get_context('spawn')
        count = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            source = iter(enumerate(zip(self.path, self.index)))
            pending = {}
            def submit_next():
                item = next(source, None)
                if item is None:
                    return False
                counter, (path_here, index_here) = item
                future = pool.submit(self._worker, counter, path_here, index_here)
                pending[future] = counter
                return True

            for _ in range(min(len(self.path), workers * 2)):
                submit_next()
            with netCDF4.Dataset(self.outfile, 'r+') as self.out:
                while pending:
                    ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        pending.pop(future)
                        timestep = future.result()  # raises worker errors in parent
                        self.dump_timestep_to_nest(timestep)
                        count += 1
                        bar.update(count)
                        submit_next()
        bar.finish()

    # Multi processing workers, listeners and managers
    # ----
    def _worker(self, counter, path_here, index_here, q=None):
        '''
        Workers download data and interpolate it to the FVCOM grid
        '''
        timestep = self._prepare_timestep(counter, path_here, index_here)
        if q is None:
            return timestep
        q.put(timestep, timeout=10)
        return ''

    def _listener(self, q):
        '''
        The listener has write access to the netCDF and is responsible for dumping
        '''
        widget = [f'  - Interpolating and dumping timesteps to nest: ', pb.Percentage(), pb.BouncingBar(), pb.ETA()]
        bar = pb.ProgressBar(widgets=widget, maxval=len(self.path))
        i=0
        bar.start()
        with netCDF4.Dataset(self.outfile, 'r+') as self.out:
            while True:
                timestep = q.get() # we expect to get timesteps from the workers
                if timestep == 'kill':
                    bar.finish()
                    break

                if type(timestep) == ROMSTimeStep:
                    i += 1
                    bar.update(i)
                    self.dump_timestep_to_nest(timestep)

    def dump_timestep_to_nest(self, timestep):
        '''
        Write the data to the output file
        '''
        # Velocities
        self.out.variables['u'][timestep.counter, :, :]   = timestep.u
        self.out.variables['v'][timestep.counter, :, :]   = timestep.v
        self.out.variables['ua'][timestep.counter, :]     = timestep.ua
        self.out.variables['va'][timestep.counter, :]     = timestep.va
        self.out.variables['hyw'][timestep.counter, :, :] = 0.0  # keep legacy h/L << 1 approximation

        # Hydrography
        self.out.variables['zeta'][timestep.counter, :]        = timestep.zeta
        self.out.variables['temp'][timestep.counter, :, :]     = timestep.temp
        self.out.variables['salinity'][timestep.counter, :, :] = timestep.salt

        # Generic (in theory, one should be able to use a time-dependent weight in the nestingzone, I've never seen it done though)
        self.out.variables['weight_node'][timestep.counter, :] = self.N4.weight_node
        self.out.variables['weight_cell'][timestep.counter, :] = self.N4.weight_cell

# ==============================================================================================
#                            Write an empty output file
# ==============================================================================================
def create_nc_forcing_file(name, NEST, mother, timesteps, latlon, epsg):
    '''
    Creates empty nc file formatted to fit FVCOM open boundary ocean forcing
    '''
    with netCDF4.Dataset(name, 'w', format='NETCDF4') as nc:
        _write_global_attributes(nc, mother, latlon, epsg)
        _create_dimensions(nc, NEST)
        _create_nesting_variables(nc)
        _write_grid_metadata(nc, NEST)
        _write_nesting_times(nc, timesteps)


def _write_global_attributes(nc, mother, latlon, epsg):
    # Write global attributes
    # ----
    nc.title        = 'FVCOM Nesting File'
    nc.institution  = 'Akvaplan-niva AS'
    nc.source       = 'FVCOM grid (unstructured) nesting file'
    nc.created      = f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} using roms_nesting_fg.py'
    nc.mother_model = f'Using data nested from {mother}'

    if latlon:
        nc.interpolation_projection = 'degrees'
    else:
        nc.interpolation_projection = epsg


def _create_dimensions(nc, NEST):
    # Create dimensions
    # ----
    nc.createDimension('time', 0)
    nc.createDimension('node', len(NEST.xn))
    nc.createDimension('nele', len(NEST.xc))
    nc.createDimension('three', 3)
    nc.createDimension('siglay', len(NEST.siglay[0,:]))
    nc.createDimension('siglev', len(NEST.siglev[0,:]))


def _create_nesting_variables(nc):
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

    # positions
    # ----
    # node
    lon                = nc.createVariable('lon', 'single', ('node',))
    lat                = nc.createVariable('lat', 'single', ('node',))
    x                  = nc.createVariable('x', 'single', ('node',))
    y                  = nc.createVariable('y', 'single', ('node',))
    h                  = nc.createVariable('h', 'single', ('node',))

    # center
    lonc               = nc.createVariable('lonc', 'single', ('nele',))
    latc               = nc.createVariable('latc', 'single', ('nele',))
    xc                 = nc.createVariable('xc', 'single', ('nele',))
    yc                 = nc.createVariable('yc', 'single', ('nele',))
    hc                 = nc.createVariable('h_center', 'single', ('nele',))

    # grid parameters
    # ----
    nv                 = nc.createVariable('nv', 'int32', ('three', 'nele',))

    # node
    lay                = nc.createVariable('siglay','single',('siglay','node',))
    lev                = nc.createVariable('siglev','single',('siglev','node',))

    # center
    lay_center         = nc.createVariable('siglay_center','single',('siglay','nele',))
    lev_center         = nc.createVariable('siglev_center','single',('siglev','nele',))

    # Weight coefficients (since we are nesting from ROMS)
    # ----
    wc                 = nc.createVariable('weight_cell','single',('time','nele',))
    wn                 = nc.createVariable('weight_node','single',('time','node',))

    # time dependent variables
    # ----
    zeta               = nc.createVariable('zeta', 'single', ('time', 'node',))
    ua                 = nc.createVariable('ua', 'single', ('time', 'nele',))
    va                 = nc.createVariable('va', 'single', ('time', 'nele',))
    u                  = nc.createVariable('u', 'single', ('time', 'siglay', 'nele',))
    v                  = nc.createVariable('v', 'single', ('time', 'siglay', 'nele',))
    temp               = nc.createVariable('temp', 'single', ('time', 'siglay', 'node',))
    salt               = nc.createVariable('salinity', 'single', ('time', 'siglay', 'node',))
    hyw                = nc.createVariable('hyw', 'single', ('time', 'siglev', 'node',))


def _write_grid_metadata(nc, NEST):
    # dump the grid metrics
    # ----
    nc.variables['lat'][:]           = NEST.lat
    nc.variables['lon'][:]           = NEST.lon
    nc.variables['latc'][:]          = NEST.latc
    nc.variables['lonc'][:]          = NEST.lonc
    nc.variables['x'][:]             = NEST.xn
    nc.variables['y'][:]             = NEST.yn
    nc.variables['xc'][:]            = NEST.xc
    nc.variables['yc'][:]            = NEST.yc
    nc.variables['h'][:]             = NEST.h
    nc.variables['h_center'][:]      = NEST.hc
    tris                             = NEST.nv+1
    nc.variables['nv'][:]            = tris.transpose()
    nc.variables['siglev'][:]        = NEST.siglev.transpose()
    nc.variables['siglay'][:]        = NEST.siglay.transpose()
    nc.variables['siglev_center'][:] = NEST.siglev_center.transpose()
    nc.variables['siglay_center'][:] = NEST.siglay_center.transpose()


def _write_nesting_times(nc, timesteps):
    # Initialize the time (so that we can multiprocess the download)
    for counter, fvcom_time in enumerate(timesteps):
        nc.variables['time'][counter] = fvcom_time
        nc.variables['Itime'][counter] = np.floor(fvcom_time)
        nc.variables['Itime2'][counter] = np.round((fvcom_time - np.floor(fvcom_time)) * 60 * 60 * 1000, decimals = 0)*24
