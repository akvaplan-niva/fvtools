# -----------------------------------------------------------------------------------------
#                  Interpolate data from ROMS to the FVCOM nesting zone
# -----------------------------------------------------------------------------------------
import netCDF4
import numpy as np
import pandas as pd
import progressbar as pb
import matplotlib.pyplot as plt
import multiprocessing as mp

from numba import njit
from time import gmtime, strftime
from functools import cached_property
from datetime import datetime, timedelta
from scipy.spatial import cKDTree as KDTree

# fvtools
from fvtools.nesting import vertical_interpolation as vi
from fvtools.grid.fvcom_grd import FVCOM_grid, NEST_grid
from fvtools.gridding.prepare_inputfiles import write_FVCOM_bath
from fvtools.interpolators.roms_interpolators import N4ROMS, LinearInterpolation
from fvtools.grid.roms_grid import get_roms_grid, NoAvailableData, RomsDownloader, ROMSTimeStep

def main(fvcom_grd,
         nest_grd,
         outfile,
         # Option 1: Time Range (Original functionality)
         start_time = None,
         stop_time  = None,
         # Option 2: Custom Time Mapping (New functionality)
         time_write = None, 
         time_read  = None,
         latlon     = True,
         mother     = None,
         weights    = [2.5e-4, 2.5e-5],
         nprocs     = 20):
    '''
    Nest from NorKyst-800 or NorShelf-2.4km

    mandatory input:
    ----
    fvcom_grd  - FVCOM grid file         (M.mat, M.npy)
    nest_grd   - FVCOM nesting grid file (ngrd.mat, ngrd.npy)
    outfile    - name of the output file
    
    Choose ONE of the following time options:
    - start_time / stop_time: 'yyyy-mm-dd' (Will use make_range_fileList)
    - time_write / time_read: list of 'yyyy-mm-dd-hh' strings (Will use make_custom_fileList)

    latlon     - True/False (latlon = True triggers adjustment of currents for utm-coordinate meridional convergence)

    optional:
    ----
    mother     - ROMS model to nest into
                 - 'HI-NK' or 'MET-NK' for NorKyst-800 
                 - 'H-NS' or 'D-NS' for hourly values or daily averages from NorShelf-2.4km
    weights    - tuple giving the weight interval for the nest nodes and cells. By default, weights = [2.5e-4, 2.5e-5].
    nprocs     - Number of processes to use when downloading (set equal to None to use all available cores)
    '''
    ROMS = get_roms_grid(mother)

    print(f'\nInterpolate data from {ROMS} to {outfile}\n---')
    print('- Load mesh info')
    #M    = FVCOM_grid(fvcom_grd)
    M    = FVCOM_grid(fvcom_grd, reference = 'epsg:3031')
    NEST = NEST_grid(nest_grd, M, proj = M.reference)
    ROMS.Proj = M.Proj
    ROMS.load_grid(NEST.x, NEST.y, offset = NEST.R*6)
    #ROMS.load_grid(NEST.x, NEST.y, offset = NEST.R*50)

    print('- Compute FVCOM nesting nudging coefficients as function of distance from OBC')
    NEST.calcWeights(M, w1 = max(weights), w2 = min(weights))
    
    print('\nMake the filelist\n---')
    # time, path, index = make_fileList(start_time, stop_time, ROMS)
    # --- Logic to choose the file-list generation method ---
    if start_time and stop_time:
        print('  - Using time range mode (start_time/stop_time)')
        time, path, index = make_fileList(start_time, stop_time, ROMS)
    elif time_write and time_read:
        print('  - Using custom time mapping mode (time_write/time_read)')
        time, path, index = make_custom_fileList(time_write, time_read, ROMS)
    else:
        raise ValueError("Must provide either (start_time and stop_time) OR (time_write and time_read).")
    # ------------------------------------------------------

    print('\nFind the nearest 4 interpolation coefficients for the nestingzone\n---')
    print(f"ROMS grid shape: {ROMS.x_rho.shape}")
    #N4R = N4ROMSNESTING(ROMS, x = NEST.x, y = NEST.y, tri = NEST.tri, uv = True)
    N4R = N4ROMSNESTING(ROMS, x = NEST.x, y = NEST.y, tri = NEST.tri, uv = True, land_check=False)
    print(f"Number of ROMS rho points in bounding box: {np.sum(ROMS.fv_rho_mask)}")
    N4R.nearest4(M)
    
    # --- START OF MODIFICATION ----------------------------------jay
    # Land correction (we can't use ROMS land points)
    # ---
    print('    - Adjust interpolation coefficients to remove land')
    N4R.rho_index, N4R.rho_coef = N4R.correct_land(N4R.rho_index, N4R.rho_coef, ROMS.cropped_x_rho, ROMS.cropped_y_rho, M.x, M.y, ROMS.Land_rho)
    if N4R.uv:
        N4R.u_index, N4R.u_coef = N4R.correct_land(N4R.u_index, N4R.u_coef, ROMS.cropped_x_u, ROMS.cropped_y_u, M.xc, M.yc, ROMS.Land_u, zero = True)
        N4R.v_index, N4R.v_coef = N4R.correct_land(N4R.v_index, N4R.v_coef, ROMS.cropped_x_v, ROMS.cropped_y_v, M.xc, M.yc, ROMS.Land_v, zero = True)
    # --- END OF MODIFICATION ------------------------------------jay

    print('\nSet the nestingzone bathymetry equal to ROMS and add a smooth transition zone.\n---')
    Match   = MatchTopo(ROMS, M, NEST, latlon)
    M, NEST = Match.add_ROMS_bathymetry_to_FVCOM_and_NEST()

    print('\nAdd vertical interpolation method\n---')
    N4R.FV  = NEST # to let vertical_interpolation know which FVCOM grid to find vertical interpolation coeficients for
    N4R     = vi.add_vertical_interpolation2N4(N4R)
    create_nc_forcing_file(outfile, NEST, mother, time, latlon, M.reference)

    print('\nInterpolate data from ROMS to the nesting zone\n---')
    R2F = Roms2FVCOMNest(outfile, path, index, N4R, latlon)
    R2F.dump(nprocs = nprocs)

    print('\n--> Fin.')

# ===============================================================================================
#                                        fileList
# ===============================================================================================
def make_fileList(start_time, stop_time, ROMS):
    '''
    Link points in time to files.
    input format: yyyy-mm-dd-hh

    The filelist is used to decide what files to use when forecasts are available, and implicitly
    to "remember" what files are available in case of server downtime during the later interpolation step.
    '''
    # List of dates to look for files
    # ----
    start = datetime(int(start_time.split('-')[0]), int(start_time.split('-')[1]), int(start_time.split('-')[2]))
    stop  = datetime(int(stop_time.split('-')[0]), int(stop_time.split('-')[1]), int(stop_time.split('-')[2]))
    dates = pd.date_range(start,stop)

    # Initialize some arrays we will be appending to
    # ----
    time  = np.empty(0); path  = []; index = []; file = []

    # See where the files are available, create a list referencing files and indices to timestamps
    # ----
    for date in dates:
        try:
            file = ROMS.test_day(date)

        except NoAvailableData:
            print(f'    - warning: {date} not found - your forcing file will have a gap')
            continue

        print(f'  - checking: {file}')
        d        = netCDF4.Dataset(file)
        t_roms   = netCDF4.num2date(d.variables['ocean_time'][:], units = d.variables['ocean_time'].units, only_use_cftime_datetimes=False)
        t_fvcom  = netCDF4.date2num(t_roms, units = 'days since 1858-11-17 00:00:00')

        # Append the timesteps, paths and indices to a fileList
        # ----
        time     = np.append(time, t_fvcom)
        path     = path + [file]*len(t_fvcom)
        index.extend(list(range(len(t_fvcom))))

    if len(index)==0:
        raise NoAvailableData('We did not find data for your period of interest')

    # Remove overlap
    # ----
    time_no_overlap  = [time[-1]]
    path_no_overlap  = [path[-1]]
    index_no_overlap = [index[-1]]

    for n in range(len(time)-1, 0, -1):
        if time[n-1] < time_no_overlap[0]:
            time_no_overlap.insert(0, time[n-1])
            path_no_overlap.insert(0, path[n-1])
            index_no_overlap.insert(0, index[n-1])

    return np.array(time_no_overlap), path_no_overlap, index_no_overlap

# ===============================================================================================
#                                        fileList (Custom Mode)
# ===============================================================================================
def make_custom_fileList(time_write, time_read, ROMS):
    '''
    (New Functionality) Link FVCOM output times (time_write, chronological) 
    to ROMS input data (time_read, arbitrary order).
    
    input time format: yyyy-mm-dd-hh
    '''
    if len(time_write) != len(time_read):
        raise ValueError('time_write and time_read must have the same number of elements.')

    successful_steps = []
    path_cache = {} 
    
    print('  - Processing ROMS input times from time_read list...')

    # Iterate over both lists simultaneously
    for i, (t_write_str, t_read_str) in enumerate(zip(time_write, time_read)):
        
        try:
            # 1. Prepare FVCOM output time (MJD)
            parts_write = t_write_str.split('-')
            time_dt_write = datetime(int(parts_write[0]), int(parts_write[1]), int(parts_write[2]), int(parts_write[3]))
            time_mjd_write = netCDF4.date2num(time_dt_write, units = 'days since 1858-11-17 00:00:00')

            # 2. Find ROMS data time and file path
            parts_read = t_read_str.split('-')
            read_date_str = '-'.join(parts_read[:3])
            read_hour = int(parts_read[3])
            read_date_dt = datetime(int(parts_read[0]), int(parts_read[1]), int(parts_read[2]))

            # Get ROMS file path (will raise NoAvailableData if file not found)
            roms_file_path = ROMS.test_day(read_date_dt)

            # 3. Load file time data or use cache
            if roms_file_path not in path_cache:
                d = netCDF4.Dataset(roms_file_path)
                t_roms = netCDF4.num2date(d.variables['ocean_time'][:], units = d.variables['ocean_time'].units, only_use_cftime_datetimes=False)
                # Store a list of hours for quick lookup
                path_cache[roms_file_path] = [t.hour for t in t_roms]
                d.close()

            # 4. Find the ROMS index corresponding to the requested hour
            try:
                roms_index = path_cache[roms_file_path].index(read_hour)
            except ValueError:
                # Hour not found in the file
                print(f'    - warning: Hour {read_hour:02d} not found in {roms_file_path}. Skipping time_write: {t_write_str}')
                continue
            
            # 5. Record the successful step
            successful_steps.append((time_mjd_write, roms_file_path, roms_index))

        except NoAvailableData:
            print(f'    - warning: Data not found for {read_date_str}. Skipping output time: {t_write_str}')
            continue
        except Exception as e:
            print(f'    - An unexpected error occurred while processing {t_read_str} for output time {t_write_str}: {e}. Skipping.')
            continue

    if not successful_steps:
        raise NoAvailableData('We did not find data for any of your time_read inputs.')

    # Unpack the list of tuples into the three required return arrays
    time_out_mjd = np.array([step[0] for step in successful_steps])
    path_list    = [step[1] for step in successful_steps]
    index_list   = [step[2] for step in successful_steps]

    return time_out_mjd, path_list, index_list

# --------------------------------------------------------------------------------------
#                             Grid and interpolation objects
# --------------------------------------------------------------------------------------
class N4nestdepth:
    '''
    the depth we interpolate to depends on the horizontal interpolation scheme
    '''
    @property
    def fvcom_rho_dpt(self):
        return np.sum(self.ROMS.h_rho[self.ROMS.fv_rho_mask][self.rho_index] * self.rho_coef, axis=1)

    @property
    def fvcom_u_dpt(self):
        return np.sum(self.ROMS.h_u[self.ROMS.fv_u_mask][self.u_index] *self.u_coef, axis=1)

    @property
    def fvcom_v_dpt(self):
        return np.sum(self.ROMS.h_v[self.ROMS.fv_v_mask][self.v_index] *self.v_coef, axis=1)

class N4ROMSNESTING(N4ROMS, N4nestdepth):
    def __init__(self, ROMS, x = None, y = None, tri = None, uv = False, land_check=False):
    #def __init__(self, ROMS, x = None, y = None, tri = None, uv = False, land_check=True):
        '''
        Initialize empty attributes
        '''
        self.x, self.y = x, y
        self.tri = tri
        self.uv = uv
        self.land_check = land_check
        self.ROMS = ROMS
        
    # copied and pasted this from interpol_roms_restart.py ---------------------jay
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

    def dump_single(self):
        '''
        Mostly for debug/illustration of the downloading/interpolation/writing process
        '''
        widget = [f'Downloading, interpolating and dumping timesteps to nest: ', pb.Percentage(), pb.BouncingBar(), pb.ETA()]
        bar = pb.ProgressBar(widgets=widget, maxval=len(self.path))
        bar.start()
        with netCDF4.Dataset(self.outfile, 'r+') as self.out:
            for counter, (path_here, index_here) in enumerate(zip(self.path, self.index)):
                bar.update(counter)
                timestep = self.read_timestep(index_here, path_here)
                timestep = self.crop_and_transpose(timestep)
                timestep = self.horizontal_interpolation(timestep)
                timestep = self.vertical_interpolation(timestep)
                timestep = self.adjust_uv(timestep) 
                timestep.counter = counter
                self.dump_timestep_to_nest(timestep)
        bar.finish()

    def dump(self, nprocs = None):
        '''
        loop over all timesteps and interpolate them to the FVCOM nesting file
        '''
        manager = mp.Manager() # We need a manager since all processes (minus one) will communicate with the writer
        q = manager.Queue()    # q is convey messages from the workers to the listener
        if nprocs is None:
            pool = mp.Pool(mp.cpu_count()+2)
        else:
            pool = mp.Pool(nprocs+2) # +2 to make sure that the listener and the watcher does not overpopulate the list

        # Put listener to work
        # ----
        watcher = pool.apply_async(self._listener, (q,))

        # Fire off workers
        # ----
        jobs = []
        for counter, (path_here, index_here) in enumerate(zip(self.path, self.index)):
            job = pool.apply_async(self._worker, (counter, path_here, index_here, q))
            jobs.append(job)

        # Collect results from the workers through the pool result queue
        # ----
        for job in jobs: 
            job.get()

        #now we are done, kill the listener
        q.put('kill')
        pool.close()
        pool.join()
        

    # Multi processing workers, listeners and managers
    # ----
    def _worker(self, counter, path_here, index_here, q):
        '''
        Workers download data and interpolate it to the FVCOM grid
        '''
        timestep = self.read_timestep(index_here, path_here)
        timestep = self.crop_and_transpose(timestep)
        timestep = self.horizontal_interpolation(timestep)
        timestep = self.vertical_interpolation(timestep)
        timestep = self.adjust_uv(timestep)
        timestep.counter = counter    # Referencing where in the nestingfile we will store this timestep
        q.put(timestep, timeout = 10)
        return ''

    def _listener(self, q):
        '''
        The listener has write access to the netCDF and is responsible for dumping
        '''
        widget = [f'  Downloading timesteps: ', pb.Percentage(), pb.BouncingBar(), pb.ETA()]
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
        self.out.variables['hyw'][timestep.counter, :, :] = np.zeros((1, len(self.N4.siglev[:,0]), len(self.N4.siglev[0,:]))) # assume h/L << 1 in the nestingzone

        # Hydrography
        self.out.variables['zeta'][timestep.counter, :]        = timestep.zeta
        self.out.variables['temp'][timestep.counter, :, :]     = timestep.temp
        self.out.variables['salinity'][timestep.counter, :, :] = timestep.salt

        # Generic (in theory, one should be able to use a time-dependent weight in the nestingzone, I've never seen it done though)
        self.out.variables['weight_node'][timestep.counter, :] = self.N4.weight_node
        self.out.variables['weight_cell'][timestep.counter, :] = self.N4.weight_cell

class MatchTopo:
    def __init__(self, ROMS, M, NEST, latlon):
        '''
        Rutine matching ROMS depth with FVCOM depth, same-same in nestingzone,
        but a smooth transition from ROMS to BuildCase topo on the outside of it
        '''
        self.M = M 
        self.ROMS = ROMS
        self.NEST = NEST
        self.latlon = latlon

    @property
    def R_edge_of_nestingzone(self):
        '''
        R_edge_of_nestingzone is the radius from obc nodes where we will modify the bathymetry to be identical to ROMS
        '''
        return 2*self.NEST.R

    @property
    def R_edge_of_smoothing_zone(self):
        '''
        Distance from the OBC where we will use the bathymetry computed by BuildCase
        '''
        return 6*self.NEST.R
    
    @cached_property
    def Rdst(self):
        '''
        The distance each node in the mesh is from the OBC
        '''
        obc_tree = KDTree(np.array([self.M.x[self.M.obc_nodes], self.M.y[self.M.obc_nodes]]).transpose())
        Rdst, _ = obc_tree.query(np.array([self.M.x, self.M.y]).transpose())
        return Rdst

    @property
    def nestingzone_nodes(self):
        '''
        Nodes where we will set h equal to h_roms
        '''
        return np.where(self.Rdst <= self.R_edge_of_nestingzone)[0]

    @property
    def transition_nodes(self):
        '''
        Nodes where we linearly (based on distance from the nestingzone) transition from ROMS depth to FVCOM depth
        '''
        return np.where(np.logical_and(self.Rdst > self.R_edge_of_nestingzone, self.Rdst <= self.R_edge_of_smoothing_zone))[0]

    @property
    def nodes_to_change(self):
        '''
        All nodes in the mesh where we will change the bathymetry
        '''
        return np.where(self.Rdst <= self.R_edge_of_smoothing_zone)[0]

    @cached_property
    def weight(self):
        '''
        weights for creathing a smooth bathymetry transition from ROMS to FVCOM near the nestingzone
        '''
        weight = np.zeros(self.M.x.shape)
        R_outer_edge = np.max(self.Rdst[self.transition_nodes])

        # Compute weights in the transition zone
        width_transition = self.Rdst[self.nestingzone_nodes].max() - R_outer_edge
        a = 1.0/width_transition
        b = R_outer_edge/width_transition
        weight[self.transition_nodes]  = a*self.Rdst[self.transition_nodes] - b

        # Just to make sure that the nestingzone is 1
        weight[self.nestingzone_nodes] = 1
        return weight

    def add_ROMS_bathymetry_to_FVCOM_and_NEST(self):
        '''
        Add ROMS bathymetry to the nestingzone, create a smooth transition from ROMS bathymetry to FVCOM bathymetry. 
        Writes new bathymetry to _dpt.dat file and updates M.npy
        '''
        # No need to re-do this step if we already have done it...
        try:
            #if self.M.info['true if updated by roms_nesting_fg'] == True:
            if self.M.true_if_updated_by_roms_nesting_fg == True:
                print('  - This experiment has already been matched with ROMS bathymetry, skipping step')
                self.match_depth_in_nest_and_model()
                return self.M, self.NEST
        except:
            pass

        # Prepare interpolator, add ROMS bathymetry to the transition-zone in h_roms
        N4B = N4ROMSNESTING(self.ROMS, x = self.M.x[self.nodes_to_change], y = self.M.y[self.nodes_to_change], uv = False, land_check = False) # the depth at ROMS land is equal to min_depth
        N4B.nearest4()

        # Make copy of FVCOM bathymetry, set depth in the "to change range" equal to ROMS bathy
        h_roms = np.copy(self.M.h)
        h_roms[self.nodes_to_change] = np.sum(N4B.ROMS.h_rho[N4B.ROMS.fv_rho_mask][N4B.rho_index]*N4B.rho_coef, axis=1)

        # Update the nodes by to their distance from the obc
        self.M.h  = h_roms*self.weight + self.M.h*(1-self.weight)
        self.match_depth_in_nest_and_model()

        # Store the smoothed bathymetry to the mesh (ends up in casename_dep.dat)
        self.store_bathymetry()
        return self.M, self.NEST

    def match_depth_in_nest_and_model(self):
        '''
        Ensures that the nest and the mother model has the same bathymetry
        '''
        ind = self.M.find_nearest(self.NEST.x, self.NEST.y)
        self.NEST.h = self.M.h[ind]

    def store_bathymetry(self):
        '''
        First to FVCOM readable input file, then update the M.npy file with the correct bathymetry
        '''
        #self.M.write_bath(filename = f"./input/{self.M.info['casename']}_dep.dat", latlon = self.latlon)
        self.M.write_bath(filename = f"./input/{self.M.casename}_dep.dat", latlon = self.latlon)
        print('- Updating M.npy with the new bathymetry')
        #self.M.info['true if updated by roms_nesting_fg'] = True # for later
        self.M.true_if_updated_by_roms_nesting_fg = True # for later
        self.M.to_npy()

# ==============================================================================================
#                            Write an empty output file
# ==============================================================================================
def create_nc_forcing_file(name, NEST, mother, timesteps, latlon, epsg):
    '''
    Creates empty nc file formatted to fit FVCOM open boundary ocean forcing
    '''
    nc = netCDF4.Dataset(name, 'w', format='NETCDF4')

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

    # Create dimensions
    # ----
    nc.createDimension('time', 0)
    nc.createDimension('node', len(NEST.xn))
    nc.createDimension('nele', len(NEST.xc))
    nc.createDimension('three', 3)
    nc.createDimension('siglay', len(NEST.siglay[0,:]))
    nc.createDimension('siglev', len(NEST.siglev[0,:]))

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

    # Initialize the time (so that we can multiprocess the download)
    for counter, fvcom_time in enumerate(timesteps):
        nc.variables['time'][counter] = fvcom_time
        nc.variables['Itime'][counter] = np.floor(fvcom_time)
        nc.variables['Itime2'][counter] = np.round((fvcom_time - np.floor(fvcom_time)) * 60 * 60 * 1000, decimals = 0)*24
    nc.close()