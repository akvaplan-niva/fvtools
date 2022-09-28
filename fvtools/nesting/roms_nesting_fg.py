# -----------------------------------------------------------------------------------------
#                  Interpolate data from ROMS to the FVCOM nesting zone
# -----------------------------------------------------------------------------------------
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numba import njit
from time import gmtime, strftime
from functools import cached_property
from datetime import datetime, timedelta
from scipy.spatial import cKDTree as KDTree
from fvtools.grid.fvcom_grd import FVCOM_grid, NEST_grid
from fvtools.grid.roms_grid import get_roms_grid, NoAvailableData, RomsDownloader
from fvtools.gridding.prepare_inputfiles import write_FVCOM_bath
from fvtools.nesting import vertical_interpolation as vi
from fvtools.interpolators.nearest4 import N4

def main(fvcom_grd,
         nest_grd,
         outfile,
         start_time,
         stop_time,
         mother  = None,
         weights = [2.5e-4, 2.5e-5],
         store_bath = True):
    '''
    Nest from NorKyst-800 or NorShelf-2.4km

    mandatory input:
    ----
    fvcom_grd  - FVCOM grid file         (M.mat, M.npy)
    nest_grd   - FVCOM nesting grid file (ngrd.mat, ngrd.npy)
    outfile    - name of the output file
    start_time - yyyy-mm-dd
    stop_time  - yyyy-mm-dd

    optional:
    ----
    mother     - ROMS model to nest into
                 - 'HI-NK' or 'MET-NK' for NorKyst-800 
                 - 'H-NS' or 'D-NS' for hourly values or daily averages from NorShelf-2.4km

    weights    - tuple giving the weight interval for the nest nodes and cells. By default, weights = [2.5e-4, 2.5e-5].
    
    store_bath - True by default, but keep in mind that you only need to do this once pr experiment
    '''
    # Get grid info
    # -----------------------------------------------------------------------------
    print(f'Writing {outfile}\n====')
    print('- Load mesh info')
    M    = FVCOM_grid(fvcom_grd)   # The full fvcom grid. Needed to get OBC nodes and vertical coordinates.
    NEST = NEST_grid(nest_grd, M)  # The grid that receives ROMS data (ngrd)
    ROMS = get_roms_grid(mother, M.Proj)

    print('- Compute FVCOM nesting nudging coefficients as function of distance from OBC')
    NEST.calcWeights(M, w1 = max(weights), w2 = min(weights))
    
    print('\nMake the filelist')
    time, path, index = make_fileList(start_time, stop_time, ROMS)

    # Load the ROMS grid, and identify parts covering FVCOM nodes (just nodes, since cells are always in bounds of nodes)
    ROMS.load_grid(NEST.xn[:,0], NEST.yn[:,0], offset = NEST.R*6)

    print('\nFind the nearest 4 interpolation coefficients for the nestingzone')
    N4R = N4ROMS(ROMS, x = NEST.xn[:,0], y = NEST.yn[:,0], 
                       xc = NEST.xc[:,0], yc = NEST.yc[:,0])
    N4R.nearest4()

    print('- Set the nestingzone bathymetry equal to ROMS and add a smooth transition zone. Write to nestingfile')
    Match    = MatchTopo(ROMS, M, NEST, store_bath)
    M, NEST  = Match.add_ROMS_bathymetry_to_FVCOM_and_NEST()
    N4R.NEST = NEST

    print('- Add vertical interpolation method')
    N4R = vi.add_vertical_interpolation2N4(N4R)
    create_nc_forcing_file(outfile, NEST)

    print('\nInterpolate data from ROMS to the nesting zone')
    R2F = Roms2FVCOMNest(outfile, time, path, index, N4R)
    R2F.dump()

    print('\nCompute the vertically averaged velocities')
    vi.calc_uv_bar(outfile, NEST, M)

    print('\n--> Fin.')

# ===============================================================================================
#                                        fileList
# ===============================================================================================
def make_fileList(start_time, stop_time, ROMS):
    '''
    Link points in time to files.
    input format: yyyy-mm-dd-hh
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
            print(f'- warning: {date} not found - your forcing file will have a gap')
            continue

        print(f'- checking: {file}')
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
    time_no_overlap     = [time[-1]]
    path_no_overlap     = [path[-1]]
    index_no_overlap    = [index[-1]]

    for n in range(len(time)-1, 0, -1):
        if time[n-1] < time_no_overlap[0]:
            time_no_overlap.insert(0, time[n-1])
            path_no_overlap.insert(0, path[n-1])
            index_no_overlap.insert(0, index[n-1])

    return np.array(time_no_overlap), path_no_overlap, index_no_overlap

# --------------------------------------------------------------------------------------
#                             Grid and interpolation objects
# --------------------------------------------------------------------------------------
class N4ROMS(N4):
    '''
    Object with indices and coefficients for ROMS to FVCOM interpolation using bilinear coefficients
    '''
    def __init__(self, ROMS, x = None, y = None, xc = None, yc = None, land_check=True):
        '''
        Initialize empty attributes
        '''
        self.x = x;   self.y = y
        self.xc = xc; self.yc = yc
        self.land_check = land_check
        self.ROMS = ROMS

    def nearest4(self):
        '''
        Create nearest four indices and weights for all of the fields
        '''
        # Compute interpolation coefficients for nearest4 interpolation
        try:
            self.rho_index, self.rho_coef = self._compute_interpolation_coefficients(
                                                    self.x, self.y,
                                                    self.rho_inds,
                                                    self.rho_tree.data,
                                                    np.empty([len(self.x),  4]), np.empty([len(self.x),  4]),
                                                    widget_title = 'rho'
                                                    )
            if self.xc is not None:
                self.u_index, self.u_coef = self._compute_interpolation_coefficients(
                                                        self.xc, self.yc, 
                                                        self.u_inds,
                                                        self.u_tree.data, 
                                                        np.empty([len(self.xc),  4]), np.empty([len(self.xc),  4]),
                                                        widget_title = 'u'
                                                        ) # note: u_tree.data = u_points used to build the u_tree

                self.v_index, self.v_coef = self._compute_interpolation_coefficients(
                                                        self.xc, self.yc, 
                                                        self.v_inds,
                                                        self.v_tree.data, 
                                                        np.empty([len(self.xc),  4]), np.empty([len(self.xc),  4]),
                                                        widget_title = 'v'
                                                        )
        except ValueError:
            self.domain_exception_plot(self.rho_tree.data)
            raise DomainError('Your FVCOM domain is outside of the ROMS domain, it needs to be changed.')

        # Check if your mesh covers ROMS land, if so kill the routine and force the user to change the grid
        self.check_if_ROMS_land_in_FVCOM_mesh()

    # Some basic grid info needed to compute the nearest-4 matrices and prepare for vertical interpolation
    # ----
    @property
    def fvcom_rho_dpt(self):
        return np.sum(self.ROMS.h_rho[self.ROMS.fv_rho_mask][self.rho_index] * self.rho_coef, axis=1)

    @property
    def fvcom_u_dpt(self):
        return np.sum(self.ROMS.h_u[self.ROMS.fv_u_mask][self.u_index] *self.u_coef, axis=1)

    @property
    def fvcom_v_dpt(self):
        return np.sum(self.ROMS.h_v[self.ROMS.fv_v_mask][self.v_index] *self.v_coef, axis=1)

    @property
    def fvcom_rho_dpt(self):
        return np.sum(self.ROMS.h_rho[self.ROMS.fv_rho_mask][self.rho_index] * self.rho_coef, axis=1)

    @property
    def fvcom_angle(self):
        return np.sum(self.ROMS.angle[self.ROMS.fv_rho_mask][self.rho_index] * self.rho_coef, axis=1)

    # KDTrees used to find the 4 nearest ROMS points
    # -----
    @property
    def psi_tree(self):
        return KDTree(np.array([self.ROMS.cropped_x_psi, self.ROMS.cropped_y_psi]).transpose())

    @property
    def rho_tree(self):
        return KDTree(np.array([self.ROMS.cropped_x_rho, self.ROMS.cropped_y_rho]).transpose())
    
    @property
    def u_tree(self):
        return KDTree(np.array([self.ROMS.cropped_x_u, self.ROMS.cropped_y_u]).transpose())

    @property
    def v_tree(self):
        return KDTree(np.array([self.ROMS.cropped_x_v, self.ROMS.cropped_y_v]).transpose())
    
    # The nearest ROMS point to any FVCOM point
    # ----
    @property
    def fv2psi(self):
        _, fv2psi      = self.psi_tree.query(self.fv_nodes) # psi is at the centre of rho cells
        return fv2psi

    @property
    def fv2u_centre(self):
        _, fv2u_centre = self.v_tree.query(self.fv_cells)   # v is a the centre of u cells
        return fv2u_centre

    @property
    def fv2v_centre(self):
        _, fv2v_centre = self.u_tree.query(self.fv_cells)   # u is at the centre of v cells
        return fv2v_centre

    # Point representation of FVCOM nodes and cells
    # ----
    @property
    def fv_nodes(self):
        return np.array([self.x, self.y]).transpose()

    @property
    def fv_cells(self):
        return np.array([self.xc, self.yc]).transpose()

    @property
    def ball_radius(self):
        '''
        We make use of the staggering of the ROMS grid to find clusters of 4 indices.
        This propery gives a distance from a centre-point in a ROMS grid-square, used to find those points making up the grid-square
        '''
        dst = np.sqrt((self.ROMS.cropped_x_rho - self.ROMS.cropped_x_psi[0])**2 + (self.ROMS.cropped_y_rho - self.ROMS.cropped_y_psi[0])**2)
        return 1.3*dst[dst.argsort()[0]] # 1.3 was arbitrary, but turns ot to make sure that we only find the points we need

    @property
    def rho_inds(self):
        return self.rho_tree.query_ball_point(self.psi_tree.data[self.fv2psi], r = self.ball_radius)
    
    @property
    def u_inds(self):
        return self.u_tree.query_ball_point(self.v_tree.data[self.fv2u_centre], r = self.ball_radius)
    
    @property
    def v_inds(self):
        return self.v_tree.query_ball_point(self.u_tree.data[self.fv2v_centre], r = self.ball_radius)

    def check_if_ROMS_land_in_FVCOM_mesh(self):
        '''
        Check if FVCOM covers ROMS land, if so return
        '''
        if not self.land_check: # return if told to not care about land
            return

        error = False
        for field in ['rho','u','v']:
            indices = getattr(self.ROMS, f'Land_{field}')[getattr(self, f'{field}_index')]
            if indices.any():
                if not error:
                    plt.figure()
                    plt.scatter(self.x, self.y, 'FVCOM')
                    error = True
                x_roms = getattr(self.ROMS, f'cropped_x_{field}')[indices]
                y_roms = getattr(self.ROMS, f'cropped_y_{field}')[indices]
                plt.scatter(x_roms, y_roms, label = f'ROMS {field} land points intersecting with FVCOM mesh')

        if error:
            plt.axis('equal')
            plt.legend()
            raise LandError('ROMS intersects with your FVCOM experiment in the nestingzone, see the figure and adjust the mesh.')

    def domain_exception_plot(self, ROMS_points):
        plt.plot(ROMS_points[:, 0], ROMS_points[:, 1], 'r.', label = 'ROMS')
        plt.plot(self.x, self.y, 'b.', label = 'FVCOM')
        plt.legend()
        plt.axis('equal')
        plt.show(block=False)

# ============================================================================================================================
#                                 Class downloading ROMS data and dumping to FVCOM
# ============================================================================================================================
class LinearInterpolation:
    '''
    Linearly interpolate data from ROMS to FVCOM grid points
    '''
    def horizontal_interpolation(self):
        '''
        bi-linear interpolation from one ROMS point to another
        '''
        zlen      = self.salt.shape[-1]
        self.u    = np.sum(self.u[self.N4.u_index, :]*np.repeat(self.N4.u_coef[:, :, np.newaxis], zlen, axis=2), axis=1)
        self.v    = np.sum(self.v[self.N4.v_index, :]*np.repeat(self.N4.v_coef[:, :, np.newaxis], zlen, axis=2), axis=1)
        self.zeta = np.sum(self.zeta[self.N4.rho_index]*self.N4.rho_coef, axis=1)
        self.temp = np.sum(self.temp[self.N4.rho_index, :]*np.repeat(self.N4.rho_coef[: ,:, np.newaxis], zlen, axis=2), axis=1)
        self.salt = np.sum(self.salt[self.N4.rho_index, :]*np.repeat(self.N4.rho_coef[:, :, np.newaxis], zlen, axis=2), axis=1)

    def vertical_interpolation(self):
        '''
        Linear vertical interpolation.
        '''
        salt = np.flip(self.salt, axis=1).T
        self.salt = salt[self.N4.vi_ind1_rho, range(0, salt.shape[1])] * self.N4.vi_weigths1_rho\
                    + salt[self.N4.vi_ind2_rho, range(0, salt.shape[1])] * self.N4.vi_weigths2_rho

        temp = np.flip(self.temp, axis=1).T
        self.temp = temp[self.N4.vi_ind1_rho, range(0, temp.shape[1])] * self.N4.vi_weigths1_rho \
                    + temp[self.N4.vi_ind2_rho, range(0, temp.shape[1])] * self.N4.vi_weigths2_rho

        u = np.flip(self.u, axis=1).T
        self.u = u[self.N4.vi_ind1_u, range(0, u.shape[1])] * self.N4.vi_weigths1_u + \
                 + u[self.N4.vi_ind2_u, range(0, u.shape[1])] * self.N4.vi_weigths2_u

        v = np.flip(self.v, axis=1).T
        self.v = v[self.N4.vi_ind1_v, range(0, v.shape[1])] * self.N4.vi_weigths1_v + \
                 + v[self.N4.vi_ind2_v, range(0, v.shape[1])] * self.N4.vi_weigths2_v

class Roms2FVCOMNest(RomsDownloader, LinearInterpolation):
    '''
    Class writing data to nesting-file.
    - Downloading ROMS data
    - Interpolating to FVCOM
    '''
    def __init__(self, outfile, time, path, index, N4):
        '''
        outfile: Name of file to write to
        time:    list of timesteps to write to
        path:    list of path to file to read timestep from
        index:   list of indices to read from file for each timestep
        N4:      Class with Nearest4 interpolation coefficients
        '''
        self.out = netCDF4.Dataset(outfile, 'r+')
        self.time = time; self.path = path; self.index = index
        self.N4 = N4
        self.ROMS = N4.ROMS
        self.verbose = True

    @property
    def angle(self):
        return np.mean(self.N4.fvcom_angle[self.N4.NEST.nv], axis=1)

    @property
    def NEST(self):
        return self.N4.NEST

    def dump(self):
        '''
        loop over all timesteps and interpolate them to the FVCOM nesting file
        '''
        for self.counter, (self.fvcom_time, self.path_here, self.index_here) in enumerate(zip(self.time, self.path, self.index)):
            self.read_timestep()
            print(f"- {netCDF4.num2date(self.fvcom_time, units='days since 1858-11-17 00:00:00')}")
            self.crop_and_transpose_roms_data()
            self.horizontal_interpolation()
            self.vertical_interpolation()
            self._dump_timestep_to_nest()
        self.out.close()
        self.nc.close()

    def crop_and_transpose_roms_data(self):
        self.salt = self.salt[:, self.ROMS.cropped_rho_mask].transpose()
        self.temp = self.temp[:, self.ROMS.cropped_rho_mask].transpose()
        self.zeta = self.zeta[self.ROMS.cropped_rho_mask].transpose()
        self.u    = self.u[:, self.ROMS.cropped_u_mask].transpose()
        self.v    = self.v[:, self.ROMS.cropped_v_mask].transpose()

    def _dump_timestep_to_nest(self):
        '''
        Write the data to the output file
        '''
        self.out.variables['time'][self.counter]           = self.fvcom_time
        self.out.variables['Itime'][self.counter]          = np.floor(self.fvcom_time)
        self.out.variables['Itime2'][self.counter]         = np.round((self.fvcom_time - np.floor(self.fvcom_time)) * 60 * 60 * 1000, decimals = 0)*24
        self.out.variables['u'][self.counter, :, :]        = self.u*np.cos(self.angle) - self.v*np.sin(self.angle)
        self.out.variables['v'][self.counter, :, :]        = self.u*np.sin(self.angle) + self.v*np.cos(self.angle)
        self.out.variables['hyw'][self.counter, :, :]      = np.zeros((1, len(self.NEST.siglev[:,0]), len(self.NEST.siglev[0,:])))
        self.out.variables['zeta'][self.counter, :]        = self.zeta
        self.out.variables['temp'][self.counter, :, :]     = self.temp
        self.out.variables['salinity'][self.counter, :, :] = self.salt
        self.out.variables['weight_node'][self.counter,:]  = self.NEST.weight_node
        self.out.variables['weight_cell'][self.counter,:]  = self.NEST.weight_cell

# ==========================================================================================================================
#   Force the FVCOM bathymetry in the nestingzone to be equal to that of ROMS, add smooth transition to FVCOM bathymetry
# ==========================================================================================================================
class MatchTopo:
    def __init__(self, ROMS, M, NEST, store_bath):
        '''
        Rutine matching ROMS depth with FVCOM depth, same-same in nestingzone,
        but a smooth transition from ROMS to BuildCase topo on the outside of it
        '''
        self.M = M 
        self.ROMS = ROMS
        self.NEST = NEST
        self.store_bath = store_bath
        self.R_edge_of_nestingzone = self.NEST.R+50
        self.R_edge_of_smoothing_zone = 5*self.NEST.R # arbitrarily chosen, this can be tuned if the nestingzone is acting weird
        self.get_distance_from_gridpoints_to_obc()
        self.get_nodes_near_obc()

    def get_distance_from_gridpoints_to_obc(self):
        '''
        compute fvcom cells distance from the obc
        '''
        obc_tree      = KDTree(np.array([self.M.x[self.M.obc_nodes], self.M.y[self.M.obc_nodes]]).transpose())
        self.Rdst_cells, _ = obc_tree.query(np.array([self.M.xc, self.M.yc]).transpose())
        self.Rdst_nodes, _ = obc_tree.query(np.array([self.M.x, self.M.y]).transpose())

    @cached_property
    def weight(self):
        _weight = np.zeros(self.M.x.shape)
        _weight[self.nestingzone_nodes] = 1
        R_outer_edge = np.max(self.Rdst_nodes[self.transition_nodes])

        # compute weights in the transition zone
        width_transition = -(R_outer_edge - self.R_edge_of_nestingzone)
        a = 1.0/width_transition
        b = R_outer_edge/width_transition
        _weight[self.transition_nodes]  = a*self.Rdst_nodes[self.transition_nodes]-b
        return _weight

    def get_nodes_near_obc(self):
        '''
        return the nodes in the nestingzone (where we will set the ROMS bathymetry) and in the transition zone
        where we will use a smooth transition from ROMS bathymetry to FVCOM bathymetry
        '''
        # First find cells
        self.nestingzone_cells     = np.where(self.Rdst_cells<self.R_edge_of_nestingzone)[0]
        self.cells_outside_of_nest = np.where(self.Rdst_cells>=self.R_edge_of_nestingzone)[0]
        self.transition_cells      = np.where(np.logical_and(self.Rdst_cells>self.R_edge_of_nestingzone, self.Rdst_cells<=self.R_edge_of_smoothing_zone))[0]
        self.cells_to_change       = np.where(self.Rdst_cells<=self.R_edge_of_smoothing_zone)[0]

        # Then get the associated nodes
        self.nestingzone_nodes     = np.unique(self.M.tri[self.nestingzone_cells])
        self.nodes_outside_of_nest = np.unique(self.M.tri[self.cells_outside_of_nest])
        self.transition_nodes      = np.unique(self.M.tri[self.transition_cells])
        self.nodes_to_change       = np.unique(self.M.tri[self.cells_to_change])

    def add_ROMS_bathymetry_to_FVCOM_and_NEST(self):
        '''
        Her må man holde tunga i munnen, viktig å huske at akkurat i denne rutinen sysler vi med _hele_ FVCOM domenet!
        '''
        # Prepare interpolator, add ROMS bathymetry to the transition-zone in h_roms
        N4B = N4ROMS(self.ROMS, x = self.M.x[self.nodes_to_change], y = self.M.y[self.nodes_to_change], land_check = False) # the depth at ROMS land is equal to min_depth
        N4B.nearest4()

        # Make copy of FVCOM bathymetry
        h_roms = np.copy(self.M.h) 
        h_roms[self.nodes_to_change] = np.sum(N4B.ROMS.h_rho[N4B.ROMS.fv_rho_mask][N4B.rho_index]*N4B.rho_coef, axis=1)

        # Update the nodes according to their distance to the obc
        self.M.h  = h_roms*self.weight + self.M.h*(1-self.weight)

        # Get M indices corresponding to the nest locations
        fvtree = KDTree(np.array([self.M.x, self.M.y]).transpose())
        _, ind = fvtree.query(np.array([self.NEST.xn[:,0], self.NEST.yn[:,0]]).transpose())

        # Store the smoothed bathymetry
        self.NEST.h       = self.M.h[ind,None]
        self.NEST.hc      = np.mean(self.NEST.h[self.NEST.nv], axis = 1)[:,None]
        self.NEST.siglayz = self.NEST.h*self.NEST.siglay
        self.store_bathymetry()
        return self.M, self.NEST

    def store_bathymetry(self):
        if self.store_bath:
            filename = f"./input/{self.M.info['casename']}_dep.dat"
            write_FVCOM_bath(self.M, filename = filename)

# ==============================================================================================
#                            Write an empty output file
# ==============================================================================================
def create_nc_forcing_file(name, NEST):
    '''
    Creates empty nc file formatted to fit FVCOM open boundary ocean forcing
    '''
    nc = netCDF4.Dataset(name, 'w', format='NETCDF3_CLASSIC')

    # Write global attributes
    # ----
    nc.title       = 'FVCOM Nesting File'
    nc.institution = 'Akvaplan-niva AS'
    nc.source      = 'FVCOM grid (unstructured) nesting file'
    nc.created     = f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} using roms_nesting_fg.py'

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
    nc.variables['lat'][:]           = NEST.latn
    nc.variables['lon'][:]           = NEST.lonn
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

    nc.close()

class LandError(Exception): pass
class DomainError(Exception): pass
class InputError(Exception): pass