'''
FVCOM streamlines

Goal:
----
Create a function that returns matplotlib-streamlines-"like" streamlines from FVCOM data

Theory:
----
This routine essentially works in a similar way to a drift model. "particles" are advected a distance dx over
a "time interval" dt. This integration done sufficiently many times gives streamlines showing how the flow
in the domain connect spatially.

Status:
----
Basic functions work, and the groundwork for using more fancy linestyles (colored with speed etc) has been layed.
QC looks good, as is sort of expected for a simple integration such as this?
Forward stepping is done in the most rudamentary way possible at the moment, can improve accuracy by either
decreasing the step (since integration is very cheap with the new search algorithm), or implementing a 
runge kutta scheme for timetepping (which I suspect will be too costly.)

Still some work to be done on the masking of streamlines (too many short streamlines at the moment, since all
seeded streamlines mask the triangle they start in at the moment.)
'''
version_number = 0.5
import sys
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.patches as patches
from numba import jit
from fvtools.grid.fvcom_grd import FVCOM_grid
from scipy.spatial import cKDTree as KDTree
from fvtools.grid import tge

class streamlines:
    '''
    Computes streamlines from a velocity dataset
    '''
    def __init__(self, grid, max_length = 1000, 
                 method='linear', 
                 verbose=False, 
                 zorder=None, 
                 color=None,
                 linewidth=None):
        '''
        Streamlines need a gridfile to compute grid metrics, and velocity data to
        compute other parameters

        grid:        grid (either FVCOM_grid object, or a path to a grid file)
        max_length:  maximum number of integration steps for a trajectory
        method:      'linear' for linear interpolation of velocity data to trajectory,
                     'nearest' for nearest neighbor interpolation of velocity to trajectory
                     (only linear interpolation is available at the moment)
        verbose:     report script progress if activated
        zorder:      set the plot order (default on top of existing)
        '''
        # Interpret grid input
        # ----
        self.verbose = verbose
        if self.verbose: print('\nInitializing FVCOM streamline maker\n-----------------------')
        if self.verbose: print('  - load grid')
        if isinstance(grid, str):
            self.M = FVCOM_grid(grid)

        elif isinstance(grid, object):
            self.M = grid

        else:
            raise InputError('Grid must either be a string or an object')

        # Store parameters
        # ----
        self.max_length = max_length
        self.method     = method

        # Get grid metrics needed by the interpolation algorithm
        # ----
        if self.method == 'linear':
            self._prepare_linear()

        else:
            self._prepare_nearest()

        # Initialize line-styles
        # ---
        self.linewidth = linewidth
        self.color = color
        self.zorder = zorder

    def get_streamlines(self, u, v, xlim = None, ylim = None, res = None, new_initial = False, axes = None):
        '''
        Runs the streamline show.

        The first time you call 
        '''
        # Get an axes that we can plot the lines to
        # ----
        if axes is None:
            self.axes = matplotlib.pyplot.gca()
        else:
            self.axes = axes

        # Load plot settings
        # ----
        self.plot_settings()

        # Initial positions should just need to be calculated once
        # ----
        try:
            self.initial_positions
        except:
            self.initialize_positions(xlim, ylim, res)

        # Set new initial positions
        if new_initial:
            self.initialize_positions(xlim, ylim, res)

        # Define cells that should _not_ be visited
        free = np.ones((len(self.M.ctri)), dtype = np.int32)

        if self.verbose: print('- compute streamlines using a linear interpolation algorithm')
        # Forward
        streamline_xf, streamline_yf, spf, free = compute_streamline_linear(self.M.xc, self.M.yc, self.M.ctri,
                                                                            self.NBSE, self.NESE, self.gridres, free,
                                                                            self.initial_positions, self.max_length, 
                                                                            u, v, self.initial_cell)

        # Otherwise we can't really start a new streamline from the same point :)
        free[self.initial_cell] = 1

        # Backward
        streamline_xb, streamline_yb, spb, free = compute_streamline_linear(self.M.xc, self.M.yc, self.M.ctri,
                                                                            self.NBSE, self.NESE, self.gridres, free,
                                                                            self.initial_positions, self.max_length, 
                                                                            -u, -v, self.initial_cell)

        # Concatenate the matrices
        # ----
        self.streamline_x = np.concatenate((np.fliplr(streamline_xb), streamline_xf), axis = 0)
        self.streamline_y = np.concatenate((np.fliplr(streamline_yb), streamline_yf), axis = 0)
        self.speed        = np.concatenate((np.fliplr(spb), spf), axis = 0)

        # Create matplotlib line 
        if self.verbose: print('- visualize streamlines')
        self._linecollection()

    def _prepare_linear(self):
        '''
        Prepare a linear interpolation algorithim (this is actually quite quick)
        '''
        # Establish a KD tree to find the nearest cell
        # ----
        self.cell_tree  = KDTree(np.array([np.mean(self.M.xc[self.M.ctri], axis = 1), 
                                           np.mean(self.M.yc[self.M.ctri], axis = 1)]).transpose())

        # Get approximate grid resolution
        # -----
        big_nv          = np.c_[self.M.ctri, self.M.ctri[:,0]]
        dx              = np.diff(self.M.xc[big_nv], axis = 1)
        dy              = np.diff(self.M.yc[big_nv], axis = 1)
        ds              = np.sqrt(dx**2+dy**2)
        self.gridres    = np.min(ds, axis = 1)

        NT = len(np.mean(self.M.xc[self.M.ctri], axis = 1)) # cells
        MT = len(self.M.xc)      # nodes

        # Find elements surrounding elements
        # ----
        if self.verbose: print('\n- Find elements surrounding elements')
        NV = tge.check_nv(self.M.ctri, self.M.xc, self.M.yc)

        if self.verbose: print('  - NBE')
        NBE                       = tge.get_NBE(NT, MT, NV)

        if self.verbose: print('  - Boundary check')
        ISBCE, ISONB              = tge.get_BOUNDARY(NT, MT, NBE, NV)

        if self.verbose: print('  - Elements around nodes')
        NBVE, NBVT, NTVE          = tge.get_NBVE_NBVT(MT, NT, NV, 15)

        if self.verbose: print('  - Nodes around nodes')
        NTSN, NBSN, NBVE, NBVT, _ = tge.get_NTSN_NBSN(NBVE, NTVE, NBVT, NBE, NV, ISONB, 15, MT)
        
        if self.verbose: print('  - Elements around elements')
        self.NBSE, self.NESE      = tge.get_NBSE_NESE(NBVE, NTVE, NV, NT)

    def _prepare_nearest(self):
        '''
        prepare grid metrics needed by the nearest neighbor interpolation algorithm
        '''
        big_nv = np.c_[self.M.tri, self.M.tri[:,0]]
        dx     = np.diff(self.M.x[big_nv], axis = 1)
        dy     = np.diff(self.M.y[big_nv], axis = 1)
        ds     = np.sqrt(dx**2+dy**2)
        raise ImplementationError('Not developed yet, linear interpolation is the way to go')

    def initialize_positions(self, xlim, ylim, res):
        '''
        Starting-points are initialized at evenly spaced positions
        '''
        # Some input management
        # ----
        if self.verbose: print('- initialize streamlines')
        if xlim is None:
            xlim = [np.min(self.M.x), np.max(self.M.x)]

        if ylim is None:
            ylim = [np.min(self.M.y), np.max(self.M.y)]

        if res is None:
            xres = (xlim[1]-xlim[0])/20
            yres = (ylim[1]-ylim[0])/20
            res = np.min([xres, yres])

        # make a (M,2) array of the initial positions
        # ----
        xpos = np.arange(np.min(xlim), np.max(xlim), res)
        ypos = np.arange(np.min(ylim), np.max(ylim), res)
        xgrd, ygrd = np.meshgrid(xpos, ypos)
        initial_positions = np.array((xgrd.ravel(), ygrd.ravel())).T

        # remove initial positions that are on land
        # ----
        if self.verbose: print('  - remove streamlines initialized on land')
        self.cell_tree
        _, cell_ind  = self.cell_tree.query(np.array([initial_positions[:,0], initial_positions[:,1]]).transpose())
        xcorr, ycorr, self.initial_cell = remove_land(self.M.xc, self.M.yc, self.M.ctri, self.NBSE, self.NESE, initial_positions, cell_ind)

        # Return to be used by the rest of the routine
        # ----
        self.initial_positions = np.array([xcorr, ycorr]).T

    def _linecollection(self):
        '''
        Lines that indicate direction, and behave relatively similar to matplotlib.streamplot
        --> More or less pure copy of code from matplotlib.streamplot
        '''        
        # create the streamlines
        if self.verbose: print('- prepare line and arrow collection')
        self._create_lines()
        
        # Prepare the arrow patches we want to plot
        arrows      = []
        for t in self.streamlines:
            # Add arrows half way along each trajectory
            s = np.cumsum(np.hypot(np.diff(t[:,0]), np.diff(t[:,1])))
            n = np.searchsorted(s, s[-1]/2)
            arrow_tail = (t[:,0][n], t[:,1][n])
            arrow_head = (np.mean(t[:,0][n:n+2]), np.mean(t[:,1][n:n+2]))

            p = patches.FancyArrowPatch(arrow_tail, arrow_head, **self.arrow_kw)
            self.axes.add_patch(p)
            arrows.append(p)

        # Add the line patches
        lc = mcollections.LineCollection(self.streamlines, **self.line_kw)
        self.axes.add_collection(lc)
        self.axes.autoscale_view()

        # Store arrows
        ac = matplotlib.collections.PatchCollection(arrows)

        # Return a streamplot set
        self.streamset = StreamplotSet(lc, ac)

    def _create_lines(self):
        self.streamlines = []
        self.streamline_color = []
        for t in range(len(self.streamline_x[:,0])): # loop over trajectories
            # Create a clean streamline (ie. without nans) for each trajectory
            px  = self.streamline_x[t,:][~np.isnan(self.streamline_x[t,:])]
            py  = self.streamline_y[t,:][~np.isnan(self.streamline_y[t,:])]
            clr = self.speed[t,:][~np.isnan(self.speed[t,:])]
            self.streamlines.extend([np.column_stack([px, py])])
            self.streamline_color.extend([np.array([clr])])

    def plot_settings(self, arrowstyle = '-|>', arrowsize = 1):
        '''
        Settings for the lineplots
        '''
        # Initialize settings dictionaries
        # ----
        self.line_kw = {}
        self.arrow_kw = dict(arrowstyle = arrowstyle, mutation_scale = 10 * arrowsize)

        # Set linewidth
        # ----
        if self.linewidth is None:
            self.line_kw['linewidth'] = matplotlib.rcParams['lines.linewidth']
            self.linewidth = matplotlib.rcParams['lines.linewidth']
        else:
            self.line_kw['linewidth'] = self.linewidth

        # Make sure that lines are plotted on top
        # ----
        if self.zorder is None:
            self.zorder = mlines.Line2D.zorder

        self.line_kw['zorder']  = self.zorder
        self.arrow_kw['zorder'] = self.zorder

        # Set color of lines and arrows
        # ----
        if self.color is None:
            self.color = self.axes._get_lines.get_next_color()
            self.line_kw['color']   = self.color
            self.arrow_kw['color']  = self.color
        else:
            self.line_kw['color']   = self.color
            self.arrow_kw['color']  = self.color

@jit(nopython=True)
def remove_land(grid_x, grid_y, nv, NBSE, NESE, initial_positions, initial_cell):
    '''
    Loop over grid, check which of the particles are positioned over land, and returns only the ocean-points
    '''
    # Cell identifiers
    # ----
    mid_x = (grid_x[nv[:,0]] + grid_x[nv[:,1]] + grid_x[nv[:,2]])/3
    mid_y = (grid_y[nv[:,0]] + grid_y[nv[:,1]] + grid_y[nv[:,2]])/3

    xpos  = initial_positions[:,0]
    ypos  = initial_positions[:,1]
    last_visit = np.copy(initial_cell)

    # Check if the point is on land
    # ---
    land = np.zeros((len(xpos)), dtype = np.int32) 

    # Loop over space
    # ----
    for i in range(len(xpos)):
        # Streamline position:
        # ----
        px = xpos[i]
        py = ypos[i]

        # Find the triangle we're within:
        # -  Continue and skip the search if you are within the closest triangle
        last_cell = last_visit[i]
        this_nbse = NBSE[last_cell,:NESE[last_cell]+1]

        if is_inside_tri(grid_x[nv[last_cell]], grid_y[nv[last_cell]], px, py):
            cind = last_visit[i]

        else:
            dst    = np.sqrt((mid_x[this_nbse]-px)**2+(mid_y[this_nbse]-py)**2)
            noluck = 0
            while True:
                ind  = np.where(dst == np.min(dst))[0][0]
                cind = this_nbse[ind]
                if is_inside_tri(grid_x[nv[cind]], grid_y[nv[cind]], px, py):
                    last_visit[i] = cind
                    break

                else:
                    noluck +=1
                    dst[ind] = 10**6
                    if noluck > 5:
                        # Then the point is most likely on land, so we prepare to remove it from the search
                        land[i] = 1
                        break

    # Remove points that are initialized on land
    # ----
    xcorr = xpos[land == 0]
    ycorr = ypos[land == 0]
    ic    = initial_cell[land == 0]

    return xcorr, ycorr, ic

@jit(nopython=True)
def is_inside_tri(xn, yn, xpos, ypos):
    '''
    Walk clockwise or counterclockwise around the triangle and project the point onto the segment we are crossing
    by using the dot product. Finally, check that the vector created is on the same side for each of the triangle's segments.

    https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
    '''
    # Points to be evaluated
    x = xpos; y = ypos

    # Triangles these are within
    ax = xn[0];  ay = yn[0]
    bx = xn[1];  by = yn[1]
    cx = xn[2];  cy = yn[2]

    # Segments in each triangle
    side_1 = (x - bx) * (ay - by) - (ax - bx) * (y - by)

    # Segment B to C
    side_2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy)

    # Segment C to A
    side_3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay)

    ret = (side_1<=0.0) == (side_2<=0.0) == (side_3<=0.0)
    return ret

@jit(nopython = True, fastmath = True)
def get_weights(xv, yv, px, py):
    '''
    Calculates linear interpolation weights for triangular data
    We will use barycentric coordinates

    --> (xv, yv) is the positions of nodes of the current triangle
    --> (px, py) is the position of the particle

    https://codeplea.com/triangular-interpolation
    '''
    # Initialize data
    weight    = np.zeros((xv.shape), dtype = np.float32)

    # Cells
    weight[0] = ((yv[1]-yv[2])*(px-xv[2]) + (xv[2]-xv[1])*(py-yv[2]))/((yv[1]-yv[2])*(xv[0]-xv[2]) + (xv[2]-xv[1])*(yv[0]-yv[2]))
    weight[1] = ((yv[2]-yv[0])*(px-xv[2]) + (xv[0]-xv[2])*(py-yv[2]))/((yv[1]-yv[2])*(xv[0]-xv[2]) + (xv[2]-xv[1])*(yv[0]-yv[2]))
    weight[2] = 1-weight[0]-weight[1] 

    return weight

@jit(nopython=True)
def compute_streamline_linear(grid_x, grid_y, nv, NBSE, NESE, gridres, free_input,
                              initial_positions, max_length, u, v, initial_cell):
    '''
    Find nearest cell, integrate velocity to the edge of *this* triangle using a linear interpolation algorithm

    Input:
    ---
    grid_x: x position of nodes in nv triangle
    grid_y: y position of nodes in nv triangle
    nv:     triangulation indicating which nodes are in the corners

    streamline_x,y: matrix keeping the position of each streamline
    u,v:            velocity at the centre of the cells
    '''
    # Get positions of centres
    # ----
    mid_x = (grid_x[nv[:,0]] + grid_x[nv[:,1]] + grid_x[nv[:,2]])/3
    mid_y = (grid_y[nv[:,0]] + grid_y[nv[:,1]] + grid_y[nv[:,2]])/3

    # Prepare output arrays
    # ----
    sp           = np.nan * np.ones((len(initial_positions), max_length), dtype = np.float32)
    streamline_x = np.nan * np.ones((len(initial_positions), max_length), dtype = np.float32)
    streamline_y = np.nan * np.ones((len(initial_positions), max_length), dtype = np.float32)

    # Initialize positions:
    # ----
    streamline_x[:,0] = initial_positions[:,0]
    streamline_y[:,0] = initial_positions[:,1]
    last_visit   = np.copy(initial_cell)

    # Initialize control arrays
    # ----
    indices    = np.arange(len(mid_x))
    free       = np.copy(free_input) #np.ones((len(nv[:,0])), dtype = np.int32)            # To say that this triangle has/hasn't been visited
    stopped    = np.zeros((len(streamline_x[:,0])), dtype = np.int32) # To indicate that we are/aren't still integrating this path


    # Dynamic min_length:
    # ----
    min_length = max_length/40
    too_short  = np.zeros((len(streamline_x[:,0])), dtype = np.int32)

    # Loop over time
    # ----
    for i in range(len(streamline_x[:,0])):
        free_local = np.copy(free)
        length     = 0
        for t in range(len(streamline_x[0,:])-1):
            # Streamline position:
            # ----
            px = streamline_x[i,t]
            py = streamline_y[i,t]

            # Streamline current grid_cell
            # ----
            last_cell = last_visit[i]
            this_nbse = NBSE[last_cell, :NESE[last_cell]+1]

            # Continue to next streamline if this has already been stopped
            if stopped[i]:
                break

            length += 1
            # Find the triangle we're within:
            # -  Continue and skip the search if you are still within the last triangle
            if is_inside_tri(grid_x[nv[last_cell]], grid_y[nv[last_cell]], px, py):
                cind = last_visit[i]

            else:
                dst    = np.sqrt((mid_x[this_nbse]-px)**2+(mid_y[this_nbse]-py)**2)
                noluck = 0
                while True:
                    ind  = np.where(dst == np.min(dst))[0][0]
                    cind = this_nbse[ind]
                    if is_inside_tri(grid_x[nv[cind]], grid_y[nv[cind]], px, py):
                        last_visit[i] = cind
                        break

                    else:
                        noluck +=1
                        dst[ind] = 10**6
                        if noluck > 5:
                            # Then the point is most likely on land, terminate trajectory
                            stopped[i] = 1
                            break

                if not free_local[cind]:
                    stopped[i] = 1
                    continue

                else:
                    # And indicate that this cell has now been used, and should not be touched by
                    # other streamlines
                    # turns out this is a rather weak criteria :(
                    free_local[cind] = 0

            # Interpolate velocity data to this point
            # ----
            if stopped[i]:
                continue

            weight  = get_weights(grid_x[nv[cind]], grid_y[nv[cind]], px, py)
            u_point = np.sum(u[nv[cind]]*weight)
            v_point = np.sum(v[nv[cind]]*weight)

            # Choose timestep in this cell
            # - first see what the grid resolution is here
            dx     = gridres[cind]/6
            speed  = np.sqrt(u_point**2+v_point**2)

            # Now, for the rare case when the speed is exactly equal to zero
            if speed == 0:
                streamline_x[i, t+1] = streamline_x[i, t]
                streamline_y[i, t+1] = streamline_y[i, t]
                sp[i,t] = 0
                continue

            dt     = dx/speed
            streamline_x[i, t+1] = streamline_x[i, t] + u_point*dt
            streamline_y[i, t+1] = streamline_y[i, t] + v_point*dt

            # Kind of akward?
            sp[i,t] = speed

        # see if we want to restrict the other streamlines
        if length > min_length:
            free[free_local==0] = 0

        else:
            too_short[i] = 1

    return streamline_x[too_short == 0,:], streamline_y[too_short == 0,:], sp[too_short == 0,:], free

# A class that holds the lines and arrows (not sure how these can be used though)
class StreamplotSet:
    def __init__(self, lines, arrows):
        self.lines = lines
        self.arrows = arrows

# A class that copies the features of FVCOM_grid that are needed in tge.main
class UV_mesh:
    def __init__(self, M, xc, yc, nv):
        # Filepath and casename
        self.filepath = M.filepath
        self.casename = M.casename

        # Grid positions
        # ----
        self.x   = M.xc
        self.y   = M.yc
        if len(xc.shape)<2:
            self.xc = xc[:,None]
        else:
            self.xc = xc

        if len(yc.shape)<2:
            self.yc = yc[:,None]
        else:
            self.yc = yc

        # triangulation
        self.tri = nv

        # obc nodes is not necessary since the routine will just leace it empty



# Custom error codes
# ----
class InputError(Exception):
    pass

class ImplementationError(Exception):
    pass
