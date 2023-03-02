'''
Resolution estimator
- Class that estimates the needed number of nodes for a given resolution
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fvtools.gridding.coast as coast

from matplotlib.path import Path
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from scipy.spatial import cKDTree as KDTree
from functools import cached_property
from numba import njit, prange

class DistanceFunction:
    '''
    These are inputs for the hyperbolicus input-type
    '''
    def __init__(self, rfact = 35.0/2, dfact = 35.0/2, Ld = 50.0e3/2, dev1 = 25.0/2, dev2 = 25.0/2):
        self.rfact = rfact
        self.dfact = dfact 
        self.Ld    = Ld 
        self.dev1  = dev1
        self.dev2  = dev2

    @property
    def rmax(self):
        return self.res
    
    @property
    def _r2(self):
        return self.res / self.rfact

    @property
    def _x2(self):
        x2 = self.dfact * self._r2
        if x2 > self.Ld:
            raise Exception('Increase Ld or rfact - or decrease dfact. With current settings the function decreases with distance from the coast.')
        return x2

    @property
    def _a1(self):
        return self._x2 / self.dev1

    @property
    def _a2(self):
        return (self.Ld-self._x2)/self.dev2

    @property
    def _xm(self):
        return 2*self._a1

    @property
    def _xm2(self):
        return self._xm + 2*self._a2
    
    def distance_function_estimate(self, graphical):
        '''
        distance from coast function estimate of necessary number of nodes
        '''
        if not graphical:
            print('Estimating necessary number of nodes...')
        resolution, nodes_needed = self._get_node_estimate_function(self._r2, self._xm, self._xm2, self._a1, self._a2, self.rmax, 
                                                                    self.points, self.coast_points, self.coast_res, self.res**2)
        if not graphical:
            print(f'Given: \n'\
                 +f'       rfact = {self.rfact},\n'\
                 +f'       dfact = {self.dfact},\n'\
                 +f'       dfact = {self.Ld},\n'\
                 +f'       dev1  = {self.dev1},\n'\
                 +f'       dev2  = {self.dev2}, \n'\
                 +f'we need approximately {int(nodes_needed/1.8)} nodes.')
        return resolution, int(nodes_needed/1.8)

    @staticmethod
    def gfunc(rmax, dfact, rfact, dev1, dev2, Ld, rcoast):
        '''
        Solve the distance function for many distances at once
        - should share code with get_node_estimate_function (same function), but not a priority rigth now.
        '''
        x     = np.arange(0,Ld,10)
        r2    = rmax / rfact
        x2    = dfact * r2
        a1    = x2  / dev1
        a2    = (Ld - x2) / dev2
        xm    = 2*a1
        xm2   = xm +2 * a2
        gf    = r2 * (2 - (1 - np.tanh((x - xm) / a1))) / 2 + (rmax - r2) * (2 - (1 - np.tanh((x - xm2) / a2))) / 2
        gf    = gf - gf[0]
        res   = gf+rcoast 
        return res, x

    @staticmethod
    @njit(fastmath = True, parallel=True)
    def _get_node_estimate_function(r2, xm, xm2, a1, a2, rmax, points, coast_points, coast_res, area):
        '''
        The distance function computes the grid resolution given certain settings
        '''
        nodes_needed = 0
        theres = np.zeros((len(points,)))
        max_len = len(points)
        for i in prange(len(points)):
            x_p    = points[i, 0]
            y_p    = points[i, 1]
            distance_from_coast = np.sqrt((coast_points[:,0]-x_p)**2 + (coast_points[:,1]-y_p)**2)
            gfunc  = r2 * (2 - (1 - np.tanh((distance_from_coast - xm) / a1))) / 2 + (rmax - r2) * (2 - (1 - np.tanh((distance_from_coast - xm2) / a2))) / 2
            gfunc0 = r2 * (2 - (1 - np.tanh((0.0 - xm) / a1))) / 2 + (rmax - r2) * (2 - (1 - np.tanh((0.0 - xm2) / a2))) / 2
            gfunc  = gfunc - gfunc0
            res    = np.min(gfunc+coast_res)
            Atri   = (np.sqrt(3.0)/4.0)*res**2
            nodes_needed += area/Atri 
            theres[i] = res
        return theres, nodes_needed

class StructuredGridPoints:
    '''
    Class that sets up a structured grid to be used when estimating necessary number of nodes

    This structured grid is uniform over the entire domain, but there should not be anything in the way for making
    separate structgrids within the domain
    '''
    @property
    def res(self):
        '''
        resolution of structured grid used to compute the estimate
        '''
        if not hasattr(self, '_res'):
            self._res = self.resolution_requirements.max_res.max()
            self._make_structgrid()
        return self._res
    
    @res.setter
    def res(self, var):
        self._res = var
        self._make_structgrid()

    @property
    def res_vector(self):
        return np.ones((len(self.res,)))*self.res

    def _make_structgrid(self):
        '''
        Makes a "structured grid" that we use to estimate necessary number of nodes
        '''
        print(f'Creating a structured grid with resolution {self._res} m')

        # Create a grid-matrix
        # ----
        square_over_domain = np.array(np.meshgrid(np.arange(self.coast_x.min(), self.coast_x.max() + self._res, self._res),
                                                  np.arange(self.coast_y.min(), self.coast_y.max() + self._res, self._res)))

        # Make a point-array of the gridpoints
        # ----
        self.points = np.array([square_over_domain[0,:].ravel(), square_over_domain[1,:].ravel()]).T
        print('  - find points within outer boundary')
        self.return_points_within_outer_boundary()

        if any(self.inner_boundary.x):
            print('  - find points outside inner boundaries')
            self.return_points_outside_inner_boundary()

class MeshLand:
    '''
    Procedures to identify points on/off land

    nomenclature:
    ---
    outer_boundary: outer boundary (defined by obc- and solid land)
    inner_islands:  islands
    '''
    @cached_property
    def outer_boundary(self):
        xb, yb, db, mob, obc = coast.read_boundary(boundary_file = self.boundary_file)
        _boundary = pd.DataFrame({'x': xb, 'y': yb, 'coast_res': db})
        return _boundary

    @cached_property
    def inner_boundary(self):
        try:
            pi, xi, yi, di = coast.read_islands(islands_file = self.islands_file)
        except:
            xi = np.empty(0)
            yi = np.empty(0)
        _islands = pd.DataFrame({'x': xi, 'y': yi, 'polygon_number': pi, 'coast_res': di})
        return _islands

    @property
    def coast_x(self):
        return np.append(self.outer_boundary.x, self.inner_boundary.x)
    
    @property
    def coast_y(self):
        return np.append(self.outer_boundary.y, self.inner_boundary.y)
    
    @property
    def coast_res(self):
        return np.append(self.outer_boundary.coast_res, self.inner_boundary.coast_res)

    @property
    def coast_points(self):
        return np.array([self.coast_x, self.coast_y]).T

    @property
    def coast_tree(self):
        return KDTree(self.coast_points)

    @cached_property
    def resolution_at_closest_coast(self):
        _, ind = self.coast_tree.query(self.points)
        return ind

    @cached_property
    def point_distance_from_coast(self):
        d, _ = self.coast_tree.query(self.points)
        return d

    @property
    def outer_boundary_path(self):
        return Path(np.array([self.outer_boundary.x, self.outer_boundary.y]).T)

    @cached_property
    def inner_boundary_path(self):
        _islands = []
        for i in np.unique(self.inner_boundary.polygon_number):
            _islands.append(
                            Path(np.array([self.inner_boundary.x[self.inner_boundary.polygon_number == i], 
                                           self.inner_boundary.y[self.inner_boundary.polygon_number == i]]).T
                                 )
                            )
        return _islands
    
    def return_points_within_outer_boundary(self):
        '''
        removes points that are not in the domain
        '''
        self.points = self.points[self.outer_boundary_path.contains_points(self.points)]

    def return_points_outside_inner_boundary(self):
        '''
        removes points that are in islands
        '''
        _bools = np.ones((len(self.points),), dtype = bool)
        for p in self.inner_boundary_path:
            _bools[p.contains_points(self.points)] = False
        self.points = self.points[_bools]

class GraphicalInterface:
    '''
    Methods to graphically interact with the resolution function
    '''
    def _init_figures(self):
        '''
        initialize figures needed
        '''
        plt.figure()
        plt.scatter

    def graphical_nodenumber(self):
        '''
        estimates for necessary nodenumber
        '''
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.40)
        _res_curve, _distance_from_land = self.gfunc(self.rmax, self.dfact, self.rfact, self.dev1, self.dev2, self.Ld, self.coast_res.min())

        # The custom_function for standard inputs
        # ----
        self.l, = plt.plot(_distance_from_land, _res_curve, lw=2)
        ax = plt.gca()
        ax.set_ylabel('grid resolution [m]')
        ax.set_xlabel('meters away from nearest coast')
        ax.margins(x=0)

        # Initializing the sliders
        # ----
        axcolor  = 'lightgoldenrodyellow'
        axrfact  = plt.axes([0.17, 0.25, 0.65, 0.03], facecolor = axcolor)
        axdfact  = plt.axes([0.17, 0.20, 0.65, 0.03], facecolor = axcolor)
        axLd     = plt.axes([0.17, 0.15, 0.65, 0.03], facecolor = axcolor)  
        axdev1   = plt.axes([0.17, 0.10, 0.65, 0.03], facecolor = axcolor)
        axdev2   = plt.axes([0.17, 0.05, 0.65, 0.03], facecolor = axcolor)

        self.srfact   = Slider(axrfact, 'rfact', 0.1, 2*self.rfact, valinit = self.rfact, valstep = self.rfact/100.0)
        self.sdfact   = Slider(axdfact, 'dfact', 0.1, 2*self.dfact, valinit = self.dfact, valstep = self.dfact/100.0)
        self.sdev1    = Slider(axdev1, 'dev1', 0.1, 2*self.dev1, valinit = self.dev1/2, valstep = self.dev1/100.0)
        self.sdev2    = Slider(axdev2, 'dev2', 0.1, 2*self.dev2, valinit = self.dev2/2, valstep = self.dev2/100.0)
        self.sLd      = Slider(axLd,   'Ld', 0.1, 2*self.Ld, valinit = self.Ld, valstep = self.Ld/100.0)

        axnodenr = plt.axes([0.02,0.9,0.2,0.03])
        giver    = Button(axnodenr,'Nodes needed:')
        
        ax.set_title(f'You need ca. {self.nodes_needed} nodes.')   

        self.srfact.on_changed(self._update_curve)
        self.sdfact.on_changed(self._update_curve)
        self.sdev1.on_changed(self._update_curve)
        self.sdev2.on_changed(self._update_curve)
        self.sLd.on_changed(self._update_curve)

        giver.on_clicked(self._get_nodenr_estimate)
        plt.show(block=True)

    def _update_curve(self, val):
        '''
        Updates input to the resolution function
        '''
        self.rfact   = self.srfact.val
        self.dfact   = self.sdfact.val
        self.dev1    = self.sdev1.val
        self.dev2    = self.sdev2.val
        self.Ld      = self.sLd.val
        
        # The function
        _res_curve, _distance_from_land = self.gfunc(self.rmax, self.dfact, self.rfact, self.dev1, self.dev2, self.Ld, self.coast_res.min())
        
        # updating the figure
        l.set_data(_distance_from_land, _res_curve)
        ax.set_xlim([0, self.Ld])
        ax.set_title('We do not have an estimate for this curve, press the button to get one')
        fig.canvas.draw_idle()

    def _get_nodenr_estimate(self, event):
        '''
        Updates the necessary number of nodes estimate
        '''
        nodenum,theres = self.get_numberofnodes()
        ax.set_title(f'You need ca. {self.nodes_needed} nodes.')
        self.plot_resolution()

class QualityControl:
    '''
    routines used to sanity check the code on the fly
    '''
    def show_points(self):
        '''
        Show the point cloud
        '''
        plt.figure()
        plt.plot(self.points[:,0], self.points[:,1], 'k.', label = 'points')
        plt.plot(self.coast_x, self.coast_y, 'r.', label = 'coastline')
        plt.axis('equal')
        plt.show(block=False)

    def plot_resolution(self):
        '''
        show how the resolution will be where
        '''
        plt.figure()
        plt.scatter(self.points[:,0], self.points[:,1], c = self.resolution)
        plt.colorbar(label='m')
        plt.axis('equal')
        plt.show(block=False)

class NodeEstimate(GraphicalInterface, StructuredGridPoints, MeshLand, DistanceFunction, QualityControl):
    '''
    estimates necessary number of nodes

    available kwargs:
        rfact, dfact, Ld, dev1, dev2
    '''
    def __init__(self, Polyparam_file, boundary_file, islands_file, **kwargs):
        '''
        loads the stuff that are agnostic for all possible distance functions/resolution fields
        '''
        super().__init__(**kwargs)
        self.Polyparam_file = Polyparam_file
        self.resolution_requirements = pd.read_csv(Polyparam_file, sep = ';')
        self.boundary_file = boundary_file
        self.islands_file = islands_file

    @cached_property
    def point_tree(self):
        return KDTree(self.points)

    @property
    def nodes_needed(self):
        '''
        The number of nodes needed to get the desired resolution is 
        '''
        return int(np.sum(self.resolution/(np.sqrt(3)self.res**2/4))/1.8)
    
    def get_nodenr(self, graphical = False):
        '''
        get estimate based on distance from land
        '''
        # Get resolution estimate from the distance function
        distance_function_resolution = self.distance_function_estimate()

        # For now, the resolution is the same as the result from the distance function
        self.resolution = distance_function_resolution