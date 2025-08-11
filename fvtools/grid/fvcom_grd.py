import os
import fvtools.grid.tge as tge
import matplotlib
import matplotlib.pyplot as plt
import cmocean as cmo
import numpy as np
import progressbar as pb

from pyproj import Proj
from netCDF4 import Dataset
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree as KDTree
from numba import njit
from functools import cached_property

import warnings
warnings.filterwarnings("ignore")

class GridLoader:
    '''
    Loads relevant grid data to be used as instance attributes
    '''
    def _direct_initialization(self, x = None, y = None, tri = None,
                               lon = None, lat = None,
                               h = None, siglay = None, siglev = None,
                               obc_nodes = [], cropped_nodes = np.array(0), cropped_cells = np.array(0)):
        '''
        Initialize a new FVCOM_grid object using known fields
        '''
        # Assertions to make sure that input is consistent, regardless of source
        assert tri.min() == 0, 'Triangulation is invalid'
        assert type(obc_nodes) == list, 'obc_nodes must be given as a list'
        assert type(cropped_nodes) == np.ndarray, 'cropped nodes must be a numpy array'
        assert type(cropped_cells) == np.ndarray, 'cropped cells must be a numpy array'
        if h is not None: assert h.min() > 0, 'Minimum depth is invalid'
        # Add more tests when bugs appear
        self.tri = tri 
        if x is None and y is None:
            self.x, self.y = np.zeros(tri.max()+1), np.zeros(tri.max()+1)
        else:
            self.x, self.y = np.squeeze(x), np.squeeze(y)
        if lon is None and lat is None:
            self.lon, self.lat = np.zeros(x.shape), np.zeros(y.shape)
        else:
            self.lon, self.lat = lon, lat
        if h is not None:   
            self.h = np.squeeze(h)
        else:
            self.h = h
        self.siglay, self.siglev = siglay, siglev

        # Identifiers
        self.obc_nodes, self.cropped_nodes, self.cropped_cells = obc_nodes, cropped_nodes, cropped_cells

        assert self.x.shape == self.y.shape, 'x, y shapes do not match'
        assert len(self.x.shape) == 1 and len(self.y.shape) == 1, 'shape of x and y needs to be (N,)'
        assert self.lon.shape == self.lat.shape, 'lon, lat shapes do not match'
        
    def _add_grid_parameters_2dm(self):
        '''
        Grid parameters from a .2dm file
        - assumes that .2dm files are always projected to carthesian coordinates
        '''
        import fvtools.grid.fvgrid as fvgrid
        tri, nodes, x, y, _, _, types = fvgrid.read_sms_mesh(self.filepath, nodestrings = True)
        l = []
        for nstr in types:
            l.extend(nstr)
        self._direct_initialization(x=x, y=y, tri=tri, obc_nodes = l)
        self.casename = self.filepath.split('.2dm')[0]

    def _add_grid_parameters_mat(self):
        '''
        Read grid attributes from matlab grid file and add them to the FVCOM_grid instance
        - Probably won't work that well anymore
        '''
        from scipy.io import loadmat
        grid_mfile = loadmat(self.filepath)
        self.casename = 'from_mat_file'
        self._direct_initialization(x = grid_mfile['Mobj']['x'][0][0][:,0], 
                                    y = grid_mfile['Mobj']['y'][0][0][:,0],
                                    tri = grid_mfile['Mobj']['tri'][0][0][:]-1, 
                                    h = grid_mfile['Mobj']['h'][0][0][:,0],
                                    siglay = grid_mfile['Mobj']['siglay'][0][0], 
                                    siglev = grid_mfile['Mobj']['siglev'][0][0], 
                                    obc_nodes = list(grid_mfile['Mobj']['obc_nodes'][0][0][0]-1),
                                    lon = grid_mfile['Mobj']['lon'][0][0][:,0],
                                    lat = grid_mfile['Mobj']['lat'][0][0][:,0])
        # Set zisf *after* initialization if it exists
        if 'zisf' in grid_mfile['Mobj'].dtype.names:
            self.zisf = grid_mfile['Mobj']['zisf'][0][0][:, 0]
  
    def _add_grid_parameters_nc(self):
        """
        Load ncdata from a fvcom-formated netCDF file
        """
        kwargs = {}
        # Add casename
        self.casename = self.filepath.split('/')[-1].split(self.filepath.split('/')[-1].split('_')[-1])[0][:-1]
        if 'restart' in self.filepath.split('/')[-1]:
            self.casename = self.filepath.split('/')[-1].split('_restart')[0]

        with Dataset(self.filepath) as d:
            obc_nodes = []
            if 'obc_nodes' in d.variables.keys():
                kwargs['obc_nodes'] = list(d['obc_nodes'][:]-1) # fortran to python indexing
            kwargs['x'], kwargs['y'], kwargs['lon'], kwargs['lat'], kwargs['tri'] = d['x'][:], d['y'][:], d['lon'][:], d['lat'][:], d['nv'][:].T-1
            kwargs['h'] = d['h'][:]
            if 'siglay' in d.dimensions.keys():
                kwargs['siglev'], kwargs['siglay'] = d['siglev'][:].T, d['siglay'][:].T
                               
            if 'zisf' in d.variables:
                zisf_data = d['zisf'][:]
                self.zisf = zisf_data[0, :] if zisf_data.ndim == 2 else zisf_data[:]
                
        self._direct_initialization(**kwargs)


    def _add_grid_parameters_npy(self):
        '''
        Load grid information stored in a M.npy file
        '''
        grid = np.load(self.filepath, allow_pickle=True).item()
        obc_nodes = []
        if 'obc_nodes' not in list(grid.keys()):
            obcs = [list(nodestring[0]) for nodestring in grid['read_obc_nodes'][0]]
            for obc in obcs:
                obc_nodes.extend(obc)
        else:
            obc_nodes = list(grid['obc_nodes'])

        self._direct_initialization(x = grid['x'], y = grid['y'], 
                                    lon = grid['lon'], lat = grid['lat'],
                                    tri = grid['nv'], h = grid['h'], 
                                    siglay = grid['siglay'], siglev = grid['siglev'], 
                                    obc_nodes = obc_nodes)
        # Conditionally assign zisf
        if 'zisf' in grid:
            self.zisf = grid['zisf']
        # Add nodestrings (for convenience, computing them can be quite slow on huge grids)
        if 'read_obc_nodes' in grid.keys():
            self.nodestrings = [list(nodestring[0]) for nodestring in grid['read_obc_nodes'][0]]
        elif 'nodestrings' in grid.keys():
            self.nodestrings = grid['nodestrings']

        if 'info' in grid.keys():
            self.info = grid['info']
            self.casename = self.info['casename']
            try:
                self.reference = self.info['reference']
            except:
                pass
        else:
            self.casename = 'FVCOM experiment'

    def _add_grid_parameters_txt(self):
        '''Grid parameters from smeshing file'''
        with open(self.filepath) as f:
            nodenum = int(f.readline())
            points  = np.loadtxt(f, delimiter = ' ', max_rows = nodenum)
            trinum  = int(f.readline())
            tri     = np.loadtxt(f, delimiter = ' ', max_rows = trinum, dtype=int)
        self._direct_initialization(x = points[:,0], y = points[:,1], tri = tri)

class InputCoordinates:
    @property
    def tri(self):
        '''Triangulation (n, 3) connecting the nodes making up triangle n'''
        return np.array(self._tri)

    @tri.setter
    def tri(self, var):
        self._tri = var

    @property
    def x(self):
        '''Node x position'''
        return np.array(self._x)

    @x.setter
    def x(self, var):
        self._x = var
    
    @property
    def y(self):
        '''Node y-position'''
        return np.array(self._y)

    @y.setter
    def y(self, var):
        self._y = var

    @property
    def lat(self):
        '''Node lat-position'''
        return np.array(self._lat)
    
    @lat.setter
    def lat(self, var):
        self._lat = var

    @property
    def lon(self):
        '''Node lon-position'''
        return np.array(self._lon)
    
    @lon.setter
    def lon(self, var):
        self._lon = var

    @property
    def zeta(self):
        '''Sea surface elevation (must be set by user for each timestep)'''
        if not hasattr(self, '_zeta'): self._zeta = np.zeros(self.x.shape)
        return np.array(self._zeta)
    
    @zeta.setter
    def zeta(self, var):
        self._zeta = var

    @property
    def h(self):
        '''Bathymetric depth relative to mean sea level'''
        return np.array(self._h)
    
    @h.setter
    def h(self, var):
        self._h = var

    @property
    def zisf(self):
        '''Ice draft depth relative to sea level'''
        if not hasattr(self, '_zisf'):
            self._zisf = np.zeros(self.x.shape)
        return np.array(self._zisf)

    @zisf.setter
    def zisf(self, var):
        self._zisf = var
 

    @property
    def siglev(self):
        '''Sigma levels - top and bottom interfaces of sigma layers'''
        return np.array(self._siglev)

    @siglev.setter
    def siglev(self, var):
        self._siglev = var

    @property
    def siglay(self):
        '''Sigma layer - centre position of sigma layer'''
        return np.array(self._siglay)
    
    @siglay.setter
    def siglay(self, var):
        self._siglay = var

class Coordinates:
    '''Coordinates we can derive from input-coordinates'''
    @property
    def latc(self):
        '''Latitude of triangle centres (cells/elements)'''
        if not hasattr(self, '_latc'): self._latc = np.mean(self.lat[self.tri], axis = 1)
        return self._latc

    @latc.setter 
    def latc(self, var):
        self._latc = var

    @property
    def lonc(self):
        '''Longitude of triangle centres (cells/elements)'''
        if not hasattr(self, '_lonc'): self._lonc = np.mean(self.lon[self.tri], axis = 1)
        return self._lonc

    @lonc.setter 
    def lonc(self, var):
        self._lonc = var

    @property
    def xc(self):
        '''x-position for cells/elements'''
        return np.mean(self.x[self.tri], axis = 1)

    @property
    def yc(self):
        '''y-position of cells/elements'''
        return np.mean(self.y[self.tri], axis = 1)

    @property
    def siglev_c(self):
        '''sigma level at cells'''
        return np.mean(self.siglev[self.tri,:], axis = 1)

    @property
    def siglay_c(self):
        '''sigma layer at cells'''
        return np.mean(self.siglay[self.tri,:], axis = 1)

    @property
    def delta_sigma(self):
        '''sigma layer thickness at nodal points'''
        return -np.diff(self.siglev, axis = 1)

    @property
    def delta_sigma_uv(self):
        '''thickness og sigma layer at uv points'''
        return np.mean(self.delta_sigma[self.tri], axis = 1)

    @property
    def hc(self):
        '''bathymetric depth at cell center'''
        return np.mean(self.h[self.tri], axis = 1)

    @property
    def h_uv(self):
        '''bathymetric depth at cell center, will return same value as hc'''
        return np.mean(self.h[self.tri], axis = 1)
    
    @property
    def zisfc(self):
        '''ice draft at cell center'''
        return np.mean(self.zisf[self.tri], axis = 1)
    
    @property
    def d(self):
        '''total water column depth at nodes'''
        if hasattr(self, 'zisf') and self.zisf is not None:
            return self.h + self.zeta - self.zisf
        else:
            return self.h + self.zeta


    # Properties dealing with the vertical coordinate
    @property
    def dc(self):
        '''total water column depth at cells'''
        return np.mean(self.d[self.tri], axis = 1)
    
    @property
    def siglayz(self):
        '''sigma layer surfaces depth below sea surface at nodes'''
        if hasattr(self, "zisf"):
            return self.d[:, None]*self.siglay - self.zisf[:, None]*self.siglay
        return self.d[:, None]*self.siglay

    @property
    def siglevz(self):
        '''sigma level surfaces depth below sea surface at nodes'''
        if hasattr(self, "zisf"):
            return self.d[:, None]*self.siglev - self.zisf[:, None]*self.siglev
        return self.d[:, None]*self.siglev

    @property
    def siglayz_uv(self):
        '''Siglay depths at cells'''
        return np.mean(self.siglayz[self.tri,:], axis = 1)

    @property
    def siglevz_uv(self):
        '''sigma level depth below mean sea surface at cells'''
        return np.mean(self.siglevz[self.tri,:], axis = 1)

    @property
    def tri_area(self):
        '''tri area is the area of the triangles'''
        return self.calculate_tri_area()

    @property
    def cell_volume(self):
        '''Volume (m^3) of a cell-based control volume'''
        return -np.diff(self.siglevz_uv) * self.tri_area[:, None]

    @property
    def node_volume(self):
        '''Volume (m^3) of a node-based control volume'''
        return -np.diff(self.siglevz) * self.art1[:, None]

    @property
    def grid_angles(self):
        '''Angles of each triangle corner'''
        return self._get_angles().T

    @property
    def grid_res(self):
        '''resolution (minimum length of triangle walls in each triangle)'''
        return self._get_res()

    def calculate_tri_area(self):
        '''Returns triangle area, A = 0.5*|AB x AC|'''
        AB    = np.array([self.x[self.tri[:,1]]-self.x[self.tri[:,0]],
                          self.y[self.tri[:,1]]-self.y[self.tri[:,0]], \
                          np.zeros(len(self.xc))]).transpose()
        AC    = np.array([self.x[self.tri[:,2]]-self.x[self.tri[:,0]],
                          self.y[self.tri[:,2]]-self.y[self.tri[:,0]], \
                          np.zeros(len(self.xc))]).transpose()
        ABxAC = np.cross(AB, AC)
        return 0.5*np.linalg.norm(ABxAC, axis=1)

    def re_project(self, projection):
        '''Change reference system of this mesh to that of projection'''
        if projection != self.reference:
            self.reference = projection
            self.Proj = Proj(projection)
            self.x, self.y = self.Proj(self.lon, self.lat)

    def _project_xy(self):
        '''Project from to or from xy (depending on what we have)'''
        if np.count_nonzero(self.lon) == 0:
            self.lon,   self.lat   = self.Proj(self.x,  self.y,  inverse=True)
            self.lonc,  self.latc  = self.Proj(self.xc, self.yc,  inverse=True)

        if np.count_nonzero(self._x) == 0:
            self.x,  self.y  = self.Proj(self.lon, self.lat)

    def _get_proj(self):
        '''Returns a Proj method for the grid coordinates provided'''
        if hasattr(self, 'info'):
            if 'reference' in self.info: 
                self.reference = self.info['reference']
        return Proj(self.reference)

    def _get_res(self):
        '''Returns the grid resolution for each cell'''
        xpos, ypos = self.x[self.tri], self.y[self.tri]
        dx   = np.array([xpos[:,i]-xpos[:,(i+1)%3] for i in range(3)])
        dy   = np.array([ypos[:,i]-ypos[:,(i+1)%3] for i in range(3)])
        return np.mean(np.sqrt(dx**2 + dy**2), axis = 0)

    def _get_angles(self):
        '''Return the angles (not radians) of corners in the grid cells'''
        # cos(theta) = a * b / |a||b|
        AB, BC, CA    = self._get_sidewall_segments()
        lAB, lBC, lCA = self._get_sidewall_lengths()

        # Get dot products
        ABAC  = -(AB[0,:]*CA[0,:]+AB[1,:]*CA[1,:])
        BABC  = -(AB[0,:]*BC[0,:]+AB[1,:]*BC[1,:])
        CABC  = -(CA[0,:]*BC[0,:]+CA[1,:]*BC[1,:])

        # Get angles of all corners
        return np.array([np.arccos(ABAC/(lAB*lCA)), np.arccos(BABC/(lAB*lBC)), np.arccos(CABC/(lCA*lBC))])*(360/(2*np.pi))

    def _get_sidewall_segments(self):
        AB    = np.array([self.x[self.tri][:,0]-self.x[self.tri][:,1], self.y[self.tri][:,0]-self.y[self.tri][:,1]])
        BC    = np.array([self.x[self.tri][:,1]-self.x[self.tri][:,2], self.y[self.tri][:,1]-self.y[self.tri][:,2]])
        CA    = np.array([self.x[self.tri][:,2]-self.x[self.tri][:,0], self.y[self.tri][:,2]-self.y[self.tri][:,0]])
        return AB, BC, CA

    def _get_sidewall_lengths(self):
        '''return sidewall length'''
        # Get length of each triangle side
        AB, BC, CA = self._get_sidewall_segments()
        lAB   = np.sqrt(AB[0,:]**2+AB[1,:]**2)
        lBC   = np.sqrt(BC[0,:]**2+BC[1,:]**2)
        lCA   = np.sqrt(CA[0,:]**2+CA[1,:]**2)
        return lAB, lBC, lCA

class GridProximity:
    def find_nearest(self, x, y, grid = 'node'):
        '''
        Find indices of nearest node to given points x,y.
        grid = 'node' (default) or 'cell'
        '''
        query_points = np.array([x,y]).T 
        _, inds = getattr(self, f'{grid}_tree').query(query_points)
        return inds

    @property
    def cell_number(self):
        return len(self.xc)

    @property
    def node_number(self):
        return len(self.x)

    @property
    def node_tree(self):
        '''KDTree for nodes'''
        return KDTree(np.array([self.x, self.y]).T)

    @property
    def cell_tree(self):
        '''KDTree for cells'''
        return KDTree(np.array([self.xc, self.yc]).T)

    def find_within(self, x, y, r = None, grid = 'node'):
        '''
        Find all grid points witin radius "r" (in meters) from the input points
        grid: 'node' (default) or 'cell'
        '''
        query_points = np.array([x,y]).T 
        inds = getattr(self, f'{grid}_tree').query_ball_point(query_points, r)
        all_in_one_list = []
        for l in inds:
            all_in_one_list.extend(l)
        return np.unique(all_in_one_list)

    def isinside(self, x, y, x_buffer=0, y_buffer=0):
        '''
        Check if points are inside the rectangle bounded by the xy limits of the nodes, with optional bufer
        '''
        x_min, x_max = self.x.min() - x_buffer, self.x.max() + x_buffer
        y_min, y_max = self.y.min() - y_buffer, self.y.max() + y_buffer
        inside_x, inside_y = np.logical_and(x>=x_min, x<=x_max), np.logical_and(y>=y_min, y<=y_max)
        return np.logical_and(inside_x, inside_y)

    def _find_cells_in_box(self, xlim, ylim):
        '''
        find triangles in scope
        '''
        xm, xp = self.xc <= max(xlim), self.xc >= min(xlim)
        ym, yp = self.yc <= max(ylim), self.yc >= min(ylim)
        xind, yind = np.logical_and(xm, xp), np.logical_and(ym, yp)
        cells = np.arange(len(self.xc))
        return cells[np.logical_and(xind, yind)]

    def _check_nv(self):
        '''
        Based on code from stackoverflow:
        https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
        '''
        neg = self._test_nv(self.x, self.y, self.tri)

        if neg.size == 0:
            if self.verbose: print('- Triangulation from source file is anti-clockwise')

        elif neg.size > 0:
            assert len(neg) == self.cell_number, 'The triangulation direction is inconsistent, this mesh is invalid.'
            self.tri  = np.array([self.tri[:, 0], self.tri[:,2], self.tri[:,1]], dtype = np.int64).T
            neg = self._test_nv(self.x, self.y, self.tri)
            assert neg.size == 0, '*Something* went wrong when trying to re-arrange the triangulation to clockwise direction'
            if self.verbose: print('- Triangulation from source was clockwise, it has now been re-arranged to be anti-clockwise')

    # Static methods
    @staticmethod
    def _test_nv(x,y,nv):
        '''checks if the triangles are sorted clockwise or anti clockwise (anti clockwise = True)'''
        # Triangle corners
        xpts, ypts  = x[nv], y[nv]

        # Edges
        e1    = (xpts[:,1] - xpts[:,0])*(ypts[:,1] + ypts[:,0])
        e2    = (xpts[:,2] - xpts[:,1])*(ypts[:,2] + ypts[:,1])
        e3    = (xpts[:,0] - xpts[:,2])*(ypts[:,0] + ypts[:,2])

        # Direction test
        loop  = e1+e2+e3
        neg   = np.where(loop > 0)[0]
        return neg

class LegacyPropertyAliases:
    '''
    Aliases for properties with mutliple names, used when writing forcing/initial files (turns out these names are actually used by FVCOM, won't get around them)
    '''
    @property
    def siglev_center(self):
        return self.siglev_c

    @property
    def siglay_center(self):
        return self.siglay_c

class PropertiesFromTGE:
    '''
    These grid connectivity indentifiers are handy when dealing with the mesh using the same methods as FVCOM does.
    '''
    @property
    def T(self, file = None):
        '''
        T computes the same grid connectivity metrics as tge.F
        Does also include routines to compute gradients, vorticity and okubo weiss parameter.
        '''
        if not hasattr(self, '_T'):
            try:
                self._T = tge.TGE('tge.npy')
                print('- read grid connectivity from tge.npy')
            except:
                self._T = tge.main(self, verbose = True)
            self._nbsn = self._T.NBSN
            self._ntsn = self._T.NTSN
            self._nbve = self._T.NBVE
            self._ntve = self._T.NTVE
            self._nbe = self._T.NBE
            self._nbse, self._nese = tge.get_NBSE_NESE(self.nbve, self.ntve, self.tri, len(self.xc))
        return self._T

    @property
    def nbe(self):
        '''indices of elements sharing a wall with *this* element'''
        if not hasattr(self, '_nbe'): 
            self._nbe = self.T.NBE
        return self._nbe

    @nbe.setter
    def nbe(self, var):
        self._nbe = var

    @property
    def nbsn(self):
        '''indices of nodes surrounding each node'''
        if not hasattr(self, '_nbsn'): 
            self._nbsn = self.T.NBSN
        return self._nbsn

    @nbsn.setter
    def nbsn(self, val):
        self._nbsn = val

    @property
    def ntsn(self):
        '''number of nodes surrounding each node'''
        if not hasattr(self, '_ntsn'): 
            self._ntsn = self.T.NTSN
        return self._ntsn

    @ntsn.setter
    def ntsn(self, val):
        self._ntsn = val

    @property
    def nbve(self):
        '''indices of elements surrounding each node'''
        if not hasattr(self, '_nbve'): 
            self._nbve = self.T.NBVE
        return self._nbve

    @nbve.setter
    def nbve(self, val):
        self._nbve = val

    @property
    def nese(self):
        '''number of elements surrounding each element'''
        if not hasattr(self, '_nese'): 
            self._nbse, self._nese = tge.get_NBSE_NESE(self.nbve, self.ntve, self.tri, len(self.xc))
        return self._nese

    @property
    def nbse(self):
        '''indices of elements surrounding each element'''
        if not hasattr(self, '_nbse'): 
            self._nbse, self._nese = tge.get_NBSE_NESE(self.nbve, self.ntve, self.tri, len(self.xc))
        return self._nbse

    @property
    def ntve(self):
        '''the number of elements connected to this node'''
        if not hasattr(self, '_ntve'): 
            self._ntve = self.T.NTVE
        return self._ntve

    @ntve.setter
    def ntve(self, var):
        self._ntve = var

    @property
    def art1(self):
        '''the node centered control area'''
        if not hasattr(self, '_art1'):
            self.T.get_art1()
            self._art1 = self.T.art1
        return self._art1

    @art1.setter
    def art1(self, val):
        self._art1 = val

class CellTriangulation:
    @property
    def ctri(self):
        '''Triangulation using xc, yc as nodes. Land is masked.'''
        if not hasattr(self, '_ctri'): 
            self._ctri = self.cell_tri()
        return self._ctri

    @ctri.setter
    def ctri(self, var):
        self._ctri = var

    def cell_tri(self):
        '''Get triangulation connecting cell centres.'''
        import fvtools.grid.tools as tools
        if self.verbose: print('- Triangulating using Delaunays method')
        ctri  = matplotlib.tri.Triangulation(self.xc, self.yc)
        masked_ctris = tools.mask_land_tris(self.x, self.y, self.tri, ctri.triangles)
        return masked_ctris

class CropGrid:
    '''
    It is convenient to extract a subset of the FVCOM grid, such as done here
    '''
    @property
    def cropped_nodes(self):
        if not hasattr(self, '_cropped_nodes'):
            return np.array(0)
        return self._cropped_nodes

    @cropped_nodes.setter 
    def cropped_nodes(self, var):
        self._cropped_nodes = var

    @property
    def cropped_cells(self):
        if not hasattr(self, '_cropped_cells'):
            return np.array(0)
        return self._cropped_cells

    @cropped_cells.setter 
    def cropped_cells(self, var):
        self._cropped_cells = var

    def interactive_cell_selection(self, xlim = None, ylim = None, inverse = False):
        '''
        Get cells within (or outside) a user defined polygon
        '''
        from matplotlib.widgets import PolygonSelector
        from matplotlib.path import Path
        ax = plt.gca()
        ax.plot(self.x[self.coastline_nodes], self.y[self.coastline_nodes], 'r.')
        ax.set_aspect('equal')
        selector = PolygonSelector(ax, lambda *args: None, useblit=True)
        print("Click on the figure to create a polygon.\nHold the 'shift' key to move all of the vertices.\nHold the 'ctrl' key to move a single vertex.")
        plt.title('Close the figure when you have a closed polygon')
        plt.show(block=True)

        p = Path(np.vstack([[x for x, y in selector.verts], [y for x, y in selector.verts]]).T)
        within = np.where(p.contains_points(np.vstack([self.xc, self.yc]).T))
        if inverse:
            _cells = np.ones(self.xc.shape, dtype=bool)
            _cells[within[0]] = False
            cells = np.where(_cells)[0]
        else:
            cells = within[0]
        return cells

    def subgrid(self, xlim=None, ylim=None, cells=None):
        '''
        Create indices that crop your FVCOM grid to a smaller, more managable version

        Two input options:
        - xlim, ylim: subdomain (x, y)-limits in carthesian coordinates (list, np.array)
        - cells:      Cells to use in cropped mesh (if chosen, we will *not* use xlim, ylim) (np.array)

        Returns:
        - FVCOM_grid instance for the cropped mesh.
          - nodestrings where the cropped mesh connects to the original mesh
          - same depth information as the original mesh
          - same mesh reference as the original mesh
          - same casenane as the original mesh
          - properties through TGE needs to be re-calculated
        '''
        if cells is None: 
            cells = self._find_cells_in_box(xlim, ylim)
        x, y, tri, nodes, cells = self.remap(self.x, self.y, self.tri, cells)

        # Create the cropped FVCOM_grid object
        kwargs = {}
        kwargs['x'], kwargs['y'], kwargs['tri'] = x, y, tri
        if self.h is not None and not np.all(self.h == None):
            kwargs['h'] = self.h[nodes]
        if self.siglev is not None and not np.all(self.siglev == None):
            kwargs['siglev'] = self.siglev[nodes, :]
        if self.siglay is not None and not np.all(self.siglay == None):
            kwargs['siglay'] = self.siglay[nodes, :]
        if self.cropped_nodes.any(): 
            kwargs['cropped_nodes'] = self.cropped_nodes[nodes]
            kwargs['cropped_cells'] = self.cropped_cells[cells]
        else:
            kwargs['cropped_nodes'] = nodes
            kwargs['cropped_cells'] = cells

        new_self = FVCOM_grid(reference = self.reference, **kwargs)
        new_self.casename = self.casename

        #coastline_bool = np.zeros(self.x.shape, dtype = bool)
        #coastline_bool[self.coastline_nodes] = True
        #new_self._update_coastline_nodes(coastline_bool[nodes])
        #new_self._update_obc()
        #new_self._project_xy()
        return new_self

    def remap(self, x, y, nv, cropped_cells):
        '''
        Returns a remapped version of the mesh, guarantees a valid mesh as output
        '''
        while True:
            # See which nodes to keep (node_ind) and define new node indices
            cropped_nodes = np.unique(nv[cropped_cells])
            new_node_ind  = np.arange(len(cropped_nodes), dtype = np.int32)

            # Create a map from old indexing to new
            all_nodes     = -1*np.ones((len(x)), dtype = np.int32)
            all_nodes[cropped_nodes] = new_node_ind
            cropped_cells_tmp = self._return_valid_triangles(x[cropped_nodes], y[cropped_nodes], all_nodes[nv[cropped_cells]])
            if len(cropped_cells) == len(cropped_cells_tmp):
                break
            else:
                cropped_cells = cropped_cells[cropped_cells_tmp.astype(int)]
        return x[cropped_nodes], y[cropped_nodes], all_nodes[nv[cropped_cells]], np.array(cropped_nodes), cropped_cells

    def _return_valid_triangles(self, x, y, tri):
        '''
        Only return valid triangles
        '''
        _tri  = matplotlib.tri.Triangulation(x, y, tri)
        _nbrs = _tri.neighbors
        _nbrs[np.where(_nbrs > 0)] = 0
        _illegal = np.where(np.sum(_nbrs, axis = 1)==-2)[0]
        return np.array([element for element in range(tri.shape[0]) if element not in _illegal])

    def _update_obc(self):
        '''
        Update local land nodes, and local obc_nodes.
        '''
        if any(self.coastline_nodes):
            ocean_obc_nodes, cells_connected_to_boundary = self._find_ocean_obc_nodes()
            self.obc_nodes = self._connect_ocean_obc_to_land(ocean_obc_nodes, cells_connected_to_boundary)

    def _update_coastline_nodes(self, coastline_bool):
        self.coastline_nodes = np.where(coastline_bool)[0].tolist()

    def _find_ocean_obc_nodes(self):
        '''Finds all obc-nodes that are not in touch with the original grids coast'''
        zeros = np.zeros(self.x.shape)
        zeros[self.full_model_boundary] = 1
        coast_sum = np.sum(zeros[self.tri], axis = 1)
        cells_connected_to_boundary = np.where(coast_sum>0)[0]
        _, comm1, _ = np.intersect1d(self.full_model_boundary, np.unique(self.coastline_nodes), return_indices=True, assume_unique=True)
        _exterior_nodes_not_in_coastline = np.ones(self.full_model_boundary.shape, dtype = bool)
        _exterior_nodes_not_in_coastline[comm1] = False
        return self.full_model_boundary[np.where(_exterior_nodes_not_in_coastline)[0].astype(int)], cells_connected_to_boundary

    def _connect_ocean_obc_to_land(self, ocean_obc_nodes, cells_connected_to_boundary):
        '''connect the obc to land, assumes that no triangle has more than one side facing the OBC'''
        _bool = np.zeros(self.x.shape, dtype=bool)
        _bool[ocean_obc_nodes] = True
        _tribool = _bool[self.tri].any(axis=1)
        obc_side_nodes = np.intersect1d(np.unique(self.tri[_tribool,:]), np.unique(self.coastline_nodes)).astype(int)
        return list(np.append(ocean_obc_nodes, obc_side_nodes))

class CoastLine:
    '''
    Identify the coastline nodes (boundary nodes that are not in the obc)
    '''
    @cached_property
    def coastline_nodes(self):
        '''
        Model nodes at the coastline
        '''
        fb_in_obc = np.intersect1d(self.full_model_boundary, self.obc_nodes)
        inds_db_in_obc = np.intersect1d(fb_in_obc, self.full_model_boundary, return_indices=True)
        obc_bool = np.ones(self.full_model_boundary.shape, dtype = bool)
        obc_bool[inds_db_in_obc[2]] = False
        coastline_nodes = self.full_model_boundary[obc_bool]

        # Add the points connecting of the nodestrings to the coastline
        #for nodestring in self.nodestrings:
        #    coastline_nodes.extend(nodestring[[0,-1]].tolist())
        return coastline_nodes

    @cached_property
    def model_boundary(self):
        '''
        Outer boundary of this model domain (x,y)
        '''
        polygons = self.get_land_polygons()
        areas = [p.area for p in polygons]
        polygon = [l for l, a in zip(polygons, areas) if a==max(areas)]
        _model_boundary = polygon[0].exterior
        return _model_boundary

    @cached_property
    def full_model_boundary(self):
        '''
        An array containing all nodes connected to the boundary (i.e. OBC and all coastline nodes)
        '''
        x, y = [], []
        for pol in self.get_land_polygons():
            xtmp, ytmp = pol.exterior.xy
            x.extend(xtmp)
            y.extend(ytmp)
        return np.unique(self.find_nearest(x, y, grid = 'node'))

    @property
    def ocean_polygons(self):
        polygons = self.get_land_polygons()
        areas = [p.area for p in polygons]
        polygon = [l for l, a in zip(polygons, areas) if a==max(areas)]
        _ = polygons.pop(np.where(np.array(polygons)==polygon)[0][0])
        return polygons

    def get_land_polygons(self):
        '''
        the same way as done in trigrid
        '''
        from shapely.ops import polygonize
        self._check_nv()

        # Get triangle wall (pairs of nodes) connected to land
        boundary_nodes = self._get_land_segments() 
        polygons = list(polygonize(
            (n1, n2) for n1, n2 in zip(np.array([self.x[boundary_nodes][:,0], self.y[boundary_nodes][:,0]]).T, 
                                       np.array([self.x[boundary_nodes][:,1], self.y[boundary_nodes][:,1]]).T)
                                  )
                        )
        return polygons

    def is_on_mesh(self, x, y):
        '''
        Check if points are within the mesh
        '''
        # Return points that are within the model domain
        from matplotlib.path import Path
        assert len(x)==len(y)
        points = np.vstack([x,y])

        def check_points(xpol, ypol, points):
            path   = Path(np.vstack([np.array(xpol), np.array(ypol)]).T)
            return path.contains_points(points.T)
            
        xpol, ypol = self.model_boundary.xy
        inside = check_points(xpol, ypol, points)
        points = points[:, inside]

        for pol in self.ocean_polygons:
            xpol, ypol = pol.exterior.coords.xy
            inside = check_points(xpol, ypol, points)
            points = points[:, ~inside]
        return points

    def _get_land_segments(self):
        '''Construct segments connected to land'''
        neighbors = matplotlib.tri.Triangulation(self.x, self.y, self.tri).neighbors
        identify  = np.where(neighbors == -1)
        boundary_elements = identify[0]  # Index of triangle next to land
        nodes_1 = identify[1]   # index at first node next to land
        nodes_2 = (nodes_1+1)%3 # Next corner at land (each node in triangulation is counted clockwise)
        return np.array([self.tri[boundary_elements, nodes_1], self.tri[boundary_elements, nodes_2]]).transpose()

class OBC:
    @property
    def obc_nodes(self):
        '''Keeps track of user defined OBC nodes. May be better to keep track of land nodes and define the leftovers to be obc-nodes (as discussed in trigrid repo)'''
        if not hasattr(self, '_obc_nodes'):
            self._obc_nodes = []
        return self._obc_nodes
    
    @obc_nodes.setter 
    def obc_nodes(self, var):
        self._obc_nodes = var

    @property
    def nodestrings(self):
        '''Each OBC nodestring stored as numpy arrays in a list [np.array(nodestring1), np.array(nodestring2), ...]'''
        if not hasattr(self, '_nodestrings'):
            if any(self.obc_nodes):
                self._nodestrings = self._get_nodestrings(self.obc_nodes)
            else:
                self._nodestrings = []
        return self._nodestrings

    @nodestrings.setter
    def nodestrings(self, var):
        self._nodestrings = var

    @property
    def x_obc(self):
        '''x-position of nodes at the open boundary'''
        return np.array([self.x[nodestring] for nodestring in self.nodestrings])

    @property
    def y_obc(self):
        '''y-position of nodes at the open boundary'''
        return np.array([self.y[nodestring] for nodestring in self.nodestrings])
    
    @property
    def lat_obc(self):
        '''latitude of nodes on the open boundary'''
        return np.array([self.lat[nodestring] for nodestring in self.nodestrings])

    @property
    def lon_obc(self):
        '''longitude of nodes on the open boundary'''
        return np.array([self.lon[nodestring] for nodestring in self.nodestrings])

    def _get_nodestrings(self, obc_nodes):
        '''Connects nodestrings to distinguished lines based on triangulation connectivity.
        - *should* guarantee that the nodes are sorted correctly, but still subject to testing
        '''
        import networkx as nx
        node_tag = np.zeros((self.cell_number,), dtype = np.int32)
        node_tag[self.full_model_boundary] = 1
        cells = np.where(np.sum(node_tag[self.tri], axis=1) > 0)[0]
        tri = self.tri[cells,:]
        side1, side2, side3 = [tri[:, [i, (i+1)%3]] for i in range(3)]
        _sides = []
        for sides in [side1, side2, side3]:
            _sides.extend([side.tolist() for side in sides if side[0] in obc_nodes and side[1] in obc_nodes])

        G = nx.Graph() 
        _ = [G.add_edge(*side) for side in _sides]
        _nodestrings = sorted(map(sorted, nx.k_edge_components(nx.DiGraph(G), 1)))
        nodestrings = [np.array(list(nodestring)) for nodestring in _nodestrings]

        # Make sure that the endpoints of each nodestring are at the start and stop of the array (since the set can't be returned in an orderly fashion)
        # - generalize to make a nice nodestring, won't need to know if it is circular or a line string
        nodes, counts = np.unique(_sides, return_counts = True)
        endpoints = nodes[np.where(counts==1)[0]]    
        ordered_nodestrings = []
        for nodestring in nodestrings:
            start, stop = [node for node in nodestring if node in endpoints]
            tmp = np.copy(nodestring)
            if start != tmp[0]:
                ind_start = np.where(tmp == start)
                tmp[ind_start] = tmp[0]
                tmp[0] = start
            if stop != tmp[-1]:
                ind_stop = np.where(tmp == stop)
                tmp[ind_stop] = tmp[-1]
                tmp[-1] = stop
            ordered_nodestrings.append(np.array(tmp))
        return ordered_nodestrings

class ControlVolumePlotter:
    def plot_cvs(self, field,
                 cmap = 'jet', 
                 cmax = None, 
                 cmin = None, 
                 Norm = None, 
                 edgecolor = 'face', 
                 verbose = True):
        '''
        Plots nodal fields on control volume patches

        Optional:
        ----
        - edgecolor: None by default. Set a color, e.g. "k" to draw interfaces between CVs
        - cmap:      colormap
        - cmax:      cap colors to cmax
        - cmin:      floor colors to cmin
        - Norm:      to create non-linear colorbars (ref matplotlib documentation)
        - verbose:   to get progress reports
        --> for cmax and cmin, the routine exclusively plots CVs with values within in that range
        '''
        if not hasattr(self, 'cv'):
            self._load_cvs(verbose)
        inds = np.arange(0, len(field))

        # mask cvs with values below/above threshold
        if cmax is not None and cmin is None:
            inds = np.where(field<=cmax)[0]
        elif cmin is not None and cmax is None: 
            inds = np.where(field>=cmin)[0]
        elif cmax is not None and cmin is not None:
            inds = np.intersect1d(np.where(field<=cmax)[0], np.where(field>=cmin)[0])

        if len(field) == len(self.cv): 
            collection = PatchCollection(self.cv, cmap=cmap, edgecolor=edgecolor, norm=Norm)
        else: 
            collection = PatchCollection([self.cv[inds]], cmap=cmap, edgecolor=edgecolor, norm=Norm)
        collection.set_array(field[inds.astype(int)])
        ax   = plt.gca()
        _ret = ax.add_collection(collection)
        ax.autoscale_view(True)
        ax.set_aspect('equal')
        # add colorbar
        cbar = plt.colorbar(collection, ax=ax)
        cbar.set_label("")  # Optional: remove label if not needed
        if _ret._A is not None: 
            plt.gca()._sci(_ret)
        return _ret

    def _load_cvs(self, verbose):
        '''Check if we already have the cv-patches'''
        try:
            self.cv = np.load('cvs.npy', allow_pickle = True)
        except:
            self._get_cvs() # create CV polygons

    def _get_cvs(self, nodes = None):
        '''
        - computes matplotlib-patches for each control voulme, adds to self as self.cvs
        - Returns numpy file (cvs.npy) containing all of the control volume patches
        '''
        from matplotlib.patches import Polygon as mPolygon
        if nodes is None: 
            nodes = np.arange(0, self.node_number)
        xcv_full, ycv_full = self._get_control_volumes(self.nbsn, self.nbve, self.x, self.y, self.xc, self.yc, nodes)
        bar    = pb.ProgressBar(widgets = ['Making control volume patches: ', pb.Percentage(), ' ', pb.BouncingBar(), pb.AdaptiveETA()], maxval=len(nodes))
        bar.start()
        self.cv = []
        for i, (xcv, ycv) in enumerate(zip(xcv_full, ycv_full)):
            bar.update(i)
            self.cv.append(mPolygon(np.array([xcv, ycv]).T, closed=True))
        bar.finish()
        np.save('cvs.npy', self.cv) # for instant access to cvs at a later date

    @staticmethod
    def _get_control_volumes(nbsn, nbve, x, y, xc, yc, nodes):
        '''returns list of control volumes'''
        xcv_full, ycv_full = [], []
        for i, node in enumerate(nodes):
            indexes   = np.where(nbsn[node,:]>=0)[0]
            nbsnhere  = nbsn[node, indexes]
            elemshere = nbve[node, indexes]

            xcv, ycv = _find_cv_walls(node, nbsnhere, elemshere, x, y, xc, yc)
            xcv_full.append(xcv)
            ycv_full.append(ycv)
        return xcv_full, ycv_full

@njit
def _find_cv_walls(node, nbsnhere, elemshere, x, y, xc, yc):
    '''find points that should connect to make a cv wall'''
    xmid  = (x[node] * np.ones((len(nbsnhere),)) + x[nbsnhere])/2
    ymid  = (y[node] * np.ones((len(nbsnhere),)) + y[nbsnhere])/2
    xcell, ycell = xc[elemshere], yc[elemshere]
    return _connect_cv_walls(xmid, ymid, xcell, ycell, x, y, nbsnhere, elemshere, node)

@njit
def _connect_cv_walls(xmid, ymid, xcell, ycell, x, y, nbsnhere, elemshere, node):
    xcv = []; ycv = []
    for i in range(len(xmid)):
        xcv.append(xmid[i])
        ycv.append(ymid[i])

        # Find direction
        boundary = False
        if i < len(xmid)-1:
            xcvmid = (xmid[i]+xmid[i+1])/2
            ycvmid = (ymid[i]+ymid[i+1])/2
            dst    = np.sqrt((xcell-xcvmid)**2+(ycell-ycvmid)**2)
            ind    = np.argwhere(dst==dst.min())[0]

            # Be aware of land
            if nbsnhere[i+1] == node:
                boundary = True
                xcv.append(x[nbsnhere[i+1]])
                ycv.append(y[nbsnhere[i+1]])
            elif nbsnhere[i] == node:
                boundary = True
        if not boundary:
            xcv.append(xcell[ind][0])
            ycv.append(ycell[ind][0])

        uxcv    = np.array(xcv)
        unique  = np.unique(uxcv)
        nunique = len(unique)
        ncv     = len(xcv)
        nunique_lt_ncv = nunique < ncv
        if boundary and nunique_lt_ncv:
            break
    return xcv, ycv

class InterpolateToZ:
    def make_interpolation_matrices_TS(self, interpolation_depths=[-5]):
        '''Make matrices (numpy arrays) that, when multiplied with fvcom T or S matrix, interpolates data to a given depth.'''
        import fvtools.grid.tools as tools
        for depth in interpolation_depths:
            interp_matrix = tools.make_interpolation_matrices(self.siglayz, depth)
            setattr(self, 'interpolation_matrix_TS_' + str(abs(int(depth))) + '_m', interp_matrix)

    def make_interpolation_matrices_uv(self, interpolation_depths=[-5]):
        '''Make matrices (numpy arrays) that, when multiplied with fvcom u or v matrix, interpolates data to a given depth.'''
        import fvtools.grid.tools as tools
        for depth in interpolation_depths:
            interp_matrix = tools.make_interpolation_matrices(self.siglayz_uv, depth)
            setattr(self, f'interpolation_matrix_uv_{abs(int(depth))}_m', interp_matrix)

    def interpolate_to_z(self, f, z, depths = None):
        '''Interpolate data to user defined z-level
        - f:       datafield
        - z:       depth [m] to interpolate to f (positive number)

        Optional:
        - depths:  depth of grid f. Positive downward. (since using a cached siglevz will be faster than forcing FVCOM_grid to find it each iteration)

        Plot returned data using .plot_contours(data)
        '''
        if depths is None:
            for depth in ['siglevz', 'siglayz', 'siglevz_uv', 'siglayz_uv']:
                depths = -getattr(self, depth).T
                if f.shape == depths.shape:
                    break
                else:
                    depths = np.empty(0)
            assert depths.any(), 'Dimension does match siglev, siglay, siglev_center or siglay_center, try to transpose the input field'

        # Find closest depth levels
        depthsmax, depthsmin = np.copy(depths), np.copy(depths)
        lt0 = (depths-z)<0
        depthsmax[np.where(lt0)] = np.nan
        depthsmin[np.where(lt0 == False)] = np.nan
        depbot, deptop = np.nanmin(depthsmax, axis = 0), np.nanmax(depthsmin, axis = 0)

        # Extract field data from nearest level above, and nearest level below. Use linear interpolation to get to z
        f_t, f_b = np.nanmax(np.where(depths == deptop, f, np.nan), axis = 0), np.nanmax(np.where(depths == depbot, f, np.nan), axis = 0)
        dz, dzt, dzb = depbot-deptop, z-deptop, depbot-z
        zf = (f_t*dzb + f_b*dzt)/dz        
        zf[np.where(np.isnan(f_t))] = f_b[np.where(np.isnan(f_t))] # Use nearest valid value if NaN at the surface (do not touch the bottom though)
        return zf

class SectionMaker:
    '''
    Rotuines to define a section within the grid, interpolate data to the section and return the transect
    - Will actually not be 100% correct, since we "crop" the sea surface elevation somewhat to make the bottom appear to be static. That is not the case, the z-level of
      the bottom sigma level is not static (since min(sigma)>-1).
    - Still room to make this more consise...
    '''
    @property
    def dst_sec(self):
        '''Distance along a transect measured in kilometers from first point'''
        dx = np.diff(self.x_sec, prepend = self.x_sec[0])/1000 # since we count starting with the first segment position
        dy = np.diff(self.y_sec, prepend = self.y_sec[0])/1000 # dive by 1000 to get distance in kilometers
        return np.cumsum(np.sqrt(dx**2 + dy**2))

    @property
    def dpt_sec(self):
        '''The depths along the section'''
        if self._transect_grid == '':
            z = matplotlib.tri.LinearTriInterpolator(self.trs, self.zeta).__call__(self.x_sec,  self.y_sec)[:, None] * (-1)*self._min_sigma
            _dpt_sec = matplotlib.tri.LinearTriInterpolator(self.trs, self.d).__call__(self.x_sec,  self.y_sec)[:, None]*getattr(self, self._transect_sigma)[0,:] + z
        else:
            z = matplotlib.tri.LinearTriInterpolator(self.trs, np.mean(self.zeta[self.tri], axis = 1)).__call__(self.x_sec,  self.y_sec)[:, None]  * (-1) * self._min_sigma
            _dpt_sec = matplotlib.tri.LinearTriInterpolator(self.trs, self.dc).__call__(self.x_sec, self.y_sec)[:, None]*getattr(self, self._transect_sigma)[0,:] + z
        return _dpt_sec

    @property
    def _min_sigma(self):
        '''bottom sigma level value. included since I want to keep bottom depth constant in section movies (which is a lie...)'''
        _min_sigma = -1
        if self._static_depth: _min_sigma = self.siglay[-1, -1]
        return _min_sigma

    def prepare_section(self, section_file = None, res = None, store_transect_img = False):
        '''Returns x, y points along a section for use in section-analysis'''
        if section_file is None: 
            x_section, y_section = self._graphical_section()
        if section_file is not None: 
            x_section, y_section = self._load_section_file(section_file)
        if res is None: 
            res = np.sum(np.sqrt(np.diff(x_section)**2 + np.diff(y_section)**2))/60
        self._assemble_section(x_section, y_section, res)
        if section_file is None: 
            self.write_section()
        if hasattr(self, 'trs'): 
            delattr(self, 'trs')
        if store_transect_img: 
            self._plot_section_line()
        plt.close('all')

    # Rotuines that define where the section goes (ie. defining nodes on the section line)
    def _graphical_section(self):
        '''Define a transect by plotting on a map'''
        c = self.plot_contour(self.h, levels = np.linspace(self.h.min(), self.h.max(), 30))
        self.plot_grid(rasterized=True)
        plt.title('Click where you want the section to go (minimum two points). Hit "Enter" to continue.')
        places = plt.ginput(timeout = -1, n = -1)
        return np.array([p[0] for p in places]), np.array([p[1] for p in places])

    def _load_section_file(self, file):
        '''
        Load section coordinates from a file
        - accepts both ',' and ' ' as delimiters
        '''
        try:
            lon, lat = np.loadtxt(file, delimiter = ',', unpack = True)
        except:
            lon, lat = np.loadtxt(file, delimiter = ' ', unpack = True)
        return self.Proj(lon, lat)

    def _assemble_section(self, x_section, y_section, res):
        '''Create an evenly spaced section'''
        self.x_sec, self.y_sec = np.empty(0), np.empty(0)
        for x_now, y_now, x_next, y_next in zip(x_section[:-1], y_section[:-1], x_section[1:], y_section[1:]):
            dst        = np.sqrt((x_next-x_now)**2+(y_next-y_now)**2)
            npoints    = np.ceil(dst/res).astype(int)
            self.x_sec = np.append(self.x_sec, np.linspace(x_now, x_next, npoints))
            self.y_sec = np.append(self.y_sec, np.linspace(y_now, y_next, npoints))

    def get_section_data(self, data,
                         section_file = None, res = None,
                         store_transect_img = False):
        '''
        Dump data from from the full solution to an interpolated transect  (ie. not following nodes/cells)
        - data:         Field for transect calculation
        - section_file: File with lon lat coordinate positions of segment fixed points. You can either use space or , as separator
        - res:          Resolution of the transect in m (defaults to 1/100 of span)
        - store_transect_img: Store a plot showing the transect overlayd on the grid

        Output (as dictionary):
          - x, y, dst, h positions, distance along and depth of transect
          - transect: field you gave the routine at x, y
        '''
        out = {}
        if data.shape[0] != self.node_number or data.shape[0] != self.cell_number: 
            data = data.T
        sigma = self._check_if_siglay_or_siglev(data)
        grid = self._check_if_cell_or_node(data)
        self._initialize_section(section_file, res, store_transect_img, sigma, grid) # TBD: in here, find way to remove points outside of the domain?
        out['h'], out['x'], out['y'] = self.dpt_sec, self.x_sec, self.y_sec

        # Interpolate data to the section
        if len(data.shape)>1:
            out['transect'] = self._interpolate_3D(data)
            out['dst'] = np.repeat(self.dst_sec[:, None], data.shape[1], axis = 1)
        else:
            out['transect'] = self._interpolate_2D(data)
            out['dst'] = self.dst_sec

        # Remove nans and crop dict, contourf won't accept the input otherwise -- do this earlier though, not every time...
        if np.isnan(out['transect']).any():
            if len(out['transect'].shape) == 1:
                not_horizontal = ~np.isnan(out['transect'])
            else:
                not_horizontal = ~np.isnan(out['transect'][:, 0])
            for key in out.keys():
                if len(out[key].shape) == 1:
                    out[key] = out[key][not_horizontal]
                else:
                    out[key] = out[key][not_horizontal, :]
        return out

    def _initialize_section(self, section_file, res, store_transect_img, sigma, grid):
        '''Prepare the section interpolators etc.'''
        self._transect_sigma, self._transect_grid = sigma, grid
        if not hasattr(self, 'x_sec') and not hasattr(self, 'y_sec'):
            self.prepare_section(section_file = section_file, res = res, store_transect_img = store_transect_img)
        if not hasattr(self, 'trs'):
            self.trs =  matplotlib.tri.Triangulation(getattr(self, f'x{self._transect_grid}'), getattr(self, f'y{self._transect_grid}'), 
                                                     triangles = getattr(self, f'{self._transect_grid}tri'))

    def _interpolate_2D(self, field):
        out_field = np.zeros((len(self.x_sec),))
        out_field[:] = matplotlib.tri.LinearTriInterpolator(self.trs, field).__call__(self.x_sec, self.y_sec)
        return out_field

    def _interpolate_3D(self, field):
        out_field = np.zeros((len(self.x_sec), field.shape[1]))
        for sigma in range(field.shape[1]):
            out_field[:, sigma] = matplotlib.tri.LinearTriInterpolator(self.trs, field[:, sigma]).__call__(self.x_sec, self.y_sec)
        return out_field

class ExportGrid:
    '''
    Routines to export data in FVCOM_grid to other formats
    '''
    def get_xy(self, latlon):
        '''
        will return the positions we need to write to a file depending on wether you write for a spherical or carthesian FVCOM run 
        '''
        if latlon:
            return self.lon, self.lat
        else:
            return self.x, self.y

    def write_bath(self, filename=None, latlon = False):
        '''- Generates an ascii FVCOM 4.x format bathymetry file'''
        if filename is None: 
            filename = f'input/{self.casename}_dep.dat'
        x_grid, y_grid = self.get_xy(latlon)
        with open(filename, 'w') as f:
            f.write(f'Node Number = {self.node_number}\n')
            for x, y, h in zip(x_grid, y_grid, self.h):
                line = '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(h)+'\n'
                f.write(line)
        print(f'  - Wrote : {filename}')

    def write_grd(self, filename = None, latlon = False):
        '''- Generates an ascii FVCOM 4.x format grid file'''
        if filename is None: 
            filename = f'input/{self.casename}_grd.dat'
        x_grid, y_grid = self.get_xy(latlon)  
        with open(filename, 'w') as f:
            f.write(f'Node Number = {self.node_number}\n')
            f.write(f'Cell Number = {self.cell_number}\n')
            for i, (t1, t2, t3) in enumerate(self.tri):
                f.write(f'{i+1} {t1+1} {t2+1} {t3+1} {i}\n')
            for i, (x, y) in enumerate(zip(x_grid, y_grid)):
                line = str(i+1) +' '+ '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(0.0)+'\n'
                f.write(line)
        print(f'  - Wrote : {filename}')

    def write_sponge(self, filename = None):
        '''- Generates an ascii FVCOM 4.x formatted sponge file'''
        if filename is None: 
            filename = f'input/{self.casename}_spg.dat'
        with open(filename, 'w') as f:
            f.write(f'Sponge Node Number = {len(self.sponge_nodes)}\n')
            if any(self.sponge_nodes):
                for node in self.sponge_nodes:
                    line = str(node+1) + ' ' + '{0:.6f}'.format(self.sponge_radius) + ' ' + '{0:.6f}'.format(self.sponge_factor)+'\n'
                    f.write(line)
        print(f'  - Wrote : {filename}')

    def write_obc(self, filename = None):
        '''- Generates an ascii FVCOM 4.x formatted obc file'''
        if filename is None: 
            filename = f'input/{self.casename}_obc.dat'
        
        print(f"  - self.obc_nodes length: {len(self.obc_nodes)}")
        print(f"  - self.obc_nodes: {self.obc_nodes}")

    
        with open(filename, 'w') as f:
            f.write(f'OBC Node Number = {len(self.obc_nodes)}\n')
            i = 0
            for i, obc_node in enumerate(self.obc_nodes):
                f.write(f'{i+1} {obc_node+1} 1\n')  # obc_node+1 to shift to 1-based indexing
        print(f'  - Wrote : {filename}')

    def write_cor(self, filename = None, latlon = True):
        '''- Generates an ascii FVCOM 4.x formatted coriolis file'''
        if filename is None: 
            filename = f'input/{self.casename}_cor.dat'
        x_grid, y_grid = self.get_xy(latlon)  
        with open(filename, 'w') as f:
            f.write(f'Node Number = {self.node_number}\n')
            for x, y, lat in zip(x_grid, y_grid, self.lat):
                line = '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(lat)+'\n'
                f.write(line)
        print(f'  - Wrote : {filename}')

    def write_draft_nc(self, filename=None):
        """Write the ice shelf draft to a NetCDF file in FVCOM format"""
        import netCDF4 as nc
        import os

        if filename is None:
            filename = f'input/{self.casename}_draft.nc'

        print(f'  - Writing NetCDF draft file: {filename}')
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with nc.Dataset(filename, 'w', format='NETCDF4') as ds:
            # Define dimensions
            node_dim = ds.createDimension('node', len(self.x))

            # Define variables
            x_var = ds.createVariable('x', 'f4', ('node',))
            y_var = ds.createVariable('y', 'f4', ('node',))
            zisf_var = ds.createVariable('zisf', 'f4', ('node',))

            # Set variable attributes
            x_var.long_name = "nodal x-coordinate"
            x_var.units = "meters"

            y_var.long_name = "nodal y-coordinate"
            y_var.units = "meters"

            zisf_var.long_name = "Iceshelf Draft"
            zisf_var.units = "m"
            zisf_var.grid = "fvcom_grid"
            zisf_var.coordinates = ""
            zisf_var.type = "data"

            # Assign data
            x_var[:] = self.x
            y_var[:] = self.y
            zisf_var[:] = self.zisf

            # Global attributes
            ds.title = "FVCOM iceshelf draft File"
            ds.CoordinateSystem = "spherical"

        print(f'  - Done writing NetCDF: {filename}')
        

    def write_coast(self, outfolder = None, buffer_width = 50000, verbose = False):
        '''
        Finds the coastline in your domain, writes to casename_coast.shp file
        ---
        - outfolder: Folder to store coast.shp
                     - default: current directory 
        - buffer:    Buffer side of polygon, basically the width of the "islands" we create for the exterior polygon
                     - must always be positive, buffer measured in meters
        '''
        # Move most of these functions to the CoastLine class
        import geopandas as gpd
        import shapely as shp
        from pyproj import Transformer
        if outfolder is None: 
            outfolder = os.getcwd()

        # Draw polygons based on the coastline
        islands = self.get_land_polygons()
        areas = [p.area for p in islands] # Find areas of the polygons to identify exterior polygon (max(areas) = exterior polygon)
        exterior = islands.pop(np.squeeze(np.where(np.array(areas) == max(areas)))) # Split islands and exterior polygon (e.g. remove exterior from islands)
        exterior_buffer = exterior.buffer(buffer_width, single_sided=True) # Exterior in one polygon

        # Cut the exterior polygon so that we can remove polygons connected to the OBC
        def construct_lines(line, polygon):
            '''Add segments of the inner and outer polygon to a list, prepare to polygonize'''
            x, y = polygon.exterior.xy
            for i in range(len(polygon.exterior.xy[0])-1):
                line.append(shp.geometry.LineString(np.array([[x[i], x[i+1]], [y[i], y[i+1]]]).T)) 

        lines = []
        construct_lines(lines, exterior_buffer)
        construct_lines(lines, exterior)

        # Add lines from OBC (in exterior) to the exterior_buffer polygon
        obc_points = []
        for nodestring in self.nodestrings:
            points = [nodestring[0], nodestring[-1]]
            ind = []
            for point in points:
                dst = np.sqrt((exterior_buffer.exterior.xy[0]-self.x[point])**2 + (exterior_buffer.exterior.xy[1]-self.y[point])**2)
                ind.append(np.where(dst==dst.min())[0][0])

            for point_ind, buffer_ind in zip(points, ind):
                lines.append(shp.geometry.LineString([[self.x[point_ind],   self.y[point_ind]],  
                                                      [np.array(exterior_buffer.exterior.xy[0])[buffer_ind], np.array(exterior_buffer.exterior.xy[1])[buffer_ind]], 
                                                      ]))

            # Make points on the obc, these will be used to mask parts of the obc polygons connecting to ocean
            obc_points.append(shp.geometry.Point(self.x[nodestring[int(len(nodestring)/2)]], self.y[nodestring[int(len(nodestring)/2)]])) # A point on the line
       
        exterior_polys = list(shp.ops.polygonize(lines)) # Connect lines to polygons
        cropped_polys = [pol for pol in exterior_polys if not exterior.contains(pol.representative_point())] # Remove interior polygon
        cropped_polys = [pol for pol in cropped_polys if not any([pol.distance(p) == 0 for p in obc_points])] # Remove polygons at obc
        islands.extend(cropped_polys) # Add obc polygons to the list with islands (land is now also an island)

        # Re-project to latlon since this is what OpenDrift expects
        project = Transformer.from_proj(self.Proj, Proj(init='epsg:3031')) # We always want to project to latlon, hence hardcoded
        islands_latlon = [shp.ops.transform(project.transform, pol) for pol in islands]

        # Put into geopandas object, save to outfolder
        print(f'- Writing {outfolder}/{self.casename}_coast.shp and associated files')
        s = gpd.GeoDataFrame({'FID': range(len(islands_latlon)), 'geometry': islands_latlon})
        s.to_file(f'{outfolder}/{self.casename}_coast.shp')
        if verbose:
            all_islands = gpd.GeoDataFrame({'FID': range(len(islands)), 'geometry': islands})
            all_islands.plot()
            self.plot_grid()
            
    def write_2dm(self, name = None):
        '''
        Writes a 2dm file for this grid
        name: to be saved
        '''
        import fvtools.grid.fvgrid as fvgrid
        if name is None: name = self.casename
        if any(self.obc_nodes):
            fvgrid.write_2dm(self.x, self.y, self.tri, nodestrings = self.nodestrings, name = name, casename = name)
        else:
            fvgrid.write_2dm(self.x, self.y, self.tri, name = name, casename = name)

    def write_section(self):
        '''write x_sec, y_sec to a casename_section.txt file'''
        lonlat = np.vstack((self.Proj(self.x_sec, self.y_sec, inverse=True))).T
        np.savetxt(f'{self.casename}_section.txt', lonlat, delimiter = ' ')
        print(f'- saved section positions to {self.casename}_section.txt')

    def to_npy(self, filename = 'M.npy'):
        '''Write no M.npy file'''
        M = {}
        self.nv = self.tri
        for var in ['x', 'y', 'lon', 'lat', 'h', 'h_raw', 'nv', 'siglay', 'siglev', 'obc_nodes', 'nodestrings', 'info', 'ts','zisf','zisf_raw','ntsn','nbsn']:
            try:
                M[var] = getattr(self, var)
            except:
                pass
        np.save(filename, M)
        print(f'- Wrote {filename}')

class PlotFVCOM:
    @property
    def transform(self):
        if not hasattr(self, '_transform'):
            return None
        return self._transform

    @transform.setter
    def transform(self, var):
        self._transform = var

    def plot_grid(self, c = 'g-', linewidth = 0.2, markersize = 0.2, show = True, *args, **kwargs):
        '''
        Plot mesh grid
        - arguments and keyword arguments are passed to pyplot.triplot
        '''
        ax = plt.gca()
        kwargs = self._transform_to_kwargs(**kwargs)
        ax.triplot(self.x, self.y, self.tri, c, *args, markersize=markersize, linewidth=linewidth, **kwargs)
        ax.set_aspect('equal')
        if show: plt.show(block = False)

    def plot_obc(self, **kwargs):
        '''plot the obc nodes, show different nodestrings'''
        ax = plt.gca()
        kwargs = self._transform_to_kwargs(**kwargs)
        for i in range(len(self.x_obc)):
            ax.scatter(self.x_obc[i], self.y_obc[i], zorder = 10, label=f'boundary nr. {i+1}', **kwargs)
            ax.scatter(self.x_obc[i][[0,-1]], self.y_obc[i][[0,-1]], zorder = 11, c = 'r', **kwargs) # start and end points 
        plt.draw()

    def plot_contour(self, field, show = True, *args, **kwargs):
        '''Plot contour of node- or cell data, basically just a shortcut for pyplot.tricontourf()'''
        f = np.squeeze(field)
        if len(f) == self.node_number:
            x, y, tri = self.x, self.y, self.tri
        elif len(f) == self.cell_number:
            x, y, tri = self.xc, self.yc, self.ctri
        else:
            assert False, f'The input fields shape {f.shape} does not match the cell dimension {self.xc.shape} or the node dimension {self.x.shape}'

        # Mask nan values if any and send to plot
        if np.isnan(f).any():
            kwargs['mask'] = np.isnan(f)[tri].any(axis=1)

        # Add transform if the current axes is a GeoAxes (for WMS georeferenced plots)
        kwargs = self._transform_to_kwargs(**kwargs)
        return self._plot_contour(x, y, tri, f, show, *args, **kwargs)

    def _transform_to_kwargs(self, **kwargs):
        '''
        Add a transformation to the plot if the background is a WMS server (and in general when the axes is a cartopy GeoAxes)
        '''
        if self.transform is not None:
            import cartopy.mpl.geoaxes as geoaxes
            if isinstance(plt.gca(), geoaxes.GeoAxes):
                kwargs['transform'] = self.transform
        return kwargs   

    def georeference(self, url='https://openwms.statkart.no/skwms1/wms.topo4.graatone?service=wms&request=getcapabilities', 
                           layers=['topo4graatone_WMS'], wms=None,
                           depth=True):
        '''
        Plot map data from WMS server as georeference. Must be done before plotting grid, contours etc.
        - requires cartopy
        ---
        url:    defaults to grey norgeskart
        layers: valid layer from the url, must be a tuple with string(s)
        wms:    overwrites other input if not None, current options: "raster" or "topo4"
        depth:  plot kartverket bathymetry

        Returns:
        ---
        ax, a GeoAxesSubplot

        - Also possible to plot fiskeridirektoratet akvakultur lokaliteter WMS on top:
          - ax.add_wms(wms='https://gis.fiskeridir.no/server/services/fiskeridirWMS_akva/MapServer/WMSServer?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities', 
                       layers=['akvakultur_lokaliteter'])
        '''
        import cartopy.crs as ccrs
        if wms is not None:
            if wms =='raster':
                url = 'http://openwms.statkart.no/skwms1/wms.toporaster4?version=1.3.0&service=wms&request=getcapabilities'
                layers = ['toporaster']
            elif wms == 'topo4':
                url = 'https://openwms.statkart.no/skwms1/wms.topo4?service=wms&request=getcapabilities'
                layers = ['topo4_WMS']
            elif wms == 'sjokart':
                url = 'https://wms.geonorge.no/skwms1/wms.sjokartraster2?service=wms&version=1.3.0&request=getcapabilities'
                layers = ['Overseiling']
                depth = False

        if int(self.reference.split(':')[-1]) == 3031:
            lonmin, lonmax = self.lon.min(), self.lon.max()
            latmin, latmax = self.lat.min(), self.lat.max()
            aspect_ratio = (float(latmax - latmin) / (float(lonmax - lonmin)))/np.cos(np.radians((latmin + latmax) / 2))

            # Draw, set projection
            crs = ccrs.Mercator()
            fig = plt.figure(figsize=(11. / aspect_ratio, 11.))
            ax = fig.add_subplot(111, projection=crs)
            self.draw_geo_grid([lonmin, lonmax, latmin, latmax])
            fig.canvas.draw()         
            fig.set_tight_layout(True)
            self.transform = ccrs.PlateCarree(globe=crs.globe)
        else:
            ax = plt.axes(projection = ccrs.epsg(int(self.reference.split(':')[-1])))

        ax.add_wms(wms=url, layers=layers)
        if wms != 'raster':
            if depth:
                ax.add_wms('https://wms.geonorge.no/skwms1/wms.dybdedata2?service=WMS&request=GetCapabilities', layers = ['Dybdedata2'])
        return ax

    def draw_geo_grid(self, extent):
        '''
        Automatically draw a geo-grid over the extent (lonmin, lonmax, latmin, latmax)
        '''
        import cartopy.crs as ccrs
        ax = plt.gca()
        ax.set_extent(extent, crs=ccrs.PlateCarree(globe=ccrs.Mercator().globe))
        gl = ax.gridlines(ccrs.PlateCarree(globe=ccrs.Mercator().globe), draw_labels=True)         
        gl.top_labels = None

    @staticmethod
    def _plot_contour(x, y, tri, field, show, *args, **kwargs):
        ax = plt.gca()
        cont = ax.tricontourf(x, y, tri, field, *args, **kwargs)
        ax.set_aspect('equal')
        if show: plt.show(block=False)
        return cont

    def _plot_section_line(self):
        self.plot_grid(rasterized=True)
        plt.plot(self.x_sec, self.y_sec, 'k', zorder = 10)
        plt.plot(self.x_sec[0], self.y_sec[0], 'g.', zorder = 11, label = 'start')
        plt.plot(self.x_sec[-1], self.y_sec[-1], 'r.', zorder = 11, label = 'stop')
        plt.xlim([self.x_sec.min()-10000, self.x_sec.max()+10000])
        plt.ylim([self.y_sec.min()-10000, self.y_sec.max()+10000])
        plt.legend()
        plt.title('Section')
        plt.savefig('Section_map.png')

class AnglesAndPhysics:
    '''
    Angles you need when rotating velocities from a UTM coordinate system to WGS84 (true north/east)
    '''
    @property
    def node_utm_angle(self):
        '''radians offset from true north in utm coordinates'''
        return self._get_utm_angle(self.x, self.y, self.lon, self.lat)

    @property
    def cell_utm_angle(self):
        '''radians offset from true north in utm coordinates'''
        return self._get_utm_angle(self.xc, self.yc, self.lonc, self.latc)

    def _get_utm_angle(self, x, y, lon, lat, dx = 100):
        '''
        utm coordinate offset of true north measured in radians

        Consider two coordinate systems (x,y), (x', y'). We want to convert vector quantities from one to the other,
        and to do so we find the angle between x and x'. In our case, we say that (x, y) are coordinates in WGS84 and (x', y') are any UTM coordinates.

        For a small displacement dx' in the UTM system, we move a distance dx and dy in the lat/lon system, R is earths radii:
        dx = R * cos(lat) * dLon/dx' * dx'
        dy = R * dLat/dx' * dx'
        angle = atan2(dx, dy)

        Use alpha rotate the coordinate system so that vec_true = (v_x, v_y)
        v_x = v_x'*cos(alpha) - v_y'*sin(alpha)
        v_y = v_x'*sin(alpha) + v_y'*cos(alpha)
        '''
        londx, latdx = self.Proj(x+dx, y, inverse = True)
        dLat_dx = (latdx-lat)/dx # distance made in lat/lon system when incrementing dx
        dLon_dx = ((londx-lon)/dx)*np.cos(lat*(2*np.pi/360)) 
        return np.arctan2(dLat_dx, dLon_dx)

    def rotate_vector_to_utm(self, u, v, angle):
        '''Rotate a vector from lat/lon to utm'''
        u_new =  u*np.cos(angle) + v * np.sin(angle)
        v_new = -u*np.sin(angle) + v * np.cos(angle)
        return u_new, v_new

    def rotate_vector_from_utm(self, u, v, angle):
        '''Rotate a vector to lat/lon from utm'''
        u_new =  u*np.cos(angle) - v * np.sin(angle)
        v_new =  u*np.sin(angle) + v * np.cos(angle)
        return u_new, v_new

    def get_coriolis(self, cell = False):
        '''Returns coriolis parameter at grid points'''
        if cell:
            self.f = np.sin(self.latc*2*np.pi/360)*4*np.pi/(24*60*60)
        else:
            self.f = np.sin(self.lat*2*np.pi/360)*4*np.pi/(24*60*60)

    def get_cfl(self, z = 1.5, u = 3.0, g = 9.81, verbose = True):
        '''
        Estimate the CFD timestep
        '''
        # Length of each triangle side
        lAB, lBC, lCA = self._get_sidewall_lengths()
        dpt      = np.max(self.h[self.tri], axis = 1) + z
        cg_speed = np.sqrt(dpt*g) + u
        ts_AB, ts_BC, ts_CA = np.array(lAB/cg_speed), np.array(lBC/cg_speed), np.array(lCA/cg_speed)
        ts_walls = np.array([ts_AB, ts_BC, ts_CA])
        ts_min   = np.min(ts_walls, axis = 0)/np.sqrt(2) # to adjust for transverse wave propagation

        if verbose:
            print(f'  - Required timestep approx: {min(ts_min):.2f} s')
            plt.figure()
            contour = self.plot_contour(ts_min, cmap = 'jet', levels = np.linspace(min(ts_min), 3*min(ts_min), 20), extend = 'max')
            plt.colorbar(contour, label = 's')
            plt.axis('equal')
            plt.title('Timestep (s)')
            plt.show(block = False)
        self.ts = ts_min

class LegacyFunctions:
    def cell2node(self, fieldin):
        '''Move data from cells to nodes
        - adds its data from each cell to the nodes connected to it. Subsequently averages based on number of cells connected to each node.
        '''
        fieldout = np.zeros(self.node_number)
        count = np.zeros(self.node_number)
        for i in np.arange(self.cell_number):
            n0, n1, n2 = [self.tri[i,j] for j in range(3)]
            for n in [n0, n1, n2]:
                fieldout[n] = fieldout[n] + fieldin[i]
                count[n]+=1
        fieldout = fieldout / count
        return fieldout

class OutputLoader:
    '''
    Methods used to load FVCOM data from netCDF file (will automatically crop to a subgrid)
    '''
    def load_netCDF(self, filepath, field, time, sig = None):
        '''
        Reads a timestep from a field in the netCDF file from "filepath"
        - filename: path to file to be read
        - field:    field you want (e.g. 'tracer')
        - time:     time-index in file to read

        optional:
        - sig:      sigma layer to extract, will automatically return all
        '''
        with Dataset(filepath, 'r') as d:
            # time variables or variables without time can be read directly and returned
            if 'time' not in d[field].dimensions or len(d[field].dimensions) == 1:
                return d[field][:]

            if self.cropped_nodes.any() or self.cropped_cells.any():
                if 'node' in d[field].dimensions:
                    cropped = self.cropped_nodes
                elif 'nele' in d[field].dimensions:
                    cropped = self.cropped_cells
            else:
                cropped = None

            # Automatically crop data
            data = self._load_single_nc_field(d, field, cropped, time, sig)
            
            # Mask data if wetting/drying
            try:
                if 'node' in d[field].dimensions:
                    wet = np.where(self._load_single_nc_field(d, 'wet_nodes', cropped, time, sig)==0)[0]

                elif 'nele' in d[field].dimensions:
                    wet = np.where(self._load_single_nc_field(d, 'wet_cells', cropped, time, sig)==0)[0]

                if len(data.shape)==1:
                    data[wet] = np.nan
                else:
                    data[:, wet] = np.nan
            except:
                pass
        return np.array(data)

    def _load_single_nc_field(self, d, field, cropped, time, sig):
        '''
        Loads data from a netCDF file, crops to mesh if needbe
        '''
        dim = len(d[field].dimensions)
        if self.cropped_nodes.any() or self.cropped_cells.any():
            if dim == 2:
                data = d[field][time, : ][cropped]
            elif dim == 3:
                if sig is None:
                    data = d[field][time, :, :][:, cropped]
                else:
                    data = d[field][time, sig, :][cropped]

        else:
            if dim == 2:
                data = d[field][time, :]

            if dim == 3:
                if sig is None:
                    data = d[field][time, :, :]
                else:
                    data = d[field][time, sig, :]
        return data

class FVCOM_grid(GridLoader, InputCoordinates, Coordinates, PropertiesFromTGE, LegacyPropertyAliases, CellTriangulation, OutputLoader,
                 PlotFVCOM, ControlVolumePlotter, InterpolateToZ, LegacyFunctions, SectionMaker, ExportGrid, CropGrid, AnglesAndPhysics,
                 CoastLine, OBC, GridProximity):
    '''FVCOM grid class compatible with ApN FVCOM workflows, and Akvaplan-branch uk-fvcom.

Attributes:
        Position data
            x, y   (lon, lat)   - node position
            xc, yc (lonc, latc) - cell position
            tri, ctri           - triangulation for node (tri) and cells (ctri)
            siglay, siglay_c    - middle of sigma box (tracers, u and v are computed here)
            siglev, siglev_c    - sigma layer top and bottom interfaces (w is computed here)
            siglayz, sialayz_uv - depth of sigma layer mid-points
            siglevz, siglevz_uv - depth of sigma layer top and bottom interfaces
            h, hc               - bathymetric depth
            d, dc               - total watercolumn depth

        Grid identifiers
            nbe    - ids of cells connected to cells
            nbse   - ids of all cells surrounding cells
            nbsn   - ids to nodes surrounding nodes
            nbve   - ids to cells surrounding nodes
            ntsn   - number of nodes surrounding nodes
            ntve   - number of cells surrounding nodes
            nese   - number of cells surrounding cells

        Grid details:
            grid_res    - array with minimum resolution in each triangle
            grid_angles - array with the angle of each corner in each triangle
            cell_utm_angle - angle betwen lat/lon and utm at cells
            node_utm_angle - angle between lat/lon and utm at nodes

        Grid area
            art1     - area of node-based control volume (CV)
            tri_area - area of triangles

        Grid volume
            node_volume - volume connected to the data on nodes (and sigma layer)
            cell_volume - volume connected to the data at cells (and sigma layer)

        Open boundary identifiers
            nodestrings    - list with nodestrings, all nodes in one distinct obc
            obc_nodes      - all nodes connected to the open boundary (np.array())
            obc_type       - identifier saying which type of OBC condition was used
                             (will only be read from casename_restart_****.nc files)

        Mesh boundary (including land)
            model_boundary - shapely polygon (linestring) bounding the model domain (on both obc and mainland)

Functions:
        Plotting:
            .plot_grid()        - plots the grid
            .plot_cvs(data)     - plots node based data on CV-patches
            .plot_field(data)   - plots node based data over triangles
            .plot_contour(data) - contour plots any data
            .plot_obc()         - plot all OBC nodes, and show which nodestring they belong to
            .georeference()     - add georeference from WMS server. Must be plotted before calling the other plotting routines

        Grid:
            .get_land_polygons() - returns shapely land/islands polygons (the one with biggest area is the exterior boundary polygon)
            .find_nearest()      - returns nearest grid point (either grid = cell or grid = node) to input x,y
            .isinside()          - returns nodes inside a search area
            .is_on_mesh(x,y)     - returns x,y points located on the mesh

        Transect maker
            .prepare_section() - Create a transect by selecting points on a map, will store to casename_section.py

        Physics
            .get_coriolis()   - Computes the coriolis parameter

        Data extraction / manipulation
            .interpolate_to_z(data) - Interpolates input-data to z-depth
            .get_section_data(data) - Interpolates input data along a section
            .smooth(data)           - Smooths input-data

        Mesh cropping
            .subgrid()        - crops the mesh object and returns a smaller object with indexing
                                to the original mesh (usefull to reduce size of the data being handled)

        Export mesh
            .write_2dm()      - return mesh as a .2dm file that can be read to SMS
            .write_grd()      - return a grid .dat file for FVCOM input
            .write_bath()     - return a bathymetry .dat
            .write_sponge()   - return a sponge .dat
            .write_obc()      - return a obc .dat
            .write_cor()      - return a coriolis .dat
            .write_draft_nc() - return a ice draft .dat
            .write_coast()    - returns a .shp file representing land masses as closed polygons
            .write_section()  - writes section points to a file

        Import data
            .load_netCDF(filename) - use netCDF4 to open filename, returns d = Dataset(filename)

Extended features through the mesh-connectivity TGE (.T) class:
    -> This contains functions to compute some heavier stuffs (will take a minute or two to set-up)
        - .T.node_gradient()     - Gradient of data stored on nodes
        - .T.cell_gradient()     - Gradient of data stored on cells
        - .T.vertical_gradient() - Vertical gradient of data
        - .T.vorticity()         - vorticity of depth-averaged currents
        - .T.vorticity_3D()      - vorticity of 3D-currents
        - .T.okubo_weiss()       - okubo weiss parameter (for eddy detection)
    '''
    def __init__(self,
                 pathToFile = None,
                 reference = 'epsg:3031',
                 verbose   = False,
                 static_depth = False,
                 x = None, y = None, 
                 lon = None, lat = None,
                 tri = None,
                 h = None, siglay = None, siglev = None,
                 obc_nodes = [],
                 cropped_nodes = np.array(0), cropped_cells = np.array(0)):
        '''
        pathToFile (if you want to read from a supported file format: 
        - .2dm, .mat, .npy, .nc or .txt

        or direct input of grid metrixs
        - x, y, tri                    - node locations, triangulation
        - lon, lat                     - node locations in spherical coordinates
        - h, siglay, siglev            - vertical grid information (depth, vertical splits)
        - obc_nodes                    - boundary identifiers
        - cropped_nodes, cropped_cells - linking a cropped mesh to a mother mesh

        reference:
        - by default polar sterographic projection

        verbose:
        - set True if you want to get progress reports

        static_depth:
        - used when interpolating sections. The bottom z-level depth in a sigma-layer model is not static, since the water column thickness is not 
          constant over the simulation period. When plotting section movies, however, it is convenient to force it to be stationary, not to confuse 
          people unfamiliar with sigma layer discretizations too much.
        '''
        self.filepath       = pathToFile
        self.reference      = reference
        self.verbose        = verbose
        self._static_depth  = static_depth
        if pathToFile is None:
            self._direct_initialization(x=x, y=y, lon = lon, lat = lat, tri = tri, h = h, siglay = siglay, siglev = siglev, 
                                        obc_nodes=obc_nodes, cropped_nodes=cropped_nodes, cropped_cells=cropped_cells)

        elif pathToFile[-3:]   == 'mat':
            self._add_grid_parameters_mat()

        elif pathToFile[-3:] == 'npy':
            self._add_grid_parameters_npy()

        elif pathToFile[-3:] == '2dm':
            self._add_grid_parameters_2dm()

        elif pathToFile[-3:] == '.nc':
            self._add_grid_parameters_nc()

        elif pathToFile[-3:] == 'txt':
            self._add_grid_parameters_txt()

        else:
            assert False, f'{self.filepath} can not be read by FVCOM_grid' # Add new readers when needbe

        self.Proj = self._get_proj()
        self._project_xy()

    def __repr__(self):
        return f'''{self.casename} FVCOM grid object

Grid read from:        {self.filepath}
Number of nodes:       {len(self.x)}
Number of triangles:   {len(self.xc)}
Grid resolution:       min: {np.min(self.grid_res):.2f} m, max: {np.max(self.grid_res):.2f} m
Grid angles (degrees): min: {np.min(self.grid_angles):.2f}, max: {np.max(self.grid_angles):.2f}
Reference:             {self.reference}'''

    @property
    def casename(self):
        if self.cropped_nodes.any():
            return f'Cropped {self._casename}'
        else:
            return self._casename
    
    @casename.setter
    def casename(self, var):
        self._casename = var

    def smooth(self, field, SmoothFactor = 0.2, n = 5):
        '''
        Smooth node-data using a laplacian filter
        - field:           Field to smooth
        - SmoothFactor, n: The degree of smoothing, Number of times to smooth
        '''
        print(f'Smoothing {n+1} times:')
        assert len(field) == self.node_number or len(field) == self.cell_number, 'field does not match cells or nodes'
        if np.squeeze(field).shape[0] == self.node_number:
            field = self._smooth(field, SmoothFactor, n, self.nbsn, self.ntsn)
        elif np.squeeze(field).shape[0] == self.cell_number:
            field = self._smooth(field, SmoothFactor, n, self.nbse, self.nese)
        return field

    @staticmethod
    @njit
    def _smooth(field, SmoothFactor, n, neighbors, number_of_neighbors):
        '''Smooths node based data'''
        field_smooth = np.copy(field)
        for n in range(n):
            for grid_point in range(field.shape[0]):
                grid_points = neighbors[grid_point, :number_of_neighbors[grid_point]]
                smooth = np.mean(field[grid_points])
                field_smooth[grid_point] = (1-SmoothFactor)*field[grid_point] + SmoothFactor*smooth
            field = np.copy(field_smooth)
        return field

    def _check_if_cell_or_node(self, data):
        '''check if some data field is stored on nodes or cells'''
        if data.shape[0] == self.node_number:
            grid = ''
        elif data.shape[0] == self.cell_number:
            grid = 'c'
        else:
            raise InputError('Your data somehow does not match with node or cell dimensions. Do the grid and data dimensions match?')
        return grid

    def _check_if_siglay_or_siglev(self, data):
        '''check if some data field is vertically spaced on siglay or siglev points'''
        sigma = 'siglev' # by default (to get z = 0 at surface, z = -h at bottom, valid for horizontal data)
        if len(data.shape) == 2:
            if data.shape[1] != self.siglev.shape[1] and data.shape[1] != self.siglay.shape[1]:
                data = data.T
            if data.shape[1] == self.siglay.shape[1]:
                sigma = 'siglay'
            elif data.shape[1] == self.siglev.shape[1]:
                sigma = 'siglev'
            elif data.shape[0] == 1:
                sigma = 'siglev'
            else:
                raise InputError('Your data somehow neither match siglev or siglay dimensions.')
        elif len(data.shape) == 3:
            raise InputError('input field can only be provided in 2D arrays or 1D arrays, time must be looped over externally.')
        return sigma

class LegacyNestPropertyAliases:
    @property
    def lonn(self):
        return self.lon[:, None]

    @property
    def latn(self):
        return self.lat[:, None]

class LoadNest:
    def add_grid_parameters_mat(self, names):
        '''Read grid attributes from mfile and add them to FVCOM_grid object'''
        from scipy.io import loadmat
        grid_mfile = loadmat(self.filepath)
        if type(names) is str:
            names=[names]

        for name in names:
            setattr(self, name, grid_mfile['ngrd'][0,0][name])

        # Translate Matlab indexing to python
        self.nid = self.nid -1
        self.cid = self.cid-1
        self.nv  = self.nv-1
        self.R   = self.R[0][0]

    def _nest_parameters_npy(self, names, add=None):
        '''add means "add dimension"'''
        nest = np.load(self.filepath, allow_pickle=True).item()
        for key in names:
            try:
                if add:
                    setattr(self, key, nest[key][:, None])
                else:
                    setattr(self, key, nest[key])
            except:
                continue

class NestROMS2FVCOM:
    def calcWeights(self, M, w1=2.5e-4, w2=2.5e-5):
        '''
        Calculates linear weights in the nesting zone from weight = w1 at the obc to
        w2 the inner end of the nesting zone. At the obc nodes, weights equals 1

        By default (matlab legacy):
        w1  = 2.5e-4
        w2  = 2.5e-5
        '''
        self.x_obc, self.y_obc = np.copy(M.x_obc), np.copy(M.y_obc)
        self.crop_nest_grid_corners()
        nest_width = self.find_max_width()
        self._compute_weights(nest_width, w1, w2)
        self.get_obc_nodes(M)
        self.get_cube_connected_to_obc()
        self.set_weights_in_obc_cube_to_one()

    def crop_nest_grid_corners(self):
        '''
        Crop the nest grid and remove parts of its corners (as suggested by Ole Anders and Qin, since it helps conserve mass near the boundary)
        '''
        # Find the max radius- and node distance vector
        if self.oend1 == 1:
            new_x_obc, new_y_obc = [], []
            for n in range(self.x_obc.shape[0]):
                dist       = np.sqrt((self.x_obc[n] - self.x_obc[n][0])**2 + (self.y_obc[n] - self.y_obc[n][0])**2)
                i          = np.where(dist>self.R)
                new_x_obc.append(self.x_obc[n][i])
                new_y_obc.append(self.y_obc[n][i])
            self.x_obc, self.y_obc = np.array(new_x_obc), np.array(new_y_obc)

        if self.oend2 == 1:
            new_x_obc, new_y_obc = [], []
            for n in range(self.x_obc.shape[0]):
                dist = np.sqrt((self.x_obc[n] - self.x_obc[n][-1])**2 + (self.y_obc[n] - self.y_obc[n][-1])**2)
                i    = np.where(dist>self.R)
                new_x_obc.append(self.x_obc[n][i])
                new_y_obc.append(self.y_obc[n][i])
            self.x_obc, self.y_obc = np.array(new_x_obc), np.array(new_y_obc)

    def find_max_width(self):
        '''Find the distance between the obc and outer perimiter of the nestingzone'''
        self._xo = []; self._yo = []
        for n in range(self.x_obc.shape[0]):
            self._xo.extend(self.x_obc[n])
            self._yo.extend(self.y_obc[n])

        self.d_node = []
        for n in range(len(self.xn)):
            self.d_node.append(np.min(np.sqrt((self._xo-self.xn[n])**2+(self._yo-self.yn[n])**2)))
        return max(self.d_node)

    def _compute_weights(self, nest_width, w1, w2):
        distance_range = [0, nest_width]
        weight_range   = [w1, w2]
        d_cell = []
        for n in range(len(self.xc)):
            d_cell.append(min(np.sqrt((self._xo-self.xc[n])**2+(self._yo-self.yc[n])**2)))
        self.weight_node = np.interp(self.d_node, distance_range, weight_range)
        self.weight_cell = np.interp(d_cell, distance_range, weight_range)
        if np.argwhere(self.weight_node<0).size != 0: self.weight_node[np.where(self.weight_node)]=min(weight_range)
        if np.argwhere(self.weight_cell<0).size != 0: self.weight_cell[np.where(self.weight_cell)]=min(weight_range)

    def get_obc_nodes(self, M):
        self.node_obc_in_nest = [];
        for x, y in zip(M.x[M.obc_nodes], M.y[M.obc_nodes]):
            dst_node = np.sqrt((self.xn-x)**2+(self.yn-y)**2)
            self.node_obc_in_nest.append(np.where(dst_node==dst_node.min())[0][0])
        self.node_obc_in_nest = np.array(self.node_obc_in_nest).astype(int)

    def get_cube_connected_to_obc(self):
        obc_here = np.zeros((len(self.xn[:,0],)))
        obc_here[self.node_obc_in_nest] = 1
        self.cell_obc_to_one = np.where(np.sum(obc_here[self.nv], axis=1)>0)[0]
        self.node_obc_to_one = np.unique(self.nv[self.cell_obc_to_one])

    def set_weights_in_obc_cube_to_one(self):
        self.weight_node[self.node_obc_to_one] = 1.0
        self.weight_cell[self.cell_obc_to_one] = 1.0

class NEST_grid(LoadNest, NestROMS2FVCOM, Coordinates, PlotFVCOM, AnglesAndPhysics, LegacyPropertyAliases, LegacyNestPropertyAliases):
    '''
    Object containing information about the nestingzone grid
    '''
    def __init__(self, path_to_nest, M=None, proj='epsg:3031'):
        """
        Reads ngrd.* file
        """
        self.filepath = path_to_nest

        if self.filepath[-3:] == 'mat':
            self.add_grid_parameters_mat(['xn', 'yn', 'h', 'nv', 'fvangle', 'R', 'nid', 'cid', 'oend1', 'oend2'])
            self.nv = self.nv-1 # to python indexing

        elif self.filepath[-3:] == 'npy':
            self._nest_parameters_npy(['xn','yn'], add = True)
            self._nest_parameters_npy(['oend1', 'oend2', 'R'], add = False)
            self._nest_parameters_npy(['nid', 'cid'], add = False)
            self._nest_parameters_npy(['h_mother', 'hc_mother', 'siglev_mother', 'siglay_mother',
                                       'siglev_center_mother', 'siglay_center_mother', 'siglayz_mother',
                                       'siglayz_uv_mother'], add = False)
            self._nest_parameters_npy(['h', 'siglev', 'siglay', 'nv'], add = False)

        # Add information from full fvcom grid (M), vertical coords and OBC-nodes
        if M is not None:
            self.Proj   = M.Proj
            self.siglay = M.siglay[:len(self.x), :]
            self.siglev = M.siglev[:len(self.x), :]
        else:
            try:
                self.Proj = Proj(self.info['reference'])
            except:
                self.Proj = Proj(proj)

        # Get lat and lon
        self.lon, self.lat   = self.Proj(self.x, self.y, inverse=True)
        self.lonc, self.latc = self.Proj(self.xc, self.yc, inverse=True)

    # Basic property names to comply with the subclasses we use to derive properties in M    
    @property
    def x(self):
        return np.array(self.xn[:,0])

    @property
    def y(self):
        return np.array(self.yn[:,0])

    @property
    def h(self):
        '''Average water column thickness'''
        return self._h
    
    @property
    def zeta(self):
        '''Sea surface elevation at nest timestep, needed in the more generic vertical interpolation coefficients routine. Not used actively by NEST_grid'''
        return np.zeros(self.h.shape)

    @h.setter 
    def h(self, var):
        self._h = var

    @property
    def tri(self):
        return np.array(self.nv)

    @property
    def nid(self):
        '''Node id in mother model for each nesting node'''
        assert hasattr(self, '_nid'), 'nid is missing'
        return np.squeeze(self._nid)

    @nid.setter
    def nid(self, var):
        self._nid = var    

    @property
    def cid(self):
        '''Cell id in mother model for each nesting cell'''
        assert hasattr(self, '_cid'), 'cid is missing'
        return np.squeeze(self._cid)

    @cid.setter
    def cid(self, var):
        self._cid = var

class InputError(Exception):
    pass
