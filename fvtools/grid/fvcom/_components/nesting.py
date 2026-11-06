"""ROMS-to-FVCOM nesting grid and associated mesh helpers.

Class implementations extracted verbatim from fvcom_grd.py.
"""

import numpy as np
from pyproj import Proj
from scipy.spatial import cKDTree as KDTree

from fvtools.grid.fvcom._components.coordinates import Coordinates
from fvtools.grid.fvcom._components.proximity import GridProximity
from fvtools.grid.fvcom._components.plotting import PlotFVCOM
from fvtools.grid.fvcom._components.physics import AnglesAndPhysics
from fvtools.grid.fvcom._components.compatibility import LegacyPropertyAliases


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

        if 'info' in nest.keys():
            self.info = nest['info']
            try:
                self.reference = self.info['reference']
            except:
                pass
        else:
            self.casename = 'FVCOM experiment'


class NestROMS2FVCOM:
    def calcWeights(self, M, w1=2.5e-4, w2=2.5e-5):
        '''
        Calculates linear weights in the nesting zone from weight = w1 at the obc to
        w2 the inner end of the nesting zone. At the obc nodes, weights equals 1

        By default (matlab legacy):
        w1  = 2.5e-4
        w2  = 2.5e-5
        '''
        self.x_obc, self.y_obc = np.concat(M.x_obc), np.concat(M.y_obc)
        dst_node = self._distance_to_nearest_obc(self.xn[:,0], self.yn[:,0])
        dst_cell = self._distance_to_nearest_obc(self.xc, self.yc)
        self._compute_weights(w1, w2, dst_node, dst_cell)
        self.get_obc_nodes(M)
        self.get_cube_connected_to_obc()
        self.set_weights_in_obc_cube_to_one()

    def _distance_to_nearest_obc(self, x, y):
        '''
        Compute the distance to the nearest obc node
        '''
        obc_tree = KDTree(np.array([self.x_obc, self.y_obc]).T)
        dst, _ = obc_tree.query(np.array([x, y]).T)
        return dst

    def _compute_weights(self, w1, w2, dst_node, dst_cell):
        distance_range = [0, dst_node.max()]
        weight_range   = [w1, w2]

        self.weight_node = np.interp(dst_node, distance_range, weight_range)
        self.weight_cell = np.interp(dst_cell, distance_range, weight_range)
        if np.argwhere(self.weight_node<0).size != 0: 
            self.weight_node[self.weight_node < 0] = min(weight_range)
        if np.argwhere(self.weight_cell<0).size != 0: 
            self.weight_cell[self.weight_cell < 0] = min(weight_range)

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


class NEST_grid(LoadNest, GridProximity, NestROMS2FVCOM, Coordinates, PlotFVCOM, AnglesAndPhysics, LegacyPropertyAliases, LegacyNestPropertyAliases):
    '''
    Object containing information about the nestingzone grid
    '''
    def __init__(self, path_to_nest, M=None, proj='epsg:32633'):
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
            self._nest_parameters_npy(['h', 'nv'], add = False)

        # Add information from full fvcom grid (M), vertical coords and OBC-nodes
        if M is not None:
            self.Proj   = M.Proj

        else:
            try:
                self.Proj = Proj(self.info['reference'])
            except:
                self.Proj = Proj(proj)

        # Get lat and lon
        self.lon, self.lat   = self.Proj(self.x, self.y, inverse=True)
        self.lonc, self.latc = self.Proj(self.xc, self.yc, inverse=True)

        # Inherit sigma coordinates
        if M is not None:
            self.inherit_sig(M)

    def inherit_sig(self, M):
        '''
        Inherit the vertical layers from the model grid
        '''
        nodes = M.find_nearest(self.x, self.y, grid = 'node')
        self.siglay = M.siglay[nodes, :]
        self.siglev = M.siglev[nodes, :]

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
    def siglay_center(self):
        return np.mean(self.siglay[self.tri,:], axis = 1)
    
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
