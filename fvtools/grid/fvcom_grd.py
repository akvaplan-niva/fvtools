import os
import fvtools.grid.tge as tge
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb
import geopandas as gpd
import shapely as shp
import networkx as nx
import cartopy.crs as ccrs
import cmocean as cmo

from pyproj import Proj, Transformer
from netCDF4 import Dataset
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as mPolygon
from scipy.spatial import cKDTree as KDTree
from numba import njit
from functools import cached_property

import asyncio
import ipywidgets as widgets
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")

from fvtools.grid.depth import setup_sigma
from fvtools.grid.netcdf_io import FVCOMNetCDFIO

class GridLoader:
    '''
    Loads relevant grid data to be used as instance attributes
    '''
    def _direct_initialization(self, x = None, y = None, tri = None,
                               lon = None, lat = None,
                               h = None, siglay = None, siglev = None,
                               obc_nodes = None, cropped_nodes = np.array(0), cropped_cells = np.array(0)):
        '''
        Initialize a new FVCOM_grid object using known fields
        '''
        # Assertions to make sure that input is consistent, regardless of source
        assert tri.min() == 0, 'Triangulation is invalid'
        if obc_nodes is None:
            obc_nodes = []
        assert isinstance(obc_nodes, list), 'obc_nodes must be given as a list'
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
            self.lon, self.lat = np.zeros(self.x.shape), np.zeros(self.y.shape)
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

        # Add nodestrings
        nodestrings = []
        for ns in types:
            nodestrings.append(np.array(ns))
        self.nodestrings = nodestrings
        self.casename = self.filepath.split('.2dm')[0]

    def _add_grid_parameters_mat(self):
        '''
        Read grid attributes from matlab grid file and add them to the FVCOM_grid instance
        - Probably won't work that well anymore
        '''
        from scipy.io import loadmat
        grid_mfile = loadmat(self.filepath)
        self.casename = 'from_mat_file'
        self._direct_initialization(
            x = grid_mfile['Mobj']['x'][0][0][:,0], 
            y = grid_mfile['Mobj']['y'][0][0][:,0],
            tri = grid_mfile['Mobj']['tri'][0][0][:]-1, 
            h = grid_mfile['Mobj']['h'][0][0][:,0],
            siglay = grid_mfile['Mobj']['siglay'][0][0], 
            siglev = grid_mfile['Mobj']['siglev'][0][0], 
            obc_nodes = list(grid_mfile['Mobj']['obc_nodes'][0][0][0]-1),
            lon = grid_mfile['Mobj']['lon'][0][0][:,0],
            lat = grid_mfile['Mobj']['lat'][0][0][:,0]
            )

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

    def read_sigma(self, sigmafile=None, **sigma_kwargs):
        """
        Read FVCOM sigma settings and calculate nodal sigma coordinates.

        Parameters
        ----------
        sigmafile : str or pathlib.Path, optional
            Path to a *_sigma.dat file.
        **sigma_kwargs
            Optional overrides passed to depth.setup_sigma.

        Notes
        -----
        KB is the number of vertical interfaces.
        KBM1 = KB - 1 is the number of layers.
        Both siglev and siglay run from surface to bottom.
        """
        if sigmafile is None:
            sigmafile = f"input/{self.casename}_sigma.dat"

        kb, stype, parameters = self._read_sigma_configuration(sigmafile)
        options = self._sigma_options(stype, parameters)
        options.update(sigma_kwargs)

        # Only Souza and generalized coordinates depend on bathymetry.
        depth = getattr(self, "_h", None)

        if depth is None:
            if stype in ("souza", "generalized"):
                raise ValueError(f"{stype.upper()} sigma coordinates require nodal bathymetry (h)")
            depth = np.ones(len(self.x))

        depth = np.atleast_1d(np.asarray(depth))

        if depth.size != len(self.x):
            raise ValueError("The number of depth values must match the number of grid nodes")

        # setup_sigma expects KBM1, not KB.
        result = setup_sigma(depth=depth, n_layers=kb - 1, stype=stype, **options)

        self.siglev = result["z"]
        self.siglay = result["zz"][:, :kb - 1]

    @staticmethod
    def _read_sigma_configuration(sigmafile):
        """
        Parse an FVCOM sigma configuration file.
        """
        import re

        parameters = {}
        values = []

        with open(sigmafile, encoding="utf-8") as file:
            for line in file:
                line = line.split("!", 1)[0].split("#", 1)[0].strip()

                if not line or "=" not in line:
                    continue

                name, value = line.split("=", 1)
                name = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
                value = value.strip().strip("\"'")

                parameters[name] = value
                values.append(value)

        if len(values) < 2:
            raise ValueError(f"Invalid FVCOM sigma file: {sigmafile}")

        # Support positional parameters from older sigma files.
        parameters["_ORDERED_VALUES"] = values

        kb_value = parameters.get("NUMBER_OF_SIGMA_LEVELS", parameters.get("KB", values[0]))

        stype = parameters.get("SIGMA_COORDINATE_TYPE", parameters.get("STYPE", values[1])).strip().lower()

        try:
            kb = int(kb_value)
        except ValueError as exc:
            raise ValueError(f"Invalid number of sigma levels (KB): {kb_value}") from exc

        if kb < 3:
            raise ValueError("KB must be at least 3")

        return kb, stype, parameters

    @staticmethod
    def _sigma_options(stype, parameters):
        """Translate FVCOM sigma parameters to setup_sigma arguments."""

        def value(*names, position=None, required=True, default=None):
            for name in names:
                if name in parameters:
                    return float(parameters[name])

            values = parameters.get("_ORDERED_VALUES", [])

            if position is not None and len(values) > position:
                return float(values[position])

            if required:
                raise ValueError(f"{stype.upper()} sigma requires one of: {', '.join(names)}")

            return default

        if stype == "uniform":
            return {"p_sigma": value("P_SIGMA", position=2, required=False, default=1.0)}

        if stype == "geometric":
            return {"p_sigma": value("P_SIGMA", position=2)}

        if stype == "tanh":
            return {"du2": value("DU", "DU2", position=2), "dl2": value("DL", "DL2", position=3)}

        if stype == "souza":
            return {
                "tcline": value("TCLINE", "TCLIN", position=2),
                "theta_s": value("THETA_S", "THETAS", position=3),
                "theta_b": value("THETA_B", "THETAB", position=4)
            }

        if stype == "generalized":
            options = {}

            for field in ("hmin1", "duu", "dll", "ku", "kl"):
                if field.upper() in parameters:
                    options[field] = value(field.upper())

            for field in ("ku", "kl"):
                if field in options:
                    options[field] = int(options[field])

            for field in ("zku", "zkl"):
                if field.upper() in parameters:
                    raw = parameters[field.upper()].replace(",", " ")
                    options[field] = [
                        float(x) for x in raw.split()
                    ]

            return options

        raise ValueError(f"Unsupported sigma coordinate type: {stype!r}")

from fvtools.grid._components.coordinates import InputCoordinates, Coordinates

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

    @cached_property
    def node_tree(self):
        """Lazily build and cache the node KDTree until coordinates change."""
        return KDTree(np.column_stack((self.x, self.y)))

    @cached_property
    def cell_tree(self):
        """Lazily build and cache the cell KDTree until the mesh changes."""
        return KDTree(np.column_stack((self.xc, self.yc)))

    def find_within(self, x, y, r=None, grid='node'):
        """Unique grid indices within radius `r` of one or more xy points.

        Scalar x/y and equal-length coordinate arrays are both supported.
        """
        if grid not in ('node', 'cell'):
            raise ValueError("grid must be 'node' or 'cell'")
        if r is None or not np.isfinite(r) or r < 0:
            raise ValueError('r must be a finite, non-negative radius')
        x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        points = np.column_stack((x.ravel(), y.ravel()))
        hits = getattr(self, f'{grid}_tree').query_ball_point(points, r)
        return np.asarray(sorted(set().union(*hits)), dtype=int)

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
            print('- Computing triangle, grid, elements (TGE) - this may take some minutes for large grids')
            self._T = tge.main(self, verbose = False)
            self._nbsn = self._T.NBSN
            self._ntsn = self._T.NTSN
            self._nbve = self._T.NBVE
            self._ntve = self._T.NTVE
            self._nbe = self._T.NBE
            self._ISBCE = self._T.ISBCE
            self._ISONB = self._T.ISONB
            self._nbse, self._nese = tge.get_NBSE_NESE(self.nbve, self.ntve, self.tri, len(self.xc))
        return self._T

    @property
    def ISBCE(self):
        '''
        "is boundary control element"
        = 0 in the interior computational domain
        = 1 on the solid boundary
        = 2 on the open boundary
        = 3 element with 2 solid boundary edges
        '''
        if not hasattr(self, '_ISBCE'):
            # Calculate it yourself otherwise
            _T = tge.calculate_nbe_and_boundaries(self, verbose = False)
            self._ISBCE = _T.ISBCE
        return self._ISBCE

    @ISBCE.setter
    def ISBCE(self, var):
        self._ISBCE = var

    @property
    def ISONB(self):
        '''
        "is on boundary"
        = 0 in the interior computational domain
        = 1 on the solid boundary
        = 2 on the open boundary
        '''
        if not hasattr(self, '_ISONB'):
            # Calculate it yourself otherwise
            _T = tge.calculate_nbe_and_boundaries(self, verbose = False)
            self._ISONB = _T.ISONB
        return self._ISONB

    @ISONB.setter
    def ISONB(self, var):
        self._ISONB = var

    @property
    def nbe(self):
        '''indices of elements sharing a wall with *this* element'''
        if not hasattr(self, '_nbe'):
            # Calculate it yourself otherwise
            _T = tge.calculate_nbe_and_boundaries(self, verbose = False)
            self._nbe = _T.NBE
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
        # A replacement of the cell connectivity invalidates the cached
        # Matplotlib triangulation (the nodal triangulation is unchanged).
        self.__dict__.pop('mtri_cells', None)

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

        coastline_bool = np.zeros(self.x.shape, dtype = bool)
        coastline_bool[self.coastline_nodes] = True
        new_self._update_coastline_nodes(coastline_bool[nodes])
        new_self._update_obc()
        new_self._project_xy()
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
        """Return indices of triangles without exactly two exterior edges.

        This preserves the previous filtering rule while avoiding the
        quadratic membership scan over all discarded triangles.
        """
        neighbors = matplotlib.tri.Triangulation(x, y, tri).neighbors
        remove = np.count_nonzero(neighbors == -1, axis=1) == 2
        return np.flatnonzero(~remove)

    def _update_obc(self):
        '''
        Update local land nodes, and local obc_nodes.
        '''
        if len(self.coastline_nodes) > 0:
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
        coastline_nodes = self.full_model_boundary[obc_bool] # Be aware: Here we loose the points connecting to the nodestring / obc

        # therefore: Add the points connecting of the nodestrings to the coastline
        for nodestring in self.nodestrings:
            coastline_nodes = np.append(coastline_nodes, nodestring[[0,-1]])
        return coastline_nodes

    @property
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
    def boundary_nodes(self):
        return self._get_land_segments() 

    @property
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
        polygons = list(polygonize(
            (n1, n2) for n1, n2 in zip(np.array([self.x[self.boundary_nodes][:,0], self.y[self.boundary_nodes][:,0]]).T, 
                                       np.array([self.x[self.boundary_nodes][:,1], self.y[self.boundary_nodes][:,1]]).T)
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
    def n_obc(self):
        '''
        Keeps track of number of open boundaries
        '''
        return len(self.nodestrings)
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
        """
        Ordered open boundary node sequences.
        """
        if not hasattr(self, "_nodestrings"):
            self._nodestrings = (
                self._get_nodestrings(self.obc_nodes)
                if len(self.obc_nodes) > 0
                else []
            )
        return self._nodestrings

    @nodestrings.setter
    def nodestrings(self, var):
        self._nodestrings = var

    @cached_property
    def cellstrings(self):
        _cellstrings = []
        for i in range(len(self.nodestrings)):
            cells = []
            for j in range(self.nodestrings[i].shape[0]-1):
                try:
                    cells.append(int(
                        np.where(
                            np.logical_and(
                                (self.tri == self.nodestrings[i][j]).max(axis=1), 
                                (self.tri == self.nodestrings[i][j+1]).max(axis=1)
                            )
                        )[0][0])
                    )
                except:
                    # This will be single cells that do not face the boundary, we do not need to track them
                    pass
            _cellstrings.append(cells)
        return _cellstrings


    @property
    def x_obc(self):
        '''x-position of nodes at the open boundary'''
        return np.array([self.x[nodestring] for nodestring in self.nodestrings], dtype = object)

    @property
    def y_obc(self):
        '''y-position of nodes at the open boundary'''
        return np.array([self.y[nodestring] for nodestring in self.nodestrings], dtype = object)
    
    @property
    def lat_obc(self):
        '''latitude of nodes on the open boundary'''
        return np.array([self.lat[nodestring] for nodestring in self.nodestrings], dtype = object)

    @property
    def lon_obc(self):
        '''longitude of nodes on the open boundary'''
        return np.array([self.lon[nodestring] for nodestring in self.nodestrings], dtype = object)

    def _get_nodestrings(self, obc_nodes):
        """
        Identify and order open boundary nodestrings.

        Only edges belonging to the outer boundary of the
        triangulation are considered. Each nodestring is ordered
        by following the actual mesh connectivity.

        Parameters
        ----------
        obc_nodes : array-like
            Zero-based node indices belonging to open boundaries.

        Returns
        -------
        nodestrings : list of np.ndarray
            One ordered array per connected open boundary.

        Raises
        ------
        ValueError
            If a node is not connected to an OBC boundary edge,
            or if the boundary contains branches or closed loops.
        """
        obc_nodes = np.unique(np.asarray(obc_nodes, dtype=np.int64))

        if obc_nodes.size == 0:
            return []

        if np.any(obc_nodes < 0) or np.any(obc_nodes >= self.node_number):
            raise ValueError("Invalid OBC node indices")

        # Identify nodes belonging to the open boundary.
        is_obc = np.zeros(self.node_number, dtype=bool)
        is_obc[obc_nodes] = True

        # Only consider edges on the actual mesh boundary.
        boundary_edges = self.boundary_nodes
        obc_edges = boundary_edges[np.all(is_obc[boundary_edges], axis=1)]

        # Verify that every OBC node belongs to a selected edge.
        missing = np.setdiff1d(obc_nodes, np.unique(obc_edges))

        if missing.size:
            raise ValueError(f"OBC nodes without a connected boundary edge: {missing.tolist()}")

        # Create a graph containing only OBC boundary edges.
        G = nx.Graph()
        G.add_edges_from(map(tuple, obc_edges))
        nodestrings = []

        # Each connected component represents a separate boundary.
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            degrees = dict(subgraph.degree())

            # An unbranched boundary cannot have more than two neighbours at any node.
            branches = [node for node, degree in degrees.items() if degree > 2]

            if branches:
                raise ValueError(f"Branched OBC at nodes {branches}")

            # An open boundary has exactly two endpoints.
            endpoints = sorted(node for node, degree in degrees.items() if degree == 1)

            if len(endpoints) != 2:
                raise ValueError("Expected an open boundary with two endpoints. Closed OBC loops are not supported.")

            # Follow the mesh connectivity from one endpoint
            # to the other, preserving the actual node order.
            path = nx.shortest_path(subgraph, source=endpoints[0], target=endpoints[1])
            nodestrings.append(np.asarray(path, dtype=np.int64))

        # Sort boundary segments deterministically.
        nodestrings.sort(key=lambda path: int(path[0]))

        return nodestrings

class ControlVolumePlotter:
    @property
    def cv(self):
        '''
        A list of patch collections for the node based control volumes
        '''
        if not hasattr(self, '_cv'):
            self._cv = self._get_cvs()
        return self._cv

    def plot_cvs(self, 
                 field,
                 cmap = 'jet', 
                 cmax = None, 
                 cmin = None, 
                 Norm = None, 
                 edgecolor = 'face', 
                 inds = None,
                 **kwargs):
        '''
        Plots nodal fields on control volume patches

        Optional:
        ----
        - edgecolor: None by default. Set a color, e.g. "k" to draw interfaces between CVs
        - cmap:      colormap
        - cmax:      cap colors to cmax
        - cmin:      floor colors to cmin
        - Norm:      to create non-linear colorbars (ref matplotlib documentation)
        --> for cmax and cmin, the routine exclusively plots CVs with values within in that range

        keyword arguments are sent to matplotlib.collections.PatchCollection
        '''
        if inds == None:
            inds = np.arange(0, len(field), dtype = int)

        # mask cvs with values below/above threshold
        if cmax is not None and cmin is None:
            inds = np.where(field<=cmax)[0]
        elif cmin is not None and cmax is None: 
            inds = np.where(field>=cmin)[0]
        elif cmax is not None and cmin is not None:
            inds = np.intersect1d(np.where(field<=cmax)[0], np.where(field>=cmin)[0])

        if len(inds) == len(self.cv): 
            collection = PatchCollection(self.cv, cmap=cmap, edgecolor=edgecolor, norm=Norm, **kwargs)
        else: 
            collection = PatchCollection(self.cv[inds], cmap=cmap, edgecolor=edgecolor, norm=Norm, **kwargs)
        collection.set_array(field[inds])
        ax   = plt.gca()
        _ret = ax.add_collection(collection)
        ax.autoscale_view(True)
        ax.set_aspect('equal')
        if _ret._A is not None: 
            plt.gca()._sci(_ret)
        return _ret

    def _get_cvs(self, nodes = None):
        '''
        - computes matplotlib-patches for each control voulme, adds to self as self.cvs
        - Returns numpy file (cvs.npy) containing all of the control volume patches
        '''
        if nodes is None: 
            nodes = np.arange(0, self.node_number)
        xcv_full, ycv_full = self._get_control_volumes(self.nbsn, self.nbve, self.x, self.y, self.xc, self.yc, nodes)
        bar    = pb.ProgressBar(widgets = ['Making control volume patches: ', pb.Percentage(), ' ', pb.BouncingBar(), pb.AdaptiveETA()], maxval=len(nodes))
        bar.start()
        cv = []
        for i, (xcv, ycv) in enumerate(zip(xcv_full, ycv_full)):
            bar.update(i)
            cv.append(mPolygon(np.array([xcv, ycv]).transpose(), closed = True))
        bar.finish()
        return cv
        #np.save('cvs.npy', self.cv) # for instant access to cvs at a later date

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

    def interpolate_to_z(self, f, z, depths=None):
        """Interpolate (sigma, location) values to z metres below surface.

        Nearest-layer values are returned above the shallowest sigma point;
        locations below the deepest valid sigma point return NaN. Exact
        matches to a sigma depth are returned without interpolation.
        """
        f = self._as_array(f) if hasattr(self, '_as_array') else np.ma.filled(
            np.ma.asarray(f, dtype=float), np.nan)
        f = np.asarray(f, dtype=float)
        if depths is None:
            candidates = ('siglayz', 'siglevz', 'siglayz_uv', 'siglevz_uv')
            depths = next(( -np.asarray(getattr(self, attr)).T
                            for attr in candidates
                            if f.shape == getattr(self, attr).T.shape), None)
            if depths is None:
                raise ValueError('Field shape does not match sigma-depth coordinates')
        depths = np.asarray(depths, dtype=float)
        if f.ndim != 2 or f.shape != depths.shape:
            raise ValueError('f and depths must have shape (sigma, location)')
        if not np.isscalar(z) or not np.isfinite(z) or z < 0:
            raise ValueError('z must be a nonnegative, finite depth')

        valid = np.isfinite(f) & np.isfinite(depths)
        above = np.where(valid & (depths <= z), depths, -np.inf)
        below = np.where(valid & (depths >= z), depths, np.inf)
        upper_index = np.argmax(above, axis=0)
        lower_index = np.argmin(below, axis=0)
        loc = np.arange(f.shape[1])
        upper_depth, lower_depth = above[upper_index, loc], below[lower_index, loc]
        upper_value = f[upper_index, loc]
        lower_value = f[lower_index, loc]

        out = np.full(f.shape[1], np.nan, dtype=float)
        exact = np.isfinite(upper_depth) & np.isfinite(lower_depth) & (upper_depth == lower_depth)
        out[exact] = upper_value[exact]
        interpolate = (np.isfinite(upper_depth) & np.isfinite(lower_depth)
                       & (lower_depth > upper_depth))
        out[interpolate] = (
            upper_value[interpolate] * (lower_depth[interpolate] - z)
            + lower_value[interpolate] * (z - upper_depth[interpolate])
        ) / (lower_depth[interpolate] - upper_depth[interpolate])

        # Shallower than topmost sigma point: reuse the shallowest value.
        shallower = (~np.isfinite(upper_depth)) & np.isfinite(lower_depth)
        out[shallower] = lower_value[shallower]
        return out

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
        """
        Calculate physical vertical coordinates along the transect.
        Supports node- and cell-based data, sigma layers andsigma levels, including horizontally varying sigma distributions.

        Returns
        -------
        np.ndarray
            Vertical coordinates in meters, with shape (n_section_points, n_vertical_levels).
            Coordinates are relative to mean sea level, positive upwards.
        """
        # Select horizontal grid and vertical coordinates
        if self._transect_grid == "":
            sigma = getattr(self, self._transect_sigma)
            h = self.h
            zeta = self.zeta

        elif self._transect_grid == "c":
            sigma = getattr(self, f"{self._transect_sigma}_c")
            h = self.hc
            zeta = np.mean(self.zeta[self.tri], axis=1)

        else:
            raise ValueError("Invalid transect grid type")

        # Physical vertical coordinates at every grid point
        z_grid = (zeta[:, None] + (h + zeta)[:, None] * sigma)

        # Visualization adjustment:
        # - Keep the deepest sigma-layer midpoint stationary when the sea surface elevation changes.
        z_grid -= zeta[:, None] * (1.0 + sigma[:, [-1]])

        # - Stick the upper level to the sea surface, not to the uppermost siglay point
        z_grid[:, 0] = zeta

        # Interpolate each vertical level to the transect
        return self._interpolate_3D(z_grid)


    @property
    def _min_sigma(self):
        '''bottom sigma level value. included since I want to keep bottom depth constant in section movies (which is a lie...)'''
        _min_sigma = -1
        if self._static_depth: _min_sigma = self.siglay[-1, -1]
        return _min_sigma

    def prepare_section(self, section_file=None, res=None, store_transect_img=False, section_points=None):
        """
        Prepare coordinates along a transect for section analysis.

        The transect can be defined using a section file, interactively,
        or by providing section points directly.
        """
        if section_points is not None:
            x_section, y_section = section_points

        elif section_file is not None:
            x_section, y_section = self._load_section_file(section_file)

        else:
            x_section, y_section = self._graphical_section()

        if res is None:
            res = np.sum(
                np.hypot(np.diff(x_section), np.diff(y_section))
            ) / 60

        self._assemble_section(x_section, y_section, res)

        if section_file is None:
            self.write_section()

        self.__dict__.pop('trs', None)
        self.__dict__.pop('_section_weights_cache', None)
        self.__dict__.pop('_section_vector_basis_cache', None)

        if store_transect_img:
            self._plot_section_line()

        # Do not close the interactive figure in JupyterLab
        if section_points is None:
            plt.close('all')

    # Rotuines that define where the section goes (ie. defining nodes on the section line)
    def _graphical_section(self):
        '''Define a transect by plotting on a map'''
        c = self.plot_contour(self.h, levels = np.linspace(self.h.min(), self.h.max(), 30))
        self.plot_grid(rasterized=True)
        plt.title('Click where you want the section to go (minimum two points). Hit "Enter" to continue.')
        places = plt.ginput(timeout = -1, n = -1)
        return np.array([p[0] for p in places]), np.array([p[1] for p in places])

    async def prepare_section_interactive(self, res=None, store_transect_img=False):
        """
        Interactively define a transect in JupyterLab.

        Left click: Add point
        Backspace/Delete: Remove last point
        Enter or Finish button: Complete the selection

        Returns
        -------
        x, y : np.ndarray
            Coordinates of the selected transect vertices.

        Example use in jupyterlab:
        %matplotlib widget
        points = await M.prepare_section_interactive()

        - will also write .x_sec, .y_sec to the grid object
        """
        points = []
        future = asyncio.get_running_loop().create_future()

        # Create the map without automatically displaying the figure
        with plt.ioff():
            plt.figure()
            self.georeference()
            self.plot_contour(self.h, levels = np.linspace(self.h.min(), self.h.max(), 30), show = False)
            self.plot_grid(rasterized=True, show = False)

            fig = plt.gcf()
            ax = plt.gca()

            line, = ax.plot([], [], "ro-", markersize=5, linewidth=1.5, zorder=100)

            ax.set_title("Create a transect")

        # Create controls
        finish_button = widgets.Button(description="Finish", button_style="success")
        undo_button = widgets.Button(description="Undo")
        status = widgets.Label(value="Select at least two points.")

        def redraw():
            """Update the transect line and status."""
            line.set_data([p[0] for p in points], [p[1] for p in points])
            status.value = f"{len(points)} points selected"
            fig.canvas.draw_idle()

        def on_click(event):
            """Register a new point from a left mouse click."""
            if future.done():
                return

            if event.inaxes != ax or event.button != 1:
                return

            # Ignore clicks while the toolbar is in pan/zoom mode
            toolbar = getattr(fig.canvas, "toolbar", None)
            if toolbar is not None and toolbar.mode:
                return

            if event.xdata is not None and event.ydata is not None:
                points.append((event.xdata, event.ydata))
                redraw()

        def on_undo(_=None):
            """Remove the most recently selected point."""
            if points and not future.done():
                points.pop()
                redraw()

        def on_finish(_=None):
            """Finish selecting the transect."""
            if future.done():
                return

            if len(points) < 2:
                status.value = "At least two points are required."
                return

            future.set_result(np.asarray(points, dtype=float))

        def on_key(event):
            """Handle keyboard shortcuts."""
            if event.key == "enter":
                on_finish()
            elif event.key in ("backspace", "delete"):
                on_undo()

        # Connect Matplotlib events
        click_id = fig.canvas.mpl_connect(
            "button_press_event", on_click
        )
        key_id = fig.canvas.mpl_connect(
            "key_press_event", on_key
        )

        finish_button.on_click(on_finish)
        undo_button.on_click(on_undo)

        # Display the interactive map and controls
        display(widgets.VBox([
            fig.canvas,
            widgets.HBox([finish_button, undo_button]),
            status
        ]))

        try:
            # Yield control to Jupyter while waiting for user input
            xy = await future

        finally:
            # Disconnect event handlers
            fig.canvas.mpl_disconnect(click_id)
            fig.canvas.mpl_disconnect(key_id)
            finish_button.disabled = True
            undo_button.disabled = True

        x_section = xy[:, 0]
        y_section = xy[:, 1]

        # Use the existing section preparation routine
        self.prepare_section(
            res=res,
            store_transect_img=store_transect_img,
            section_points=(x_section, y_section)
        )

        return x_section, y_section

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
        """Create equally spaced transect points along each segment."""

        points = np.column_stack((x_section, y_section))

        if len(points) < 2:
            raise ValueError("A section requires at least two points")

        if not np.isfinite(res) or res <= 0:
            raise ValueError("res must be positive")

        segments = []

        for start, stop in zip(points[:-1], points[1:]):
            distance = np.linalg.norm(stop - start)

            if distance == 0:
                continue

            n = max(1, int(np.ceil(distance / res)))
            t = np.linspace(0, 1, n + 1)

            # Avoid duplicate points at segment junctions.
            if segments:
                t = t[1:]

            segments.append(
                start + t[:, None] * (stop - start)
            )

        if not segments:
            raise ValueError("Section has zero length")

        xy = np.concatenate(segments, axis=0)
        self.x_sec, self.y_sec = xy.T

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
        if data.shape[0] not in (self.node_number, self.cell_number):
            data = data.T

        sigma = self._check_if_siglay_or_siglev(data)
        grid = self._check_if_cell_or_node(data)
        self._initialize_section(section_file, res, store_transect_img, sigma, grid) # TBD: in here, find way to remove points outside of the domain?
        out['h'], out['x'], out['y'] = self.dpt_sec, self.x_sec, self.y_sec

        # Interpolate data to the section
        # ----
        if len(data.shape)>1:
            out['transect'] = self._interpolate_3D(data)
            out['dst'] = np.repeat(self.dst_sec[:, None], data.shape[1], axis = 1)
        else:
            out['transect'] = self._interpolate_2D(data)
            out['dst'] = self.dst_sec

        # Remove nans and crop dict, contourf won't accept the input otherwise -- do this earlier though, not every time...
        # ----
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

    def _section_velocity_basis(self, velocity_coordinates='east_north'):
        """Unit tangent and normal in the velocity component coordinate frame.

        x_sec/y_sec must be in a metric, projected CRS. For east_north,
        rotate geographic east/north into grid x/y using short geodesic steps;
        the resulting basis is cached across frames.
        """
        if velocity_coordinates not in ('east_north', 'projected'):
            raise ValueError("velocity_coordinates must be 'east_north' or 'projected'")

        xy = np.column_stack((self.x_sec, self.y_sec))
        cache = self.__dict__.get('_section_vector_basis_cache')
        if (cache is not None and cache['coordinates'] == velocity_coordinates
                and cache['reference'] == self.reference
                and np.array_equal(cache['xy'], xy)):
            return cache['tangent_e'], cache['normal_e'], cache['tangent_n'], cache['normal_n']

        if len(xy) < 2 or not np.isfinite(xy).all():
            raise ValueError('Section requires at least two finite projected points')
        from pyproj import CRS, Geod
        crs = CRS.from_user_input(self.reference)
        if not crs.is_projected or not all(axis.unit_conversion_factor and
                                            np.isclose(axis.unit_conversion_factor, 1.0)
                                            for axis in crs.axis_info[:2]):
            raise ValueError('Section coordinates must use a projected CRS with metre units')

        # A length-weighted central tangent is appropriate for an interpolated
        # polyline. At sharp corners its normal is an average of both segments.
        dx = np.gradient(xy[:, 0])
        dy = np.gradient(xy[:, 1])
        length = np.hypot(dx, dy)
        if np.any(length <= 0):
            raise ValueError('Section contains points with undefined tangent')
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx  # positive normal: left of start -> end

        if velocity_coordinates == 'east_north':
            lon, lat = self.Proj(xy[:, 0], xy[:, 1], inverse=True)
            geod = Geod(ellps='WGS84')
            lon_e, lat_e, _ = geod.fwd(lon, lat, np.full(len(xy), 90.0), np.full(len(xy), 10.0))
            lon_n, lat_n, _ = geod.fwd(lon, lat, np.zeros(len(xy)), np.full(len(xy), 10.0))
            xe, ye = self.Proj(lon_e, lat_e)
            xn, yn = self.Proj(lon_n, lat_n)
            e_dx, e_dy = xe - xy[:, 0], ye - xy[:, 1]
            n_dx, n_dy = xn - xy[:, 0], yn - xy[:, 1]
            e_norm = np.hypot(e_dx, e_dy)
            n_norm = np.hypot(n_dx, n_dy)
            if np.any(e_norm <= 0) or np.any(n_norm <= 0):
                raise ValueError('Could not derive east/north vectors from CRS')
            ex, ey = e_dx / e_norm, e_dy / e_norm
            gx, gy = n_dx / n_norm, n_dy / n_norm
            tangent_e, normal_e = tx * ex + ty * ey, nx * ex + ny * ey
            tangent_n, normal_n = tx * gx + ty * gy, nx * gx + ny * gy
        else:
            tangent_e, normal_e, tangent_n, normal_n = tx, nx, ty, ny

        self._section_vector_basis_cache = dict(
            xy=xy.copy(), reference=self.reference, coordinates=velocity_coordinates,
            tangent_e=tangent_e, normal_e=normal_e,
            tangent_n=tangent_n, normal_n=normal_n,
        )
        return tangent_e, normal_e, tangent_n, normal_n

    def get_section_velocity(self, u, v, velocity_coordinates='east_north'):
        """Interpolate FVCOM 3-D cell velocities onto the current section.

        Parameters
        ----------
        u, v : array-like
            Cell-centred velocities with shape (sigma_layers, n_cells) or
            (n_cells, sigma_layers). Normally read using load_netCDF.
        velocity_coordinates : {'east_north', 'projected'}
            'east_north' (default): u geographic east and v geographic north.
            'projected': u/v already aligned to the projected mesh x/y axes.
            Section x/y must be in metres in either case.

        Returns
        -------
        dict
            'normal', 'along', 'speed', 'u', 'v', 'h', 'dst', 'x', 'y'.
            Arrays of velocity and physical z coordinate are indexed by
            (section_point, sigma_layer). Positive normal is to the LEFT
            looking from the first section point toward the last.
            NaNs in either velocity component mask BOTH components.
        """
        def normalize(field, name):
            arr = np.ma.filled(np.ma.asarray(field, dtype=float), np.nan)
            layers = self.siglay.shape[1]
            if arr.ndim != 2:
                raise ValueError(f'{name} must be a 3-D FVCOM velocity field at one timestep')
            if arr.shape == (layers, self.cell_number):
                return arr.T
            if arr.shape == (self.cell_number, layers):
                return arr
            raise ValueError(f'{name} has shape {arr.shape}; expected (sigma, cell) or (cell, sigma)')

        u = normalize(u, 'u')
        v = normalize(v, 'v')
        if not hasattr(self, 'x_sec') or not hasattr(self, 'y_sec'):
            raise ValueError('Call prepare_section() before get_section_velocity()')

        self._initialize_section(None, None, False, 'siglay', 'c')
        u_sec = self._interpolate_3D(u)
        v_sec = self._interpolate_3D(v)

        # Physical z of each SIGLAY midpoint. Unlike the visualization-only
        # dpt_sec, this is NOT adjusted to pin the bottom at a fixed level.
        # Masked/dry zeta must not change section geometry from frame to frame.
        # Velocity is still masked there, but use zeta=0 to locate its grid layers.
        safe_zeta = np.where(np.isfinite(self.zeta), self.zeta, 0.0)
        zeta_cells = np.mean(safe_zeta[self.tri], axis=1)
        z_cells = (zeta_cells[:, None]
                   + (self.hc + zeta_cells)[:, None] * self.siglay_c)
        z_sec = self._interpolate_3D(z_cells)

        te, ne, tn, nn = self._section_velocity_basis(velocity_coordinates)
        normal = u_sec * ne[:, None] + v_sec * nn[:, None]
        along = u_sec * te[:, None] + v_sec * tn[:, None]
        speed = np.hypot(u_sec, v_sec)

        invalid = (~np.isfinite(u_sec) | ~np.isfinite(v_sec) | ~np.isfinite(z_sec))
        for arr in (normal, along, speed, u_sec, v_sec):
            arr[invalid] = np.nan

        # Exclude only locations lacking GEOMETRY (not temporarily dry
        # locations); prevents changing horizontal sampling across frames.
        valid_rows = np.all(np.isfinite(z_sec), axis=1) & np.isfinite(self.dst_sec)
        dst = np.broadcast_to(self.dst_sec[:, None], z_sec.shape)
        return dict(
            u=u_sec[valid_rows], v=v_sec[valid_rows],
            normal=normal[valid_rows], along=along[valid_rows], speed=speed[valid_rows],
            h=z_sec[valid_rows], dst=dst[valid_rows],
            x=np.asarray(self.x_sec)[valid_rows], y=np.asarray(self.y_sec)[valid_rows],
        )

    def _initialize_section(self, section_file, res, store_transect_img, sigma, grid):
        '''
        Prepare the section interpolators etc.
        '''
        # A cached triangulation for node data must not be used for cell data.
        if getattr(self, '_transect_grid', None) != grid:
            self.__dict__.pop('trs', None)
            self.__dict__.pop('_section_weights_cache', None)
        self._transect_sigma, self._transect_grid = sigma, grid
        if not (hasattr(self, 'x_sec') and hasattr(self, 'y_sec')):
            self.prepare_section(section_file=section_file, res=res,
                                 store_transect_img=store_transect_img)
        if not hasattr(self, 'trs'):
            self.trs = matplotlib.tri.Triangulation(
                getattr(self, f'x{self._transect_grid}'), 
                getattr(self, f'y{self._transect_grid}'), 
                triangles = getattr(self, f'{self._transect_grid}tri')
                )

    def _section_weights(self):
        """Return cached triangle vertices/weights for the section points.

        A triangulation lookup is needed only when the transect/grid changes,
        not for every vertical level and every movie frame.
        """
        xy = np.column_stack((self.x_sec, self.y_sec))
        cache = self.__dict__.get('_section_weights_cache')
        if (cache is not None and cache['triangulation'] is self.trs
                and np.array_equal(cache['xy'], xy)):
            return cache['vertices'], cache['weights'], cache['inside']

        triangle_ids = np.asarray(self.trs.get_trifinder()(
            xy[:, 0], xy[:, 1]), dtype=int)
        inside = triangle_ids >= 0
        vertices = np.zeros((len(xy), 3), dtype=int)
        weights = np.zeros((len(xy), 3), dtype=float)
        if inside.any():
            vertices[inside] = self.trs.triangles[triangle_ids[inside]]
            v = vertices[inside]
            x0, y0 = self.trs.x[v[:, 0]], self.trs.y[v[:, 0]]
            x1, y1 = self.trs.x[v[:, 1]], self.trs.y[v[:, 1]]
            x2, y2 = self.trs.x[v[:, 2]], self.trs.y[v[:, 2]]
            xx, yy = xy[inside].T
            denominator = ((y1 - y2) * (x0 - x2)
                           + (x2 - x1) * (y0 - y2))
            w0 = ((y1 - y2) * (xx - x2)
                  + (x2 - x1) * (yy - y2)) / denominator
            w1 = ((y2 - y0) * (xx - x2)
                  + (x0 - x2) * (yy - y2)) / denominator
            weights[inside] = np.column_stack((w0, w1, 1.0 - w0 - w1))

        self._section_weights_cache = {
            'triangulation': self.trs, 'xy': xy.copy(),
            'vertices': vertices, 'weights': weights, 'inside': inside
        }
        return vertices, weights, inside

    def _interpolate_2D(self, field):
        """Interpolate a 1-D nodal/cell field to the fixed section."""
        values = np.ma.filled(np.ma.asarray(field, dtype=float), np.nan)
        if values.ndim != 1 or len(values) != len(self.trs.x):
            raise ValueError('field must have one value per triangulation vertex')
        vertices, weights, inside = self._section_weights()
        out = np.full(len(self.x_sec), np.nan)
        if inside.any():
            out[inside] = np.sum(values[vertices[inside]] * weights[inside], axis=1)
        return out

    def _interpolate_3D(self, field):
        """Interpolate (grid_point, level) data using cached section weights."""
        values = np.ma.filled(np.ma.asarray(field, dtype=float), np.nan)
        if values.ndim != 2 or values.shape[0] != len(self.trs.x):
            raise ValueError('field must have shape (grid_point, level)')
        vertices, weights, inside = self._section_weights()
        out = np.full((len(self.x_sec), values.shape[1]), np.nan)
        if inside.any():
            out[inside] = np.einsum(
                'pv,pvk->pk', weights[inside], values[vertices[inside]],
                optimize=True)
        return out

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

    def write_bath(self, filename = None, latlon = False):
        '''
        - Generates an ascii FVCOM 5.x format bathymetry file
        '''
        if filename is None: 
            filename = f'input/{self.casename}_dep.dat'

        x_grid, y_grid = self.get_xy(latlon)
        with open(filename, 'w') as f:
            f.write(f'Node Number = {self.node_number}\n')
            for x, y, h in zip(x_grid, y_grid, self.h):
                line = '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(h)+'\n'
                f.write(line)
        print(f'  - Wrote : {filename}')

    def write_grd(self, filename = None, latlon = False, store_z = False):
        '''
        - Generates an ascii FVCOM 5.x format grid file
        '''
        if filename is None: 
            filename = f'input/{self.casename}_grd.dat'

        x_grid, y_grid = self.get_xy(latlon)  
        if store_z:
            h = self.h
        else:
            h = np.zeros(x_grid.shape)
            
        with open(filename, 'w') as f:
            f.write(f'Node Number = {self.node_number}\n')
            f.write(f'Cell Number = {self.cell_number}\n')
            for i, (t1, t2, t3) in enumerate(self.tri):
                f.write(f'{i+1} {t1+1} {t2+1} {t3+1} {i}\n')

            for i, (x, y, z) in enumerate(zip(x_grid, y_grid, h)):
                line = str(i+1) +' '+ '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(z)+'\n'
                f.write(line)
        print(f'  - Wrote : {filename}')

    def write_sponge(self, filename = None):
        '''
        - Generates an ascii FVCOM 5.x formatted sponge file
        '''
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
        '''
        - Generates an ascii FVCOM 5.x formatted obc file
        '''
        if filename is None: 
            filename = f'input/{self.casename}_obc.dat'

        with open(filename, 'w') as f:
            f.write(f'OBC Node Number = {len(self.obc_nodes)}\n')
            i = 0
            for nodestring in self.nodestrings:
               for obc_node in nodestring:
                    i += 1
                    f.write(f'{i} {obc_node+1} 1\n')
        print(f'  - Wrote : {filename}')

    def write_cor(self, filename = None, latlon = False):
        '''
        Generates an ascii FVCOM 5.x formatted coriolis file
        - filename: name of the output file
        - latlon:   True if the model is configured in spherical mode with WGS84 coordinates
        '''
        if filename is None: 
            filename = f'input/{self.casename}_cor.dat'

        if latlon:
            x, y = self.lon, self.lat
        else:
            x, y = self.x, self.y

        with open(filename, 'w') as f:
            f.write(f'Node Number = {self.node_number}\n')
            for x, y, lat in zip(x, y, self.lat):
                line = '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(lat)+'\n'
                f.write(line)
        print(f'  - Wrote : {filename}')

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
        project = Transformer.from_proj(self.Proj, Proj(init='epsg:4326')) # We always want to project to latlon, hence hardcoded
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
        '''
        write x_sec, y_sec to a casename_section.txt file
        '''
        lonlat = np.vstack((self.Proj(self.x_sec, self.y_sec, inverse=True))).T
        np.savetxt(f'{self.casename}_section.txt', lonlat, delimiter = ' ')
        print(f'- saved section positions to {self.casename}_section.txt')

    def to_npy(self, filename = 'M.npy'):
        '''
        Write no M.npy file
        '''
        M = {}
        self.nv = self.tri
        for var in ['x', 'y', 'lon', 'lat', 'h', 'h_raw', 'nv', 'siglay', 'siglev', 'obc_nodes', 'nodestrings', 'reference', 'info', 'ts']:
            try:
                M[var] = getattr(self, var)
            except:
                pass
        np.save(filename, M)
        print(f'- Wrote {filename}')

class PlotFVCOM:
    @cached_property
    def mtri_nodes(self):
        """Cached Matplotlib triangulation for nodal fields.

        The geometry is shared by successive movie frames.  Only the
        triangle mask changes for wet/dry or below-bottom values.
        """
        return matplotlib.tri.Triangulation(self.x, self.y, self.tri)

    @cached_property
    def mtri_cells(self):
        """Cached cell-centre triangulation for FVCOM velocity fields."""
        return matplotlib.tri.Triangulation(self.xc, self.yc, self.ctri)

    @property
    def transform(self):
        if not hasattr(self, '_transform'):
            return None
        return self._transform

    @transform.setter
    def transform(self, var):
        self._transform = var

    def plot_grid(self, ax = None, c = 'g-', linewidth = 0.2, markersize = 0.2, show = True, *args, **kwargs):
        '''
        Plot mesh grid
        - arguments and keyword arguments are passed to pyplot.triplot
        '''
        if ax == None:
            ax = plt.gca()
        kwargs = self._transform_to_kwargs(**kwargs)
        ax.triplot(self.x, self.y, self.tri, c, *args, markersize=markersize, linewidth=linewidth, **kwargs)
        ax.set_aspect('equal')
        if show: plt.show(block = False)

    def plot_obc(self, ax = None, **kwargs):
        '''
        plot the obc nodes, show different nodestrings
        '''
        if ax == None:
            ax = plt.gca()
        kwargs = self._transform_to_kwargs(**kwargs)
        for i in range(len(self.x_obc)):
            ax.scatter(self.x_obc[i], self.y_obc[i], zorder = 10, label=f'boundary nr. {i+1}', **kwargs)
            ax.scatter(self.x_obc[i][[0,-1]], self.y_obc[i][[0,-1]], zorder = 11, c = 'r', **kwargs) # start and end points 
        #plt.draw()

    def plot_contour(self, field, show=True, ax=None, *args,
                     on_empty='raise', **kwargs):
        """Contour node- or cell-centred data on a cached triangulation.

        The mesh geometry is created once per grid.  Invalid values in a
        timestep only update its triangle mask, so masked z-level fields
        never reach an unmasked triangle in ``tricontourf``.  This method
        never copies or mutates the FVCOM connectivity arrays themselves.

        The cached triangulation's mask is the mask from its **most recent**
        contour call.  Callers should not rely on that mask remaining fixed.
        """
        if on_empty not in ('raise', 'skip'):
            raise ValueError("on_empty must be 'raise' or 'skip'")

        values = np.ma.asarray(np.squeeze(field), dtype=float)
        if values.ndim != 1:
            raise ValueError(f'Expected a one-dimensional contour field, got {values.shape}')
        if values.size == self.node_number:
            triangulation = self.mtri_nodes
        elif values.size == self.cell_number:
            triangulation = self.mtri_cells
        else:
            raise ValueError(
                f'Field length {values.size} does not match '
                f'{self.node_number} nodes or {self.cell_number} cells'
            )

        # Matplotlib rejects *masked* or non-finite values in the vertices
        # of a visible triangle.  Mask triangles, not just vertex data.
        data = np.asarray(np.ma.masked_invalid(values).filled(np.nan))
        triangles = triangulation.triangles
        triangle_mask = (~np.isfinite(data))[triangles].any(axis=1)
        caller_mask = kwargs.pop('mask', None)
        if caller_mask is not None:
            caller_mask = np.asarray(caller_mask, dtype=bool)
            if caller_mask.shape != (len(triangles),):
                raise ValueError('Triangle mask has wrong shape')
            triangle_mask |= caller_mask

        # set_mask() invalidates Matplotlib's cached C++ triangulation. Avoid
        # calling it when the mask has not changed (common for dry-area masks).
        previous = triangulation.mask
        replacement = triangle_mask if np.any(triangle_mask) else None
        changed = ((previous is None) != (replacement is None))
        if not changed and previous is not None:
            changed = not np.array_equal(previous, replacement)
        if changed:
            triangulation.set_mask(replacement)

        if replacement is not None and np.all(replacement):
            if on_empty == 'skip':
                return None
            raise ValueError('No valid triangles to contour')
        if triangles.size == 0:
            if on_empty == 'skip':
                return None
            raise ValueError('No triangles to contour')

        kwargs = self._transform_to_kwargs(ax=ax, **kwargs)
        return self._plot_contour(triangulation, data, show, ax=ax,
                                  *args, **kwargs)

    def _transform_to_kwargs(self, ax=None, **kwargs):
        """Pass the CRS transform to a Cartopy axes when applicable.

        Explicit axes avoid pyplot global-figure lookups during animation
        saving, where Matplotlib temporarily detaches the figure manager.
        """
        if self.transform is not None:
            import cartopy.mpl.geoaxes as geoaxes
            target = ax if ax is not None else plt.gca()
            if isinstance(target, geoaxes.GeoAxes):
                kwargs['transform'] = self.transform
        return kwargs

    def georeference(self, 
                    url='https://wms.geonorge.no/skwms1/wms.topograatone?service=wms&request=getcapabilities', 
                    layers=['topograatone'], wms=None, 
                    depth = True,
                    ax = None):
        '''
        Plot map data from WMS server as georeference. Must be done before plotting grid, contours etc.
        - requires cartopy
        ---
        url:    defaults to grey norgeskart
                - For international use, try e.g. url = 'https://ows.terrestris.de/osm/service?', layers = ['OSM-WMS-no-labels']
        layers: valid layer from the url, must be a tuple with string(s)
        wms:    overwrites other input if not None, current options: "raster" or "topo4"
        depth:  plot kartverket bathymetry
        ax:     cartopy axes (e.g. fig, ax = plt.subplots(2, 1, subplot_kw={'projection': ccrs.PlateCarree()}))

        Returns:
        ---
        ax, a GeoAxesSubplot

        - Also possible to plot fiskeridirektoratet akvakultur lokaliteter WMS on top:
          - ax.add_wms(wms='https://gis.fiskeridir.no/server/services/fiskeridirWMS_akva/MapServer/WMSServer?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities', 
                       layers=['akvakultur_lokaliteter'])
        '''
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
        if not ax:
            if int(self.reference.split(':')[-1]) == 4326:
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
        ax = plt.gca()
        ax.set_extent(extent, crs=ccrs.PlateCarree(globe=ccrs.Mercator().globe))
        gl = ax.gridlines(ccrs.PlateCarree(globe=ccrs.Mercator().globe), draw_labels=True)         
        gl.top_labels = None

    @staticmethod
    def _plot_contour(triangulation, field, show=False, ax=None, *args, **kwargs):
        """Draw on the already constructed/masked triangulation."""
        if ax is None:
            ax = plt.gca()
        cont = ax.tricontourf(triangulation, field, *args, **kwargs)
        ax.set_aspect('equal')
        if show:
            plt.show(block=False)
        return cont

    def _plot_section_line(self):
        '''
        Show where the section goes relative to the grid
        '''
        self.plot_grid(rasterized=True, show = False)
        plt.plot(self.x_sec, self.y_sec, 'k', zorder = 10)
        plt.plot(self.x_sec[0], self.y_sec[0], 'g.', zorder = 11, label = 'start')
        plt.plot(self.x_sec[-1], self.y_sec[-1], 'r.', zorder = 11, label = 'stop')
        plt.xlim([self.x_sec.min()-10000, self.x_sec.max()+10000])
        plt.ylim([self.y_sec.min()-10000, self.y_sec.max()+10000])
        plt.legend()
        plt.title('Section')
        plt.savefig('Section_map.png')

    def plot_section_3D(self, x, y, z, data, vmin = None, cmap = cmo.cm.balance):
        '''
        Plot data at scattered x,y,z points
        - x and y are (N,) arrays
        - z and data are (N, D) arrays

        Optional
        
        '''
        import matplotlib.colors as colors

        vmin = vmin if vmin is not None else data.min()

        # Expand array
        x = np.tile(x[:, None], [1, data.shape[1]])
        y = np.tile(y[:, None], [1, data.shape[1]])

        # Prepare the figure
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        norm = colors.Normalize(vmin = vmin, vmax = data.max())

        # Plot the data
        surf = ax.plot_surface(x, y, z, facecolors = cmap(norm(data)), cmap = cmap)
        surf.set_clim(vmin = 0, vmax = data.max())
        fig.colorbar(surf, label = 'm/s')
        ax.set_zlabel('depth [m]')
        return fig, ax

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
            contour = self.plot_contour(ts_min, cmap = 'jet', levels = np.linspace(min(ts_min), 3*min(ts_min), 20), extend = 'max', show = False)
            plt.colorbar(contour, label = 's')
            plt.axis('equal')
        self.ts = ts_min

class LegacyFunctions:
    def cell2node(self, fieldin):
        """Average cell values at connected mesh nodes (equal cell weights).

        Vectorized via bincount. Nodes without any attached cell return NaN.
        """
        values = np.asarray(fieldin, dtype=float)
        if values.ndim != 1 or values.size != self.cell_number:
            raise ValueError('fieldin must have one value per cell')
        nodes = self.tri.ravel()
        total = np.bincount(nodes, weights=np.repeat(values, 3),
                            minlength=self.node_number)
        count = np.bincount(nodes, minlength=self.node_number)
        return np.divide(total, count, out=np.full(self.node_number, np.nan),
                         where=count > 0)

class OutputLoader(FVCOMNetCDFIO):
    """FVCOM_grid NetCDF compatibility layer. See netcdf_io.FVCOMNetCDFIO."""


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
                .cvs            - matplotlib.collection.PatchCollection objects - one pr. node based control volume
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
            .prepare_section_interactive() 
                - Same as prepare_section, but to be used in jupyter notebooks with %matplotlib widget.
                - Must be called with await (e.g. await M.prepare_section_interactive())
                                            

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
                 reference = 'epsg:32633',
                 verbose   = False,
                 static_depth = False,
                 x = None, y = None, 
                 lon = None, lat = None,
                 tri = None,
                 h = None, siglay = None, siglev = None,
                 obc_nodes = None,
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
        - by default UTM33W

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
        self.info = {}
        self.info['reference'] = reference

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
        repr_start = f'''{self.casename} FVCOM grid object

Grid read from:         {self.filepath}
Number of nodes:        {len(self.x)}
Number of triangles:    {len(self.xc)}
'''
        repr_end = f'''
Grid resolution:   min: {np.min(self.grid_res):.2f} m, max: {np.max(self.grid_res):.2f} m
Grid angles (deg): min: {np.min(self.grid_angles):.2f}, max: {np.max(self.grid_angles):.2f}
Reference:              {self.reference}
'''     
        
        try:
            repr_sig = f'''
Number of sigma layers: {self.siglay.shape[1]}
Number of sigma levels: {self.siglev.shape[1]}
'''
            repr = repr_start + repr_sig + repr_end
        except:
            repr = repr_start + repr_end

        return repr

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

class InputError(Exception):
    pass