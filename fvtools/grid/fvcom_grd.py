import os
import numpy as np
import geopandas as gpd
import shapely as shp

from pyproj import Proj, Transformer
from scipy.spatial import cKDTree as KDTree
from numba import njit

import warnings
warnings.filterwarnings("ignore")

from fvtools.grid.netcdf_io import FVCOMNetCDFIO
from fvtools.grid._components.grid_loader import GridLoader
from fvtools.grid._components.coordinates import InputCoordinates, Coordinates
from fvtools.grid._components.proximity import GridProximity
from fvtools.grid._components.topology import PropertiesFromTGE
from fvtools.grid._components.triangulation import CellTriangulation
from fvtools.grid._components.cropping import CropGrid
from fvtools.grid._components.coastline import CoastLine
from fvtools.grid._components.obc import OBC
from fvtools.grid._components.control_volumes import ControlVolumePlotter
from fvtools.grid._components.sections import SectionMaker
from fvtools.grid._components.plotting import PlotFVCOM
from fvtools.grid._components.physics import AnglesAndPhysics

from fvtools.grid._components.compatibility import LegacyPropertyAliases, LegacyFunctions, OutputLoader

from fvtools.grid._components.interpolation import InterpolateToZ

from fvtools.grid._components.export import ExportGrid




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

from fvtools.grid._components.nesting import LegacyNestPropertyAliases, LoadNest, NestROMS2FVCOM, NEST_grid




class InputError(Exception):
    pass