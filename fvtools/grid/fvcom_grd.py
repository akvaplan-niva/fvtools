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