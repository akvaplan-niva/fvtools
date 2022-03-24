#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

import os
import pickle
import fvtools.grid.tge as tge
import fvtools.grid.fvgrid as fvgrid
import fvtools.grid.tools as tools
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cmocean as cmo
import numpy as np
import time
import progressbar as pb

from pyproj import Proj
from scipy.io import loadmat

from netCDF4 import Dataset
from matplotlib.patches import Polygon as mPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree as KDTree
from numba import njit

class FVCOM_grid():
    '''Represents FVCOM grid

    Will automatically assume that you want to use UTM33W coordinates. If your mesh input file contains a
    reference to a UTM coordinate, it will use that instead.
    '''
    def __init__(self,
                 pathToFile,
                 reference = 'epsg:32633',
                 verbose   = False):
        '''
        Create the fields we expect in a FVCOM-grid object
        '''
        self.filepath       = pathToFile
        self.reference      = reference
        self.Proj           = self._get_proj(reference)
        self.cropped_object = False
        global verbose_g
        verbose_g   = verbose

        if pathToFile[-3:] == 'mat':
            self._load_mat()

        elif pathToFile[-3:] == 'npy':
            self._load_npy()

        elif pathToFile[-3:] == '2dm':
            self._load_2dm()

        elif pathToFile[-3:] == '.nc':
            self._load_nc()

        else:
            raise InputError(f'{self.filepath} is not a valid for FVCOM_grid')

        # For the case where the input file does not contain x,y or lon,lat data
        # ----
        if np.count_nonzero(self.x) == 0:
            self.x,  self.y  = self.Proj(self.lon, self.lat)
            self.xc, self.yc = self.Proj(self.lonc, self.latc)

        if np.count_nonzero(self.lon) == 0:
            self.lon,   self.lat   = self.Proj(self.x,  self.y,  inverse=True)
            self.lonc,  self.latc  = self.Proj(self.xc, self.yc,  inverse=True)

    def __str__(self):
        return f'''FVCOM grid from {self.filepath}
--

Attributes:
        Position data
            x, y   (lon, lat)   - node position
            xc, yc (lonc, latc) - cell position
            tri                 - triangulation
            siglay, siglay_c    - middle of sigma box (scalar variables are stored)
            siglev, siglev_c    - top and bottom of sigma layer top and bottom interfaces
            siglayz, sialayz_uv - depth of sigma layer mid-points (where u,v are stored)
            siglevz, siglevz_uv - depth of sigma layer top and bottom interfaces (where w is stored)
            h, hc               - total depth

        Grid identifiers
            nbsn   - nearby surrounding nodes  (to a node)
            nbve   - nearby surroundiing cells (to a node)
            ntsn   - number of nodes surrounding a node
            ntve   - number of elements surrounding a node

        Grid area
            art1     - area of node-based control volume (CV)
            tri_area - area of triangles

        Grid volume
            node_volume - volume connected to the data on a node
            cell_volume - volume connected to the data at a cell

        Open boundary identifiers
            read_obc_nodes - numpy object with nodes in each obc
            obc_nodes      - simple list of all obc nodes
            obc_type       - identifier saying which type of OBC condition was used
                             (will only be read from _restart_****.nc files)

Functions:
        Plotting:
            .plot_grid()      - plots a grid
            .plot_cvs(data)   - plots the CV-based data on CV-patches
            .plot_field(data) - plots data over triangles

        Grid:
            .write_2dm()      - return mesh as a .2dm file that can be read to SMS
            .get_obc()        - turns a .read_obc_nodes object to a .obc_nodes list, 
                                and optionally plots .obc-nodes
            .get_res()        - computes an estimate of the mesh resolution 
            .get_angle()      - computes the angle of each triangle corner
            .get_coast()      - returns coastline polygons (will also include the obc)
            .find_nearest()   - find nearest grid point (either cell or node) to a point
            .isinside()       - find nodes inside a search area
 
        Physics
            .get_coriolis()   - Computes the coriolis parameter

        Data extraction
            .interpolate_to_z()  - Interpolates input-field to z-depth
            .get_section_data()  - Interpolates input field along a section
            .smooth()            - smooths a input field
            .make_interpolation_matrices_TS() - creates matrix to interpolate node data to z-depth
            .make_interpolation_matrices_uv() - creates matrix to interpolate cell data to z-depth

        Mesh cropping
            .subgrid()        - crops the mesh object and returns a smaller one with indexing
                                to the old one. (usefull to reduce data-handling size)

Extended features through the mesh-connectivity TGE (.T) class:
        -> This contains functions to compute some heavier stuffs (will take a minute or two to set-up)
            - .T.node_gradient()     - Gradient of data stored on nodes
            - .T.cell_gradient()     - Gradient of data stored on cells
            - .T.vertical_gradient() - Vertical gradient of data
            - .T.vorticity()         - vorticity of depth-averaged currents
            - .T.vorticity_3D()      - vorticity of 3D-currents
            - .T.okubo_weiss()       - okubo weiss parameter (for eddy detection)


-------------------------------------------------------------------------------------------------------
Short summary of this mesh:
----
Number of nodes:       {len(self.x)}
Number of triangles:   {len(self.xc)}
Grid resolution:       {np.min(self.get_res()):.2f} m to {np.max(self.get_res()):.2f} m
Grid angles (degrees): min: {np.min(self.get_angles()):.2f}, max: {np.max(self.get_angles()):.2f}
Casename:              {self.casename}

'''

    # Some grid properties we can't always read from files, and may need to compute.
    # ----
    @property
    def T(self):
        '''
        T computes the same grid connectivity metrics as tge.F

        Does also include routines to compute gradients, vorticity and okubo weiss parameter.
        '''
        if not hasattr(self, '_T'):
            print('- computing grid connectivity using tge, may take a minute.')
            self._T = tge.main(self, verbose = False)
            self._nbsn = self._T.NBSN 
            self._nbve = self._T.NBVE
        return self._T
    
    @property
    def art1(self):
        '''
        art1 is the control volume (node centered) area
        '''
        if not hasattr(self, '_art1'):
            self.T.get_art1()
            self._art1 = self.T.art1
        return self._art1

    @art1.setter
    def art1(self, val):
        self._art1 = val

    @property
    def tri_area(self):
        '''
        tri area is the area of the triangles
        '''
        if not hasattr(self, '_tri_area'):
            self._tri_area = self.calculate_tri_area()
        return self._tri_area

    @tri_area.setter
    def tri_area(self, val):
        self._tri_area = val
        return self._tri_area

    @property
    def nbsn(self):
        '''
        nbsn is a reference to nodes surrounding each node
        '''
        if not hasattr(self, '_nbsn'):
            self._nbsn = self.T.NBSN
        return self._nbsn 

    @nbsn.setter
    def nbsn(self, val):
        self._nbsn = val

    @property
    def nbve(self):
        '''
        nbve is a reference to the elements surrounding each node
        '''
        if not hasattr(self, '_nbve'):
            self._nbve = self.T.NBVE
        return self._nbve

    @nbve.setter
    def nbve(self, val):
        self._nbve = val

    @property
    def siglev_c(self):
        if hasattr(self, 'siglev'):
            self._siglev_c = np.mean(self.siglev[self.tri,:], axis = 1)
        else:
            self._siglev_c = None
        return self._siglev_c

    @siglev_c.setter
    def siglev_c(self, var):
        self._siglev_c = var 

    @property
    def siglay_c(self):
        '''
        sigma layer at cells
        '''
        if hasattr(self, 'siglay'):
            self._siglay_c = np.mean(self.siglay[self.tri,:], axis = 1)
        else:
            self._siglay_c = None
        return self._siglay_c

    @siglay_c.setter
    def siglay_c(self, var):
        self._siglay_c = var

    @property
    def siglev_center(self):
        return self.siglev_c

    @property
    def siglay_center(self):
        return self.siglay_c
    
    @property
    def node_volume(self):
        '''
        Volume (m^3) of a node-based control volume
        '''
        if not hasattr(self, '_node_volume'):
            self._node_volume = -np.diff(self.siglevz) * self.art1[:, None]
        return self._node_volume

    @property
    def cell_volume(self):
        '''
        Volume (m^3) of a cell-based control volume
        '''
        if not hasattr(self, '_cell_volume'):
            self._cell_volume = -np.diff(self.siglevz_uv) * self.tri_area[:, None]
        return self._cell_volume

    @property
    def hc(self):
        '''
        depth at cell center
        '''
        if not hasattr(self, '_hc'):
            if hasattr(self, 'h'):
                self._hc = np.mean(self.h[self.tri], axis = 1)
            else:
                raise ValueError(f'- {self.filepath} does not contain depth info')
        return self._hc

    @hc.setter
    def hc(self, var):
        self._hc  = var

    @property
    def h_uv(self):
        '''
        depth at cell center, will return same value as hc
        '''
        if hasattr(self, 'h'):
            self._hc = np.mean(self.h[self.tri], axis = 1)
        else:
            raise InputError(f'- {self.filepath} does not contain depth info')
        return self._hc

    @h_uv.setter
    def h_uv(self, var):
        self._hc = var

    @property
    def siglayz(self):
        if not hasattr(self, '_siglayz'):
            self._siglayz = self.h[:,None]*self.siglay
        return self._siglayz

    @siglayz.setter 
    def siglayz(self, var):
        self._siglayz = var
    
    @property
    def siglevz(self):
        '''
        sigma level surfaces depth below mean sea surface at nodes
        '''
        if not hasattr(self, '_siglevz'):
            self._siglevz = self.h[:,None]*self.siglev
        return self._siglevz

    @siglevz.setter 
    def siglevz(self, var):
        self._siglevz = var

    @property
    def siglayz_uv(self):
        '''Siglay depths at cells'''
        if not hasattr(self, '_siglayz_uv'):
            self._siglayz_uv = np.mean(self.siglayz[self.tri,:], axis = 1)
        return self._siglayz_uv

    @siglayz_uv.setter
    def siglayz_uv(self,var):
        self._siglayz_uv = var

    @property
    def siglevz_uv(self):
        '''
        sigma level depth below mean sea surface at cells
        '''
        if not hasattr(self, '_siglevz_uv'):
            self._siglevz_uv = np.mean(self._siglevz[self.tri,:], axis = 1)
        return self._siglevz_uv

    @siglevz_uv.setter 
    def siglevz_uv(self, var):
        self._siglevz_uv = var

    # Functions we use to load relevant grid data to be used as instance attributes
    # ----------------------------------------------------------------------------------
    def _load_2dm(self):
        '''
        Grid parameters from a .2dm file
        '''
        self.tri, self.nodes, self.x, self.y, _, _, self.types \
                    = fvgrid.read_sms_mesh(self.filepath, nodestrings = True)

        ron = np.ndarray((1,len(self.types)), dtype = object)
        for n in range(len(self.types)):
            ron[0,n] = np.array([self.types[n]], dtype = np.int32)

        self.read_obc_nodes = ron
        self.num_obc = 0
        for i in range(len(self.read_obc_nodes[0,:])):
            self.num_obc += len(self.read_obc_nodes[0,i][0])

        self.xc = np.mean(self.x[self.tri], axis = 1)
        self.yc = np.mean(self.y[self.tri], axis = 1)
        self.lon,  self.lat  = self.Proj(self.x, self.y, inverse=True)
        self.lonc, self.latc = self.Proj(self.xc, self.yc, inverse=True)
        self.casename = self.filepath.split('.2dm')[0]

    def _load_nc(self):
        '''
        Grid parameters from .nc file
        '''
        self._add_nc_grid(['x', 'y', 'lon', 'lat', 'xc', 'yc', 'lonc', 'latc', 'h'])
        self._add_nc_grid(['nbe', 'nbve', 'nv', 'nbsn','ntsn', 'ntve'], grid = True, transpose = True)
        self._add_nc_grid(['art1', 'art2'])
        self._add_nc_grid(['siglev_center', 'siglay_center', 'siglev', 'siglay'], transpose = True)
        self._add_nc_grid(['obc_nodes', 'obc_type'], grid = True)
        self.__dict__['tri'] = self.__dict__.pop('nv')

        if hasattr(self, 'obc_nodes'):
            self._get_read_obc_nodes()

    def _load_mat(self):
        '''
        Grid parameters from .mat file (legacy)
        '''
        self._add_grid_parameters(['x', 'y', 'lon', 'lat', 'h', 'hraw', 'xc', 'yc','tri',\
                                  'siglayz','siglay','siglev','ntsn','nbsn','nObs',\
                                  'nObcNodes','obc_nodes','read_obc_nodes'])
        self.tri   = np.array(self.tri)-1
        self.nbsn -= 1
        self.read_obc_nodes -= 1

    def _load_npy(self):
        '''Grid parameters from M.npy file'''
        self._add_grid_parameters_npy(['x', 'y', 'lon', 'lat', 'h', 'h_raw', 'xc', 'yc','nv',\
                                      'siglay','siglev','ntsn','nbsn','read_obc_nodes',\
                                      'siglayz','siglevz', 'info', 'nbsn', 'ntsn', 'ts'])

    # Routines add data from input files to the instance
    # ----
    def _add_grid_parameters(self, names):
        '''
        Read grid attributes from matlab grid file and add them to the FVCOM_grid instance
        '''
        grid_mfile = loadmat(self.filepath)

        if type(names) is str:
            names=[names]

        for name in names:
            setattr(self, name, np.squeeze(grid_mfile['Mobj'][0,0][name]))

        self.casename = 'from_mat_file'

    def _add_nc_grid(self,
                    names,
                    grid      = False,
                    transpose = False):
        """
        Load ncdata from a fvcom-output formated file 

        Supports any file, both output and intended to force FVCOM.
        """
        data = Dataset(self.filepath)
        for name in names:
            try:
                if transpose:
                    if grid:
                        setattr(self, name, data[name][:].data.T-1)
                    else:
                        setattr(self, name, data[name][:].data.T)
                else:
                    if grid:
                        setattr(self, name, data[name][:].data-1)
                    else:
                        setattr(self, name, data[name][:].data)
            except:
                if verbose_g: print(f'- did not find {name} in {self.filepath})')

        # Add casename
        # ----
        file = self.filepath.split('/')[-1]
        if 'restart' in file:
            self.casename = file.split('_restart')[0]
        else:
            self.casename = file[:-8]

    def _add_grid_parameters_npy(self, names):
        '''
        Load grid information stored in a M.npy file
        '''
        grid = np.load(self.filepath, allow_pickle=True).item()
        for key in grid:
            if key == 'nv':
                setattr(self, 'tri', grid[key][:])

            elif type(grid[key]) == np.ndarray:
                setattr(self, key, grid[key])

            else:
                setattr(self, key, grid[key])

        if hasattr(self, 'info'):
            self.casename = self.info['casename']

    # Functions that interpret more information about the grid than explicitly stored in the grid files
    # ----------------------------------------------------------------------------------------------------
    def _get_proj(self, reference):
        '''
        return a Proj method
        '''
        self.reference = reference

        # Unless there is a reference from BuildCase in the info dict
        # ----
        if hasattr(self, 'info'):
            if 'reference' in self.info:
                self.reference = self.info['reference']

        return Proj(self.reference)

    def get_res(self):
        '''
        Return the grid resolution for each cell
        '''
        xpos = self.x[self.tri]
        ypos = self.y[self.tri]
        dx   = np.array([xpos[:,i]-xpos[:,(i+1)%2] for i in range(3)])
        dy   = np.array([ypos[:,i]-ypos[:,(i+1)%2] for i in range(3)])
        ds   = np.sqrt(dx**2 + dy**2)
        return np.mean(ds, axis = 0)

    def get_angles(self):
        '''
        Return the angles of corners in the grid cells
        '''
        # Triangle corners
        x     = self.x[self.tri]
        y     = self.y[self.tri]

        # COMPUTE ANGLES
        # cos(theta) = a * b / |a||b|
        AB    = np.array([x[:,0]-x[:,1], y[:,0]-y[:,1]])
        BC    = np.array([x[:,1]-x[:,2], y[:,1]-y[:,2]])
        CA    = np.array([x[:,2]-x[:,0], y[:,2]-y[:,0]])

        # Get length of each triangle side
        lAB   = np.sqrt(AB[0,:]**2+AB[1,:]**2)
        lBC   = np.sqrt(BC[0,:]**2+BC[1,:]**2)
        lCA   = np.sqrt(CA[0,:]**2+CA[1,:]**2)

        # Get dot products
        ABAC  = -(AB[0,:]*CA[0,:]+AB[1,:]*CA[1,:])
        BABC  = -(AB[0,:]*BC[0,:]+AB[1,:]*BC[1,:])
        CABC  = -(CA[0,:]*BC[0,:]+CA[1,:]*BC[1,:])

        # Get angles
        return np.array([np.arccos(ABAC/(lAB*lCA)), np.arccos(BABC/(lAB*lBC)), np.arccos(CABC/(lCA*lBC))])*(360/(2*np.pi))

    def _get_read_obc_nodes(self):
        '''
        Identify obc nodes, write them to read_obc_nodes (e.g. with nodestrings and everything)
        '''
        ron  = np.ndarray(100, dtype = object) # Temporary version of read_obc_nodes
        nobc = -1
        in_this_obc = False
        for i in range(len(self.obc_nodes)-1):
            if in_this_obc is False:
                nobc += 1
                ron[nobc] = np.array([], dtype = np.int32)
                if i == 0:
                    ron[nobc] = np.append(ron[nobc], self.obc_nodes[i])

            # Check if any triangle has both this and the next obc nodes
            tris        = self.tri[np.where(self.tri == self.obc_nodes[i])[0],:]
            in_this_obc = False
            for j, t in enumerate(tris):
                if self.obc_nodes[i+1] in t:
                    ron[nobc] = np.append(ron[nobc], self.obc_nodes[i+1])
                    in_this_obc = True

        self.read_obc_nodes = np.ndarray((1,nobc+1), dtype = object)
        for n in range(nobc+1):
            self.read_obc_nodes[0,n] = np.array([ron[n]])
        
    def cell2node(self,fieldin):
        '''
        Move data from cells to nodes
        '''
        nn = len(self.x)
        fieldout = np.zeros(nn)
        count = np.zeros(nn)
        for i in np.arange(len(self.xc)):
            n0 = self.tri[i,0]
            n1 = self.tri[i,1]
            n2 = self.tri[i,2]
            fieldout[n0] = fieldout[n0] + fieldin[i]
            count[n0]+=1
            fieldout[n1] = fieldout[n1] + fieldin[i]
            count[n1]+=1
            fieldout[n2] = fieldout[n2] + fieldin[i]
            count[n2]+=1
        fieldout = fieldout / count
        return fieldout

    def calculate_tri_area(self):
        '''
        Calculate triangle area

           C
          / \
         /___\
        A     B

        using the formula A = 0.5*|AB x AC|
        '''
        # Find AB and AC
        AB    = np.array([self.x[self.tri[:,1]]-self.x[self.tri[:,0]],
                          self.y[self.tri[:,1]]-self.y[self.tri[:,0]], \
                          np.zeros(len(self.xc))]).transpose()
        AC    = np.array([self.x[self.tri[:,2]]-self.x[self.tri[:,0]],
                          self.y[self.tri[:,2]]-self.y[self.tri[:,0]], \
                          np.zeros(len(self.xc))]).transpose()

        # Find their cross product
        ABxAC = np.cross(AB, AC)

        # Find the area
        tri_area  = 0.5*np.linalg.norm(ABxAC, axis=1)
        return tri_area

    # Find nearest node, check if point in polygon etc.
    # ---------------------------------------------------------------------------------------------------
    def find_nearest(self, x, y, grid = 'node'):
        '''
        Find indices of nearest node to given points.

        grid = 'node' or 'cell'
        '''
        if type(x) == int or type(x) == float or type(x) == np.float32:
            x = [x]
        if type(y) == int or type(y) == float or type(y) == np.float32:
            y = [y]

        indices = []
        if grid == 'node':
            for xi, yi in zip(x, y):
                ds = np.sqrt(np.square(self.x-xi) + np.square(self.y - yi))
                indices.append(ds.argmin())   

        elif grid == 'cell':
            for xi, yi in zip(x, y):
                ds = np.sqrt(np.square(self.xc-xi) + np.square(self.yc - yi))
                indices.append(ds.argmin())

        else:
            raise InputError(f'"{grid}" is not supported, choose between "node" and "cell"')

        return indices

    def isinside(self, x, y, x_buffer=0, y_buffer=0):
        '''
        Check if points are inside the rectangle bounded by the xy limits of the grid
        '''
        x_min = self.x.min() - x_buffer
        x_max = self.x.max() + x_buffer
        y_min = self.y.min() - y_buffer
        y_max = self.y.max() + y_buffer

        inside_x = np.logical_and(x>=x_min, x<=x_max)
        inside_y = np.logical_and(y>=y_min, y<=y_max)
        inside   = np.logical_and(inside_x, inside_y)

        return inside

    # Routines used to triangulate cell points
    # ----
    def cell_tri(self):
        '''
        Get triangulation connecting cell centres.

        Creates a triangulation for (xc,yc) points so that one can use plt.tricontourf to get a contour
        plot of velocity data (or other data stored on cells)
        '''
        if verbose_g: print('- Triangulating using Delaunays method')
        ctri  = tri.Triangulation(self.xc, self.yc)

        # Mask triangles covering land
        masked_ctris = self._mask_land_tris(ctri.triangles)

        return masked_ctris

    def _mask_land_tris(self, ctri):
        '''
        Mask triangles facing land
        '''
        # Figure out which elements connect to land
        if verbose_g: print(' - Masking triangles covering land')
        NV       = tge.check_nv(self.tri, self.x, self.y)
        NBE      = tge.get_NBE(len(self.xc), len(self.x), NV)
        ISBCE, _ = tge.get_BOUNDARY(len(self.xc), len(self.x), NBE, NV)

        # Illegal triangles are connected to land at all three vertices
        # --> ie. where sum of ISBCE for all triangles is > 3
        all_bce   = ISBCE[:][ctri].sum(axis=1)
        mask      = all_bce == 3
        self.ctri = ctri[mask==False]

        # Masked_tri:
        return ctri[mask==False]
        
    # Routines often used for plotting
    # -----------------------------------------------------------------------------------------
    def plot_grid(self, c = 'g-'):
        '''
        Plot mesh grid
        '''
        plt.triplot(self.x,
                    self.y,
                    self.tri,
                    c, markersize=0.2, linewidth=0.2)

        plt.axis('equal')
        plt.show(block = False)
        plt.pause(0.1) # to force it to be plotted on the go

    def plot_field(self, field):
        '''
        Plot scalar field on grid
        '''
        plt.tripcolor(self.x, self.y, self.tri, np.squeeze(field))
        plt.axis('equal')
        try:
            cb=plt.colorbar()
        except RuntimeError:
            if verbose_g: print('no colorrange, therefore no colorbar')
        plt.show(block = False)
        return cb

    def plot_cvs(self, field, 
                 cmap = 'jet', cmax = None, cmin = None, Norm = None,
                 edgecolor = 'face', verbose = True):
        '''
        Plots data (field) on control volume patches

        Optional:
        ----
        - edgecolor: Set to None (or a color, e.g. 'k') to show the interface between CVs
        - cmap:      set a colormap
        - cmax:      do not plot CVs with values greater than cmax
        - cmin:      do not plot CVs with values smaller than cmin
        - Norm:      to create a colorbar that is not linear (check matplotlib documentation)
        - verbose:   to get progress reports
        --> for cmax and cmin, the routine exclusively plots CVs
            with values within in that range
        '''
        # Load control volume patches
        # ----
        cv    = self._load_cvs(verbose)
        full  = True
        inds  = np.arange(0,len(field))

        # mask cvs with values below/above threshold
        # ------------------------------------
        if cmax is not None and cmin is None:
            inds         = np.where(field<=cmax)[0]; 
            full         = False

        elif cmin is not None and cmax is None:
            inds         = np.where(field>=cmin)[0]
            full         = False

        elif cmax is not None and cmin is not None:
            full         = False
            inds_smaller = np.where(field<=cmax)
            inds_bigger  = np.where(field>=cmin)
            inds         = np.array([[a for a in inds_bigger[0] if a in inds_smaller[0]]])[0]

        # plot the patches
        # ----
        if verbose: print('Plotting the patches')
        if full:
            collection   = PatchCollection(self.cv, cmap=cmap, edgecolor=edgecolor, norm = Norm)

        else:
            cv           = [self.cv[int(a)] for a in inds]
            collection   = PatchCollection(cv, cmap=cmap, edgecolor=edgecolor, norm = Norm)

        # Add values to the patches
        # ----
        collection.set_array(field[inds.astype(int)])
        ax               = plt.gca()
        _ret             = ax.add_collection(collection)

        # Center the axis over the area covered with patches and return
        # ----
        ax.autoscale_view(True)
        if _ret._A is not None: plt.gca()._sci(_ret)
        return _ret

    def _load_cvs(self, verbose):
        '''
        Check if we already have the cv-patches
        '''
        if hasattr(self,'cv'):
            if verbose:
                print('CVs already calculated.')
            return self.cv

        else:
            try:
                if verbose:
                    print('try to load cvs.npy from the working directory')
                cv        = np.load('cvs.npy', allow_pickle = True)
                self.cv   = cv
                if verbose:
                    print('sucess\n')

            except:
                if verbose:
                    print('\nFailure.')
                    print('We need to calculate the control volume edges for each node\n'+\
                          'to continue. This must only be done once for each M-instance.')
                cv       = self._get_cvs() # create CV polygons
        return cv

    def _get_cvs(self, nodes = None):
        '''
        Give nodenumber (as list) and this routine will return polygons for the control volume at each node.
        If no nodelist is given, it will calculate for all nodes in the domain

        Create a numpy file (cvs.npy) containing all of the control volumes to speed up the procedure
        the next time you run the code.
        '''
        if nodes is None:
            nodes = np.arange(0, len(self.x))

        # Initiate the progressbar to make the wait less painfull
        # ----
        widget = ['Finding control volume areas: ', pb.Percentage(), ' ', pb.BouncingBar()]
        bar    = pb.ProgressBar(widgets=widget, maxval=len(nodes))
        bar.start()

        # Find the positions that correspond to each controlvolume in the selected nodes
        # ----
        cv = []
        for i, node in enumerate(nodes):
            bar.update(i)
            # nodes surrounding node
            # -------
            nbsnhere = self.nbsn[node,np.where(self.nbsn[node,:]>=0)][0]

            # elements surrounding node
            # -------
            elemshere = self.nbve[node,np.where(self.nbve[node,:]>=0)][0]
            xmid      = (np.tile(self.x[node],len(nbsnhere))+self.x[nbsnhere])/2
            ymid      = (np.tile(self.y[node],len(nbsnhere))+self.y[nbsnhere])/2
            xcell     = self.xc[elemshere]; ycell     = self.yc[elemshere]

            # connect xc and yc to draw the control volume
            # ------
            xcv, ycv = self._connect_cv_walls(xmid, ymid, xcell, ycell, self.x, self.y, nbsnhere, elemshere, node)

            cv.append(mPolygon(np.array([xcv,ycv]).transpose(), True))

        bar.finish()
        self.cv = cv

        if verbose_g: print('Success! These cvs will be stored in your working directory as cvs.npy for future use')
        np.save('cvs.npy',cv)
        return cv

    # Some methods Frank developed to interpolate data to a z-level. Similar functionality as used in interpolate_to_z
    # ----
    def make_interpolation_matrices_TS(self, interpolation_depths=[-5]):
        ''' 
        Make matrices (numpy arrays) that, when multiplied with fvcom T or S matrix,
        interpolates data to a given depth.

        --> (adds a .interpolation_matrix_TS_{depth}_m to this instance)

        To interpolate data to a depth:
        ---
        FVCOM_grid.make_interpolation_matrices_TS(interpolation_depths = [-depth])
        out = np.sum(data * FVCOM_grid.interpolation_matrix_TS_{depth}_m, axis = 1)
        --> nan values will indicate that the grid point is below the depth
        '''
        for depth in interpolation_depths:
            interp_matrix = tools.make_interpolation_matrices(self.siglayz, depth)
            setattr(self, 'interpolation_matrix_TS_' + str(abs(int(depth))) + '_m', interp_matrix)

    def make_interpolation_matrices_uv(self, interpolation_depths=[-5]):
        '''
        Make matrices (numpy arrays) that, when multiplied with fvcom u or v matrix,
        interpolates data to a given depth. 

        --> (adds a .interpolation_matrix_uv_{depth}_m to this instance)

        To interpolate data to a depth:
        ---
        FVCOM_grid.make_interpolation_matrices_uv(interpolation_depths = [-depth])
        out = np.sum(data * FVCOM_grid.interpolation_matrix_uv_{depth}_m, axis = 1)
        --> nan values will indicate that the grid point is below the depth
        '''

        for depth in interpolation_depths:
            interp_matrix = tools.make_interpolation_matrices(self.siglayz_uv, depth)
            setattr(self, f'interpolation_matrix_uv_{abs(int(depth))}_m', interp_matrix)

    # To interpolate FVCOM sigma layer output to a z-level.
    # ----
    def interpolate_to_z(self, f, z, depths = None, verbose = False):
        '''
        Interpolate data to z-level
        - f:       datafield
        - z:       depth [m] to interpolate to f
        - depths:  depth of grid f. Positive downward.
        - verobse: report progress in routine

        Returns field at depth z. Will contain nans at z deeper than gridpoint.
        -------------------
        You need to mask the nans to plot using triconturf, eg:

        f      = M.interpolate_to_z(d['temp'][0,:,:], z)
        mask   = np.isnan(f[M.tri]).any(axis = 1)
        plt.tricontourf(M.x[:,0], M.y[:,0], M.tri, f, mask = mask)
        '''
        # Figure out which depth matrix this data belongs to
        # ----
        if depths is None:
            if f.shape == self.siglev.T.shape:
                depths = -self.h[:,None]*self.siglev.T

            elif f.shape == self.siglev_center.T.shape:
                depths = -self.h_uv[:,None]*self.siglev_center.T

            elif f.shape == self.siglay.T.shape:
                depths = -self.h[:,None]*self.siglay.T

            elif f.shape == self.siglay_center.T.shape:
                depths = -self.h_uv[:,None]*self.siglay_center.T

            else:
                raise ValueError('Dimension does match siglev, siglay, siglev_center or siglay_center, perhaps you should transpose the input field?')

        if verbose:
            print('- find nearest top and bottom')

        depthsmax = np.copy(depths)
        depthsmin = np.copy(depths)
        depthmin  = depths-z

        # Find first index less than 0
        lt0       = depthmin<0

        # Find depth for index over and under
        depthsmax[np.where(lt0)]          = np.nan
        depthsmin[np.where(lt0 == False)] = np.nan

        depbot    = np.nanmin(depthsmax, axis = 0)
        deptop    = np.nanmax(depthsmin, axis = 0)

        # Extract neighboring data from field
        # ----
        if verbose:
            print('- extract data at nearby depths')
        f_t       = np.nanmax(np.where(depths == deptop, f, np.nan), axis = 0)
        f_b       = np.nanmax(np.where(depths == depbot, f, np.nan), axis = 0)

        # Interpolate to z
        # ----
        if verbose:
            print('- interpolate to depth')
        dz        = depbot-deptop
        dzt       = z-deptop
        dzb       = depbot-z

        zf = (f_t*dzb+f_b*dzt)/dz

        # Use nearest valid value if NaN at the surface
        # ----
        zf[np.where(np.isnan(f_t))] = f_b[np.where(np.isnan(f_t))]
        return zf

    # Identify the OBC and the coastline
    # ---------------------------------------------------------------------------------------------------------------
    def get_obc(self, debug = False):
        '''
        Returns the positions and IDs of the open boundary nodes
        - xn,yn
        - obcnodes

        Primarilly used by the routines dealing with nesting zones in some way.
        '''
        obcnodes   = []
        nobc       = self.read_obc_nodes.shape[1]
        self.x_obc   = []; self.y_obc = []
        self.lon_obc = []; self.lat_obc = []
        for n in range(nobc):
            nodes  = self.read_obc_nodes[0,n]
            obcnodes.extend(nodes)
            self.x_obc.append(self.x[obcnodes[n]])
            self.y_obc.append(self.y[obcnodes[n]])
            self.lon_obc.append(self.lon[obcnodes[n]])
            self.lat_obc.append(self.lat[obcnodes[n]])

        # Check if the OBC is circular
        # ----
        if nobc == 1:
            if self.x_obc[0][0] == self.x_obc[0][-1] and self.y_obc[0][0] == self.y_obc[0][-1]:
                if verbose_g: print('-- identified circular OBC')
                self.x_obc[0] = self.x_obc[0][:-1]
                self.y_obc[0] = self.y_obc[0][:-1]

        self.obcnodes = obcnodes
        if debug:
            plt.figure()
            plt.scatter(self.x,self.y)
            for i in range(len(self.x_obc)):
                plt.scatter(self.x_obc[i], self.y_obc[i], label='boundary')
            plt.legend()
            plt.axis('equal')
            plt.show()

        # To numpy array
        self.lon_obc = np.array(self.lon_obc, dtype=object)
        self.lat_obc = np.array(self.lat_obc, dtype=object)

    def get_coast(self, verbose = False):
        '''
        Return boundary- and island polygons
        ----
        Input:
        - verbose (default = False)
        --> tells the routine to print info about its progress, and plot coast polygons successively.

        Output: coast_node, n_pol
        - M.coast_node  - node index in mesh
        - M.n_pol       - polygon each coast_node belongs to

        The longest polygon is normally the outer boundary of the domain (land + obc)
        '''
        # Make sure that the triangles are ordered the clockwise
        # ----
        self._check_nv()

        if verbose:
            print('- Find neighbors to each triangle')

        neighbors         = tri.Triangulation(self.x, self.y, self.tri).neighbors

        # ----
        if verbose:
            print('- Identify nodes near land')

        identify          = np.where(neighbors == -1)  # Identify triangles that are adjacent to land
        boundary_elements = identify[0]                # Index of triangle next to land


        nodes_1       = identify[1]                    # Corner at land
        nodes_2       = (nodes_1+1)%3                  # Next corner at land (each node in triangulation is
                                                       # counted clockwise)

        boundary_nodes_1  = self.tri[boundary_elements, nodes_1] # Node if the first corner
        boundary_nodes_2  = self.tri[boundary_elements, nodes_2] # Node in the second corner
        boundary_nodes    = np.append(boundary_nodes_1,boundary_nodes_2) # Create a full node list
        boundary_nodes    = np.unique(boundary_nodes)  # To remove duplicates

        # Connect boundary nodes to segment
        nodes_from_tri    = self.tri[identify[0],:]

        # Send nodes in boundary_nodes to a list
        line_segment   = []
        for n in nodes_from_tri:
            here = []
            for m in n:
                if m in boundary_nodes:
                    here.append(m)
            line_segment.append(here)

        # Connect line_segments to return separate boundary/islands polygons
        # ----
        if verbose:
            print('- connect nodes near land to segments')
        node, n_pol = tools.coast_segments(line_segment)

        self.coast_node  = np.array(node)
        self.n_pol       = np.array(n_pol)

        # Show result
        # ----
        if verbose:
            self.plot_grid()
            unique = np.unique(self.n_pol)
            for pol in unique:
                nodes = self.coast_node[self.n_pol == pol]
                plt.plot(self.x[nodes], self.y[nodes], '.')

    # Write a new .2dm file
    # ----
    def write_2dm(self, name = None):
        '''
        Writes a 2dm file for this grid
        name: to be saved
        '''
        if name is None:
            name = self.casename

        if hasattr(self, 'read_obc_nodes'):
            fvgrid.write_2dm(self.x, self.y, self.tri, read_obc_nodes = self.read_obc_nodes, name = name, casename = name)

        else:
            fvgrid.write_2dm(self.x, self.y, self.tri, name = name, casename = name)

    # A laplacian filter that lets you smooth a field if you have the NBSN, NTSN fields
    # ----
    def smooth(self, field, SmoothFactor = 0.2, n = 5):
        '''
        Smooth node-data using a laplacian filter

        Parameters:
        -----------
        field:        Field to smooth
        SmoothFactor: The degree of smoothing
        '''
        print(f'Smoothing {n} times:')
        for n in range(n):
            field_smooth = np.copy(field)
            print(f'- round {n}')
            for node in np.arange(len(self.x)):
                nodes  = self.nbsn[node,:self.ntsn[node]]
                smooth = np.mean(field[nodes])
                field_smooth[node] = (1-SmoothFactor)*field[node] + SmoothFactor*smooth
            field = np.copy(field_smooth)

        return field

    # Subgrid crops the mesh to a user defined square (but with minor alterations, all geometries should be possible).
    # The primary advantage of this is to speed up plotting and grid-post processing.
    # ----
    def subgrid(self, xlim, ylim, full = False):
        '''
        Create indices that crop your FVCOM grid to a smaller, more managable version
    
        Input:
        ----
        xlim: x-limits of domain in same units as self.x
        ylim: y-limits of domain in same units as self.y
        full: Will update the all grid fields in the FVCOM_grid object with the new indices (default: False)
        
        Output:
        ----
        if: full = False: Adds cropped_nv, cropped_x, cropped_y, cropped_cells and cropped_nodes to the mesh instance.  
            full = True:  Update all cell and node fields, adds cropped_cells and cropped_nodes to the instance,
                          removes fields (other than tri and nv) that reference grid ids

        Nb! I hope to turn full = True to default some time in the future.
        '''
        # Find triangles within the scope and crop the triangulation
        # ----
        xm   = self.xc <= max(xlim); xp = self.xc >= min(xlim)
        xind = np.logical_and(xm, xp)

        ym   = self.yc <= max(ylim); yp = self.yc >= min(ylim)
        yind = np.logical_and(ym, yp)

        cells    = np.arange(len(self.xc))
        cell_ind = cells[np.logical_and(xind, yind)]
        nv       = self.tri[cell_ind,:]

        # Unique nodes
        # ----
        node_ind     = np.unique(nv)
        new_node_ind = np.arange(len(node_ind))
        all_nodes    = np.zeros((len(self.xc)), dtype = np.int32)
        all_nodes[node_ind] = new_node_ind

        self.cropped_nv = all_nodes[nv]
        self.cropped_x  = self.x[node_ind]
        self.cropped_y  = self.y[node_ind]
        self.cropped_cells = cell_ind
        self.cropped_nodes = node_ind

        # If doing a complete overhaul of the mesh info...
        # ----
        if full:
            node_shape = len(self.x); cell_shape = len(self.xc)
            for key in list(self.__dict__):
                # We don't want to remove reference to old grid indices, and we know the values of tri
                # ----
                if key in ['cropped_cells', 'cropped_nodes']:
                    continue

                if key == 'tri':
                    self.tri = np.copy(self.cropped_nv)
                    self.nv  = np.copy(self.cropped_nv)
                    continue

                # Some fields must be removed to avoid confusion
                # ----
                if key in ['nbe', 'nbve', 'nbsn', 'ntsn', 'ntve', 'art1', 'art2']:
                    delattr(self, key)

                # Check if this field has a shape
                # ----
                try: 
                    shape = getattr(self, key).shape
                except:
                    continue

                # Crop to only retain the triangles we need
                # ----
                if len(shape)>1:
                    if shape[0] == node_shape:
                        setattr(self, key, getattr(self, key)[self.cropped_nodes,:])

                    elif shape[0] == cell_shape:
                        setattr(self, key, getattr(self, key)[self.cropped_cells,:])

                    else:
                        delattr(self, key)

                else:
                    if shape[0] == node_shape:
                        setattr(self, key, getattr(self, key)[self.cropped_nodes])

                    elif shape[0] == cell_shape:
                        setattr(self, key, getattr(self, key)[self.cropped_cells])

                    else:
                        delattr(self, key)
            self.cropped_object = True

        # Store since qc_gif needs to know this stuff
        # ----
        self.xlim = xlim
        self.ylim = ylim

    def _check_nv(self):
        '''
        Based on code from stackoverflow:
        https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
        '''
        neg = self._test_nv(self.x,self.y,self.tri)

        if neg.size == 0:
            if verbose_g: print('- Triangulation from source file is anti-clockwise')

        elif neg.size > 0:
            if len(neg) != len(self.tri[:,0]):
                raise ValueError('The triangulation direction is inconsistent, are you sure these triangles are good?\n'+\
                                 ' TGE will not fix it for now, since I am a bit qurious to see if triangulations can have inconsistent directions.')

            self.tri  = np.array([self.tri[:,0], self.tri[:,2], self.tri[:,1]], dtype = np.int64).T
            neg = self._test_nv(self.x, self.y, self.tri)
            if neg.size > 0:
                raise ValueError('*Something* went wrong when trying to re-arrange the triangulation to clockwise direction :(')
            else:
                if verbose_g: print('- Triangulation from source was clockwise, it has now been re-arranged to be anti-clockwise')

    # To extract data on a transect from the mesh. prepare_section is used to initialize a transect.
    # get_section_data will initialize a transect the first round, but will otherwise just extract data along it.
    # ----
    def prepare_section(self, section_file = None, res = None, store_transect_img = False):
        '''
        Returns x, y points along a section for use in section-analysis
        '''
        graphical = False
        if section_file is not None:
            try:
                lon, lat = np.loadtxt(section_file, delimiter = ',', unpack = True)
            except:
                lon, lat = np.loadtxt(section_file, delimiter = ' ', unpack = True)

            if len(lon) == 2:
                x, y = self.Proj(lon, lat)
                if x[0] > x[1]: # Ensure that the westernmost point in the section is first.
                    x1 = x[1]; x2 = x[0]
                    y1 = y[1]; y2 = y[0]
                else:
                    x1 = x[0]; x2 = x[1]
                    y1 = y[0]; y2 = y[1]

                # equation for the straight line between points.
                a = (y2-y1) / (x2-x1) # slope
                b = y1 - a*x1 # intersection 

                x_section = np.array([np.round(x1), np.round(x2)])
                y_section = a*x_section + b

            else:
                x_section, y_section = self.Proj(lon, lat)

        else:
            print('- Graphical input of section')
            plt.figure()
            self.plot_grid()
            plt.title('Click where you want the section to go (minimum two points)')
            places = plt.ginput(timeout = -1, n = -1)
            plt.close()
            x_section = []; y_section = []
            for p in places:
                x_section.append(p[0])
                y_section.append(p[1])
            x_section = np.array(x_section)
            y_section = np.array(y_section)
            graphical = True

        # Create an even points spacing
        # ----
        self.x_sec = np.empty(0)
        self.y_sec = np.empty(0)

        if res is None:
            # Should most likely be replaced by something that is grid size dependent in a not too distant future
            dx  = np.diff(x_section)
            dy  = np.diff(y_section)
            S   = np.sum(np.sqrt(dx**2+dy**2))
            res = S/60

        for i in range(len(x_section)-1):
            dst        = np.sqrt((x_section[i+1]-x_section[i])**2+(y_section[i+1]-y_section[i])**2)
            npoints    = np.ceil(dst/res).astype(int)
            xtmp       = np.linspace(x_section[i], x_section[i+1], npoints)
            ytmp       = np.linspace(y_section[i], y_section[i+1], npoints)
            self.x_sec = np.append(self.x_sec, xtmp)
            self.y_sec = np.append(self.y_sec, ytmp)

        self.fresh_section = True

        if graphical:
            plt.scatter(self.x_sec, self.y_sec, s = 30, c = 'k', zorder = 10)
            plt.scatter(x_section[0], y_section[0], s = 60, c = 'g', zorder = 11, label = 'start')
            plt.scatter(x_section[-1], y_section[-1], s = 60, c = 'r', zorder = 11, label = 'stop')
            plt.legend()
            plt.title('Section')
            plt.draw()

        # Plot section
        # ----
        if store_transect_img:
            plt.figure()
            self.plot_grid()
            plt.scatter(self.x_sec, self.y_sec, s = 30, c = 'k', zorder = 10)
            plt.scatter(x_section[0], y_section[0], s = 60, c = 'g', zorder = 11, label = 'start')
            plt.scatter(x_section[-1], y_section[-1], s = 60, c = 'r', zorder = 11, label = 'stop')
            plt.legend()
            plt.title('Section')
            plt.axis('equal')
            plt.savefig('Section_map.png')
            plt.close()


    def get_section_data(self, data,
                         section_file = None, res = None,
                         store_transect_img = False):
        '''
        dump data from from the full solution to an interpolated transect  (ie. not true node/cell data)

        Input:
        -----
        - data:         Field you want to get a transect of. Must be from a specific timestep
        - section_file: File with lon lat coordinate positions of segment fixed points. You can either use space or , as separator
        - res:          Resolution of the transect in m (defaults to 1/100 of span)
        - store_transect_img: Store a plot showing the transect overlayd on the grid

        Output:
        -----
        - dictionary containing:
          - x,y  positions of transect
          - h:   depth of transect positions)
          - dst: distance along transect
          - transect of field you gave the routine
        '''
        # Check that input dimensions are compatible with what the routine expects
        # ----
        if len(data.shape) == 2:
            # Force the grid to be of the same shape as sigma-coordinates
            if data.shape[1] != self.siglev.shape[1] and data.shape[1] != self.siglay.shape[1]:
                data = data.T

            # Thereafter see if we are dealing with sigma or siglev
            if data.shape[1] == self.siglay.shape[1]:
                sigma = 'siglay'

            elif data.shape[1] == self.siglev.shape[1]:
                sigma = 'siglev'

            elif data.shape[0] == 1:
                data  = data[0,:]
                sigma = 'siglev'

            else:
                raise InputError('Your data somehow does not match with siglev or siglay dimensions.')

        elif len(data.shape) == 3:
            raise InputError('Sorry, field can only be provided in 2D arrays or 1D arrays, time must be looped over externally.')

        else:
            sigma = 'siglev'

        # Thereafter identify if we have node or cell data
        # ----
        if data.shape[0] == len(self.x):
            grid = ''

        elif data.shape[0] == len(self.xc.shape):
            grid = 'c'
            # Then we also need cell triangles
            if not hasattr(self, 'ctri'):
                self.cell_tri()
        else:
            raise InputError('Your data somehow does not match with node or cell dimensions. Do the grid and data dimensions match?')

        # Just for the first call. Can be changed by calling prepare_section separately
        if not hasattr(self, 'x_sec') and not hasattr(self, 'y_sec'):
            self.prepare_section(section_file = section_file, res = res, store_transect_img = store_transect_img)

        if not hasattr(self, 'trs') or self.fresh_section:
            self.trs     =  tri.Triangulation(getattr(self, 'x'+grid), getattr(self, 'y'+grid), triangles = getattr(self, grid+'tri'))
            if grid == '':
                self.dpt_sec = tri.LinearTriInterpolator(self.trs, getattr(self, 'h')).__call__(self.x_sec,  self.y_sec)[:, None]*getattr(self, sigma)[0,:]
            else:
                self.dpt_sec = tri.LinearTriInterpolator(self.trs, getattr(self, 'hc')).__call__(self.x_sec, self.y_sec)[:, None]*getattr(self, sigma)[0,:]

        # Interpolate data to the section
        # ----
        out = {}
        if len(data.shape)>1:
            out['transect'] = self._interpolate_3D(data)
        else:
            out['transect'] = self._interpolate_2D(data)

        # Store depth and positions
        out['h'] = self.dpt_sec
        out['x'] = self.x_sec
        out['y'] = self.y_sec

        # Store transect distance
        dx = np.diff(self.x_sec, prepend = self.x_sec[0]) # since we count starting with the first segment position
        dy = np.diff(self.y_sec, prepend = self.y_sec[0])
        dst_seg = np.sqrt(dx**2 + dy**2)
        if len(data.shape)>1:
            out['dst'] = np.repeat(np.cumsum(dst_seg)[:,None], data.shape[1], axis = 1)
        else:
            out['dst'] = np.cumsum(dst_seg)
        return out

    def _interpolate_2D(self, field):
        '''
        Interpolate 2D data
        '''
        out_field = tri.LinearTriInterpolator(self.trs, field).__call__(self.x_sec, self.y_sec)
        return out_field

    def _interpolate_3D(self, field):
        out_field = np.zeros((len(self.x_sec), field.shape[1]))
        for sigma in range(field.shape[1]):
            out_field[:, sigma] = tri.LinearTriInterpolator(self.trs, field[:, sigma]).__call__(self.x_sec, self.y_sec)
        return out_field

    def get_coriolis(self, cell = False):
        '''
        Returns coriolis parameter at grid points
        '''
        if cell:
            self.f = np.sin(self.latc)*4*np.pi/(24*60*60)
        else:
            self.f = np.sin(self.lat)*4*np.pi/(24*60*60)

    # Read data from .dat files (so far just for *_sigma.dat?)
    # ----
    def get_kb(self, sigmafile=None):
        '''Reads kb from the input/casename_sigma.dat file'''
        self.kb = self._get_kb(sigmafile)

    # Static methods
    # ----------------------------------------------------------------------------
    @staticmethod
    def _test_nv(x,y,nv):
        # Triangle corners
        xpts  = x[nv]
        ypts  = y[nv]

        # Edges
        e1    = (xpts[:,1]-xpts[:,0])*(ypts[:,1]+ypts[:,0])
        e2    = (xpts[:,2]-xpts[:,1])*(ypts[:,2]+ypts[:,1])
        e3    = (xpts[:,0]-xpts[:,2])*(ypts[:,0]+ypts[:,2])

        # Direction test
        loop  = e1+e2+e3
        neg   = np.where(loop > 0)[0]
        return neg

    @staticmethod
    def _get_kb(sigmafile=None):
        '''Reads kb from the input/casename_sigma.dat file'''
        if sigmafile is None:
            casestr   = tools.get_cstr()
            sigmafile = os.path.join('input', casestr + '_sigma.dat')

        fid = open(sigmafile, 'r')
        for line in fid:
            if 'NUMBER OF SIGMA LEVELS' in line:
                kb = int(line.split('=')[-1].rstrip('\r\n'))

        fid.close()
        return kb

    @staticmethod
    @njit()
    def _connect_cv_walls(xmid, ymid, xcell, ycell, x, y, nbsnhere, elemshere, node):
        xcv = []; ycv = []
        for i in range(len(xmid)):
            xcv.append(xmid[i])
            ycv.append(ymid[i])

            # Find direction
            # ---
            boundary = False
            if i < len(xmid)-1:
                xcvmid = (xmid[i]+xmid[i+1])/2
                ycvmid = (ymid[i]+ymid[i+1])/2
                dst    = np.sqrt((xcell-xcvmid)**2+(ycell-ycvmid)**2)

                # Hack to be aware of land
                # ----
                ind    = np.argwhere(dst==dst.min())[0]

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

class NEST_grid():
    '''
    Object containing information about the FVCOM grid
    '''
    def __init__(self, path_to_nest, M = None, proj='epsg:32633'):
        """
        Reads ngrd.* file. Converts it to general format.
        """

        self.filepath = path_to_nest
        self.Proj     = Proj(proj)

        if self.filepath[-3:] == 'mat':
            self.add_grid_parameters_mat(['xn', 'yn', 'h', 'nv', 'fvangle', 'xc',
                                          'yc', 'R', 'nid', 'cid', 'oend1', 'oend2'])

            self.lonc, self.latc = self.Proj(self.xc, self.yc, inverse=True)
            self.lonn, self.latn = self.Proj(self.xn, self.yn, inverse=True)
            self.nv = self.nv

        elif self.filepath[-3:] == 'npy':
            self.add_grid_parameters_npy(['xn','yn','nv','xc','yc','lonn',
                                          'latn','lonc','latc','nid','cid'], add = True)
            self.add_grid_parameters_npy(['nv'], add = False)

            # The width of nestingzone
            try:
                self.oend1 = nest.item()['oend1'] # Matlab legacy
                self.oend2 = nest.item()['oend2'] # Matlab legacy
                self.R     = [[nest.item()['R']]] # Matlab legacy

            except:
                print('detected fvcom-fvcom nesting')

            # Read depth info
            try:
                self.add_grid_parameters_npy(['h_mother', 'hc_mother', 'siglev_mother', 'siglay_mother',
                                              'siglev_center_mother', 'siglay_center_mother', 'siglayz_mother',
                                              'siglayz_uv_mother'], add = False)
                self.add_grid_parameters_npy(['h', 'hc', 'siglev', 'siglay','siglev_center', 'siglay_center',
                                              'siglayz', 'siglayz_uv'], add = False)
            except:
                print('detected roms-fvcom nesting')

        # Add information from full fvcom grid (M), vertical coords and OBS-nodes
        # ----
        if M is not None:
            self.siglay  = M.siglay[:len(self.xn), :]
            self.siglev  = M.siglev[:len(self.xn), :]
            self.siglayz = M.siglayz[:len(self.xn), :]

            self.siglay_center = (
                self.siglay[self.nv[:,0], :]
                + self.siglay[self.nv[:,1], :]
                + self.siglay[self.nv[:,2], :]
            )/3

            self.siglev_center = (
                self.siglev[self.nv[:,0], :]
                + self.siglev[self.nv[:,1], :]
                + self.siglev[self.nv[:,2], :]
            )/3

            self.calcWeights(M)

    def add_grid_parameters_mat(self, names):
        '''
        Read grid attributes from mfile and add them to FVCOM_grid object
        '''
        grid_mfile = loadmat(self.filepath)

        if type(names) is str:
            names=[names]

        for name in names:
            setattr(self, name, grid_mfile['ngrd'][0,0][name])

        # Translate Matlab indexing to python
        self.nid = self.nid -1
        self.cid = self.cid-1
        self.nv  = self.nv-1

    def add_grid_parameters_npy(self, names, add=None):
        if add is None:
            raise ValueError('For the moment, you must explicitly say wether or not you want to add a dimension')
        nest = np.load(self.filepath, allow_pickle=True)
        for key in names:
            if add:
                setattr(self, key, nest.item()[key][:,None])
            else:
                setattr(self, key, nest.item()[key])

        # Parameters
        # Discussion: I will not change nid (as it is not used) nor will i change
        #             cid, as it seems to have been wrongly used in the original
        #             input from matlab-implementation of this routine.

    def calcWeights(self, M, w1=2.5e-4, w2=2.5e-5):
        '''
        Calculates linear weights in the nesting zone from weight = w1 at the obc to
        w2 the inner end of the nesting zone. At the obc nodes, weights equals 1

        By default (matlab legacy):
        w1  = 2.5e-4
        w2  = 2.5e-5

        This routine differs from the matlab sibling since the matlab version
        didn't work well for grids with several obcs.

        The ROMS model will be weighted less near the land than elsewhere (except
        at the outermost obc-row)
        '''
        M.get_obc()

        # Find the max radius- and node distance vector
        # ----
        if self.oend1 == 1:
            for n in range(len(M.x_obc)):
                dist       = np.sqrt((M.x_obc[n]-M.x_obc[n][0])**2+(M.y_obc[n] - M.y_obc[n][0])**2)
                i          = np.where(dist>self.R[0][0])
                M.x_obc[n] = M.x_obc[n][i]
                M.y_obc[n] = M.y_obc[n][i]

        if self.oend2 == 1:
            for n in range(len(M.x_obc)):
                dist       = np.sqrt((M.x_obc[n]-M.x_obc[n][-1])**2+(M.y_obc[n]-M.y_obc[n][-1])**2)
                i          = np.where(dist>self.R[0][0])
                M.x_obc[n] = M.x_obc[n][i]
                M.y_obc[n] = M.y_obc[n][i]

        # Find the distances between the nesting zone and the obc
        # ----
        # 1. Gather the obc nodes in one vector
        xo = []; yo = []
        for n in range(len(M.x_obc)):
            xo.extend(M.x_obc[n])
            yo.extend(M.y_obc[n])

        R = []; d_node = []
        for n in range(len(self.xn)):
            d_node.append(np.min(np.sqrt((xo-self.xn[n])**2+(yo-self.yn[n])**2)))

        R = max(d_node)

        # Define the interpolation values
        # ----
        distance_range = [0,R]
        weight_range   = [w1,w2]

        # Do the same for the cell values
        # ----
        d_cell = []
        for n in range(len(self.xc)):
            d_cell.append(min(np.sqrt((xo-self.xc[n])**2+(yo-self.yc[n])**2)))

        # Estimate the weight coefficients
        # ==> Kan det hende at disse må være lik for vektor og skalar?
        # ----
        weight_node = np.interp(d_node, distance_range, weight_range)
        weight_cell = np.interp(d_cell, distance_range, weight_range)

        if np.argwhere(weight_node<0).size != 0:
            weight_node[np.where(weight_node)]=min(weight_range)

        if np.argwhere(weight_cell<0).size != 0:
            weight_cell[np.where(weight_cell)]=min(weight_range)

        # ======================================================================================
        # The weights are calculated, now we need to overwrite some of them to get a full row of
        # forced values
        # ======================================================================================
        # Force the weight at the open boundary to be 1
        # ----
        # 1. reload the full obc
        M.get_obc()
        for n in range(len(M.x_obc)):
            xo.extend(M.x_obc[n])
            yo.extend(M.y_obc[n])

        # 2. Find the nesting nodes on the boundary
        node_obc_in_nest = [];
        for x,y in zip(xo,yo):
            dst_node = np.sqrt((self.xn-x)**2+(self.yn-y)**2)
            node_obc_in_nest.append(np.where(dst_node==dst_node.min())[0][0])

        # 3. Find the cells connected to these nodes
        cell_obc_in_nest = []
        nv               = self.nv
        nest_nodes       = np.array([node_obc_in_nest[:-1],\
                                     node_obc_in_nest[1:]]).transpose()

        # If you want a qube as the outer row
        # ===========================================================================
        for i, nest_pair in enumerate(nest_nodes):
            cells = [ind for ind, corners in enumerate(nv) if nest_pair[0] in \
                     corners or nest_pair[1] in corners]
            cell_obc_in_nest.extend(cells)

        cell_obc_to_one = np.unique(cell_obc_in_nest)

        # 4. Get all of the nodes in this list (builds the nodes one row outward)
        node_obc_to_one = np.unique(nv[cell_obc_to_one].ravel())

        # If you want triangles as the outer row
        # ---------------------------------------------------------------------------
        #for i, nest_pair in enumerate(nest_nodes):
        #    cells = [ind for ind, corners in enumerate(nv) if nest_pair[0] in \
        #             corners and nest_pair[1] in corners]
        #    cell_obc_in_nest.extend(cells)

        #cell_obc_to_one = np.unique(cell_obc_in_nest)
        #node_obc_to_one = np.unique(node_obc_in_nest)

        # 5. Finally force the weight at the outermost OBC-row
        weight_node[node_obc_to_one] = 1.0
        weight_cell[cell_obc_to_one] = 1.0

        #weight_node[node_obc_2ndrow] = 0.02
        #weight_cell[node_obc_2ndrow] = 0.02

        # --> Store everything in the NEST object
        # ----
        self.weight_node = weight_node
        self.weight_cell = weight_cell
        self.obc_nodes   = node_obc_to_one
        self.obc_cells   = cell_obc_to_one

class InputError(Exception):
    pass

class SomethingWentWrongError(Exception):
    pass