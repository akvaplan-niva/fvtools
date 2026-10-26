"""Coordinate inputs, geometry and derived grid properties."""

import numpy as np
from pyproj import Proj


class InputCoordinates:
    @property
    def tri(self):
        '''Triangulation (n, 3) connecting the nodes making up triangle n'''
        return self._tri

    @tri.setter
    def tri(self, var):
        self._tri = np.array(var)
        self._invalidate_spatial_cache()

    def _invalidate_spatial_cache(self):
        """Discard geometry-dependent lookups after coordinate/topology changes.

        Changes made *in place* (e.g. `M.x[:] = ...`) cannot be detected;
        assign through the property instead.
        """
        for name in ('node_tree', 'cell_tree', '_ctri', '_latc', '_lonc',
                     'boundary_nodes', 'coastline_nodes', 'cellstrings',
                     '_T', '_cv', '_art1', '_nbsn', '_ntsn', '_nbve', '_ntve',
                     '_nbe', '_nbse', '_nese', '_ISBCE', '_ISONB',
                     'trs', '_section_weights_cache', '_section_vector_basis_cache',
                     'mtri_nodes', 'mtri_cells'):
            self.__dict__.pop(name, None)

    @property
    def x(self):
        '''Node x position'''
        return self._x

    @x.setter
    def x(self, var):
        self._x = np.array(var)
        self._invalidate_spatial_cache()
    
    @property
    def y(self):
        '''Node y-position'''
        return self._y

    @y.setter
    def y(self, var):
        self._y = np.array(var)
        self._invalidate_spatial_cache()

    @property
    def lat(self):
        '''Node lat-position'''
        return self._lat
    
    @lat.setter
    def lat(self, var):
        self._lat = np.array(var)
        self.__dict__.pop('_latc', None)

    @property
    def lon(self):
        '''Node lon-position'''
        return self._lon
    
    @lon.setter
    def lon(self, var):
        self._lon = np.array(var)
        self.__dict__.pop('_lonc', None)

    @property
    def zeta(self):
        '''Sea surface elevation (must be set by user for each timestep)'''
        if not hasattr(self, '_zeta'): self._zeta = np.zeros(self.x.shape)
        return self._zeta
    
    @zeta.setter
    def zeta(self, var):
        self._zeta = np.array(var)

    @property
    def h(self):
        '''Bathymetric depth relative to mean sea level'''
        return self._h
    
    @h.setter
    def h(self, var):
        self._h = np.array(var)

    @property
    def siglev(self):
        '''Sigma levels - top and bottom interfaces of sigma layers'''
        return self._siglev

    @siglev.setter
    def siglev(self, var):
        self._siglev = np.array(var)

    @property
    def siglay(self):
        '''Sigma layer - centre position of sigma layer'''
        return self._siglay
    
    @siglay.setter
    def siglay(self, var):
        self._siglay = np.array(var)

class Coordinates:
    '''Coordinates we can derive from input-coordinates'''
    @property
    def latc(self):
        '''Latitude of triangle centres (cells/elements)'''
        if not hasattr(self, '_latc'): 
            self._latc = np.mean(self.lat[self.tri], axis = 1)
        return self._latc

    @latc.setter 
    def latc(self, var):
        self._latc = var

    @property
    def lonc(self):
        '''Longitude of triangle centres (cells/elements)'''
        if not hasattr(self, '_lonc'): 
            self._lonc = np.mean(self.lon[self.tri], axis = 1)
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
    def d(self):
        '''total water column depth at nodes'''
        return self.h + self.zeta

    # Properties dealing with the vertical coordinate
    @property
    def dc(self):
        '''total water column depth at cells'''
        return np.mean(self.d[self.tri], axis = 1)
    
    @property
    def siglayz(self):
        return self.d[:, None]*self.siglay

    @property
    def siglevz(self):
        '''sigma level surfaces depth below mean sea surface at nodes'''
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
            self.info['reference'] = projection
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
