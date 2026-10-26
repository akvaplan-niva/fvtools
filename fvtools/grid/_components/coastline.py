"""Coastline polygons, mesh boundary geometry and coastal node identification.

Internal implementation; CoastLine remains public in fvcom_grd.
"""

from functools import cached_property

import matplotlib
import numpy as np


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
