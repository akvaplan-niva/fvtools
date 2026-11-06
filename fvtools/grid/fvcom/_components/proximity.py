"""Spatial nearest-neighbour queries and mesh geometry checks."""

from functools import cached_property
import numpy as np
from scipy.spatial import cKDTree as KDTree


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
