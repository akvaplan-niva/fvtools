"""FVCOM subgrid extraction, mesh remapping and cropped boundary helpers.

Internal implementation; CropGrid remains exported from fvcom_grd.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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

        from fvtools.grid.fvcom_grd import FVCOM_grid
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
