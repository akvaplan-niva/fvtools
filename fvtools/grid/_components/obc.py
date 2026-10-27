"""Open-boundary nodes, connectivity-ordered nodestrings and cellstrings.

Internal implementation; OBC remains public in fvcom_grd.
"""

from functools import cached_property

import networkx as nx
import numpy as np


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
        # Invalidate derived boundaries when the selected OBC nodes change.
        # In-place modifications of the user-supplied list cannot be detected;
        # assign via this setter after changing the list.
        self.__dict__.pop('_nodestrings', None)
        self.__dict__.pop('cellstrings', None)
        self.__dict__.pop('coastline_nodes', None)

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
        self.__dict__.pop('cellstrings', None)
        self.__dict__.pop('coastline_nodes', None)

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
