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
        """Order open and circular OBCs from actual mesh connectivity.

        Open chains contain their endpoints once. Closed chains have their
        first node repeated at the end, matching SMS 2DM nodestrings.

        Mesh-boundary edges are preferred. An interior circular chain can be
        inferred only if its OBC-induced mesh graph is unbranched (degree 2).
        Ambiguous interior loops require an explicitly ordered ``nodestrings``.
        """
        obc_nodes = np.unique(np.asarray(obc_nodes, dtype=np.int64))
        if obc_nodes.size == 0:
            return []
        if np.any(obc_nodes < 0) or np.any(obc_nodes >= self.node_number):
            raise ValueError("Invalid OBC node indices")

        is_obc = np.zeros(self.node_number, dtype=bool)
        is_obc[obc_nodes] = True

        boundary_edges = np.asarray(self.boundary_nodes, dtype=np.int64).reshape(-1, 2)
        obc_edges = boundary_edges[np.all(is_obc[boundary_edges], axis=1)]
        graph = nx.Graph()
        graph.add_nodes_from(map(int, obc_nodes))
        graph.add_edges_from(map(tuple, obc_edges))

        missing = np.asarray([n for n in obc_nodes if graph.degree(int(n)) == 0], dtype=np.int64)
        if missing.size:
            # A circular OBC may run along *interior* triangle edges rather
            # than along the exposed mesh perimeter. Only infer such a loop
            # when the induced graph is unambiguous; do not pick arbitrary
            # shortcuts across triangles.
            tri = np.asarray(self.tri, dtype=np.int64)
            all_edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
            interior_edges = all_edges[np.all(is_obc[all_edges], axis=1)]
            candidates = nx.Graph()
            candidates.add_nodes_from(map(int, obc_nodes))
            candidates.add_edges_from(map(tuple, interior_edges))
            missing_set = set(map(int, missing))
            for component in nx.connected_components(candidates):
                if not missing_set.intersection(component):
                    continue
                subgraph = candidates.subgraph(component)
                if len(component) < 3 or any(degree != 2 for _, degree in subgraph.degree()):
                    raise ValueError(
                        "Cannot uniquely order OBC nodes away from the mesh boundary. "
                        "For a circular interior OBC, assign explicitly ordered "
                        "M.nodestrings = [np.array([... , first_node])] "
                        "before writing the 2DM file."
                    )
                graph.add_edges_from(subgraph.edges())

        missing = [int(n) for n in obc_nodes if graph.degree(int(n)) == 0]
        if missing:
            raise ValueError(f"OBC nodes without a connected boundary edge: {missing}")

        nodestrings = []
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)
            branches = sorted(int(node) for node, degree in subgraph.degree() if degree > 2)
            if branches:
                raise ValueError(f"Branched OBC at nodes {branches}")

            endpoints = sorted(int(node) for node, degree in subgraph.degree() if degree == 1)
            if len(endpoints) == 2:
                # Preserve the legacy direction for open boundaries.
                path = nx.shortest_path(subgraph, source=endpoints[0], target=endpoints[1])
            elif len(endpoints) == 0 and len(component) >= 3:
                # A simple cycle: start at the smallest node and choose the
                # smaller of its two neighbours for deterministic orientation.
                first = min(component)
                previous = first
                current = min(subgraph.neighbors(first))
                path = [first, current]
                while current != first:
                    following = [int(node) for node in subgraph.neighbors(current)
                                 if node != previous]
                    if len(following) != 1 or len(path) > len(component):
                        raise ValueError("Invalid circular OBC connectivity")
                    previous, current = current, following[0]
                    path.append(current)
                if len(path) != len(component) + 1:
                    raise ValueError("Circular OBC does not traverse all selected nodes")
            else:
                raise ValueError("OBC must be an unbranched open chain or a closed loop")
            nodestrings.append(np.asarray(path, dtype=np.int64))

        nodestrings.sort(key=lambda path: int(path[0]))
        return nodestrings
