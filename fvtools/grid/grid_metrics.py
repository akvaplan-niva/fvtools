"""
Calculate stuff necessary to identify wether a node is on the boundary or not
"""

import numpy as np
from matplotlib.tri import Triangulation

def get_nbe(M):
    """
    Reads a FVCOM mesh object, returns the indices of nodes (and cells) connecting to land and to the OBC
   
    Returns:
    ----
    - nodes: dict with
             - boundary nodes
             - id (to identify if node is on land or to OBC)
    - cells: dict with
             - near_tri (elements surrounding elements)
             - boundary cells
             - id (to identify if cell is connecting to land or to OBC)
    """
    print('- Identify boundary nodes and cells...')
    print('  - build triangulation')
    tri      = Triangulation(M.x, M.y, triangles = M.tri)
    print('  - find neighbors')
    near_tri = tri.neighbors

    # Find triangles at the boundary
    print('  - find triangles near the model boundary')
    boundary_tri      = [i for i, tri in enumerate(near_tri) if -1 in tri]

    print('  - isolate the nodes and cells in the domain from those near the boundary')
    btri_neighbors    = near_tri[boundary_tri, :]
    identify          = np.where(near_tri == -1)
    boundary_elements = identify[0]
    nodes_1           = identify[1]
    nodes_2           = (nodes_1+1)%3
    boundary_nodes_1  = M.tri[boundary_elements, nodes_1]
    boundary_nodes_2  = M.tri[boundary_elements, nodes_2]
    boundary_nodes    = np.append(boundary_nodes_1,boundary_nodes_2)
    boundary_nodes    = np.unique(boundary_nodes)

    # Mark elements and nodes bording to land with 1, and 2 with obc
    # ----
    print('  - seperate land boundaries from the obc')
    boundary_nodes_id = np.ones(len(boundary_nodes))
    for nodestring in M.nodestrings:
        common, int1, int2 = np.intersect1d(nodestring, boundary_nodes, return_indices=True)
        boundary_nodes_id[int2] = 2

    boundary_elements_id = np.ones(len(boundary_elements))
    obc_nodes = boundary_nodes[np.where(boundary_nodes_id == 2)[0]]
    on_obc = np.array([i for i, elem in enumerate(boundary_elements) if M.tri[elem,0] in obc_nodes or M.tri[elem,1] in obc_nodes or M.tri[elem,2] in obc_nodes])
    boundary_elements_id[on_obc] = 2

    nodes = {}
    nodes['boundary'] = boundary_nodes.astype(int)
    nodes['id']       = boundary_nodes_id.astype(int)
    
    elements = {}
    elements['boundary'] = boundary_elements.astype(int)
    elements['id']       = boundary_elements_id.astype(int)
    elements['near tri'] = near_tri.astype(int)
    return nodes, elements


def get_surrounding_nodes(M):
    """
    Setup metrics for secondary connectivity (nodes surrounding nodes)
    - Alreday calculated if mesh info comes from M.npy
    """
    # Find triangle edges
    tri    = Triangulation(M.x, M.y, M.tri)
    edges  = tri.edges
    
    # Connect nodes around each triangle
    # ----------------------------------------------------------------
    # Allocate storage
    ntsn = np.zeros((len(M['x']))).astype(int)
    nbsn = -1*np.ones((len(M['x']),12)).astype(int)

    # This is a slow one, nice to keep track
    widget = ['Identiying nodes surrounding nodes: ', pb.Percentage(), ' ', pb.Bar()]
    bar = pb.ProgressBar(widgets=widget, maxval=len(edges[:,0]))
    bar.start()
    bar_count = 1
    for edge in edges:
        bar.update(bar_count)
        lmin = min(abs(nbsn[edge[0],:]-edge[1]))
        if lmin != 0:
            nbsn[edge[0], ntsn[edge[0]]] = edge[1]
            ntsn[edge[0]] += 1

        lmin = min(abs(nbsn[edge[1],:]-edge[0]))
        if lmin != 0:
            nbsn[edge[1], ntsn[edge[1]]] = edge[0]
            ntsn[edge[1]] += 1

        bar_count +=1

    bar.finish()

    return nbsn, ntsn, edges
