from netCDF4 import Dataset
import numpy as np
import sys
from fvcom_pytools.grid.fvcom_grd import NEST_grid, FVCOM_grid

def main(nestfile, nestgrd, fvgrd, weights = [2.5e-4,1]):

    M = FVCOM_grid(fvgrd)
    N = NEST_grid(nestgrd)
    w1  = np.min(weights)
    w2  = np.max(weights)
    nobc = M.read_obc_nodes[0][0][0]
    M.x_obc = M.x[nobc][:, None]
    M.y_obc = M.y[nobc][:, None]

    xo = []; yo = []
    for n in range(len(M.x_obc)): 
        xo.extend(M.x_obc[n]) 
        yo.extend(M.y_obc[n]) 
        
    R = []; d_node = []
    for n in range(len(N.xn)):
        d_node.append(np.min(np.sqrt((xo-N.xn[n])**2+(yo-N.yn[n])**2)))

    R = max(d_node)

    # Define the interpolation values
    # ----
    distance_range = [0,R]
    weight_range   = [w2,w1]

    # Do the same for the cell values
    # ----
    d_cell = []
    for n in range(len(N.xc)):
        d_cell.append(min(np.sqrt((xo-N.xc[n])**2+(yo-N.yc[n])**2)))
        
    # Estimate the weight coefficients
    # ==> Kan det hende at disse må være lik for vektor og skalar?
    # ----
    wn = np.interp(d_node, distance_range, weight_range)
    wc = np.interp(d_cell, distance_range, weight_range)
        
    if np.argwhere(wn<0).size != 0:
        weight_node[np.where(wn)]=min(weight_range)
        
    if np.argwhere(wc<0).size != 0:
        weight_cell[np.where(wc)]=min(weight_range)

    # ======================================================================================
    # The weights are calculated, now we need to overwrite some of them to get a full row of
    # forced values
    # ======================================================================================
    # Force the weight at the open boundary to be 1
    # ----
    # 1. Find the nesting nodes on the boundary
    node_obc_in_nest = [];
    for x,y in zip(xo,yo):
        dst_node = np.sqrt((N.xn-x)**2+(N.yn-y)**2)
        node_obc_in_nest.append(np.where(dst_node==dst_node.min())[0][0])

    # 3. Find the cells connected to these nodes
    cell_obc_in_nest = []
    nv               = N.nv
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
    wn[node_obc_to_one] = 1.0
    wc[cell_obc_to_one] = 1.0

    # Load results into nestingfile
    nc = Dataset(nestfile, 'r+')
    nt = len(nc.variables['time'])
    weight_node = np.zeros((nt,len(wn)))
    weight_cell = np.zeros((nt,len(wc)))
    for tt in np.arange(nt):
        weight_cell[tt,:] = wc
        weight_node[tt,:] = wn
    nc.variables['weight_cell'][:] = weight_cell
    nc.variables['weight_node'][:] = weight_node
    nc.close()
