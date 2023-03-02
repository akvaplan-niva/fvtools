# ---------------------------------------------------------------------
#       Create a file containing information about the nestgrid 
#       for fvcom2fvcom nesting
# ---------------------------------------------------------------------
import numpy as np
import sys
import fvtools.grid.fvgrid as fvg
import matplotlib.pyplot as plt
from pyproj import Proj, transform
from scipy.spatial import cKDTree as KDTree
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.gridding.prepare_inputfiles import write_FVCOM_bath

def main(mesh, fvcom_mother):
    '''
    Create a "ngrd.npy" file to be read by the routines setting up nest metrics

    Parameters:
    ----
    mesh:          M.mat, M.npy or for the experiment
    fvcom_mother:  M.mat, M.npy or output_xxxx.nc file from mother mesh

    Optional:
    ----
    casename:      will be added to _dep.dat. No need to specify if you use a
                   M.npy file
    '''
    print('\nCrop a nestingzone from the child mesh')
    M = FVCOM_grid(mesh)
    M_mother = FVCOM_grid(fvcom_mother, verbose = False)
    print(f'  Child model:  {M.casename}\n  Mother model: {M_mother.casename}')

    print('\nBuild the nest-mesh')
    NEST = build_nest_mesh(M)

    print('- Find nearest mesh points:')
    NEST['nid'], NEST['cid'] = nearest_mother(M_mother, NEST)

    print('- Copy data from the mother model to ngrd')
    NEST = add_depth_info(NEST, M_mother)

    print('\nMatch the bathymetry of the mother and child model')
    M, NEST = match_bathymetry(NEST, M_mother, M)
        
    # Show end result
    plt.figure()
    M_mother.plot_grid(c = 'b-', label = 'mother')
    M.plot_obc()
    plt.triplot(NEST['xn'], NEST['yn'], NEST['nv'], c = 'r', lw = 2, label = 'nest')
    plt.axis('equal')
    plt.legend(loc = 'upper right')
    plt.xlim([min(M.x), max(M.x)])
    plt.ylim([min(M.y), max(M.y)])
    
    plt.figure()
    plt.title('model bathymetry')
    M.plot_contour(M.h)
    plt.axis('equal')
        
    if M.reference == 'epsg:4326':
        M.write_bath(filename = f'{M.casename}_dep.dat', latlon = True)
    else:
        M.write_bath(filename = f'{M.casename}_dep.dat')

    print('Update M.npy with adjusted bathymetry, store nest grid to ngrd.npy')
    M.to_npy() # To also update the bathymetry in the numpy file
    np.save('ngrd.npy', NEST)
    print('- Wrote ngrd.npy')

    plt.show(block=False)

# ------------------------------------------------------------------------------------------------------------
#                                       Subroutines
# ------------------------------------------------------------------------------------------------------------
def nearest_mother(M_mother, NEST):
    '''
    Find common indices of the nest and computation mesh
    '''
    print('  - node')
    tree         = KDTree(np.array([M_mother.x, M_mother.y]).transpose())
    p, node_inds = tree.query(np.array([NEST['xn'], NEST['yn']]).transpose())
    print('  - cell')
    tree         = KDTree(np.array([M_mother.xc, M_mother.yc]).transpose())
    p, cell_inds = tree.query(np.array([NEST['xc'], NEST['yc']]).transpose())
    return node_inds.astype(int), cell_inds.astype(int)

def build_nest_mesh(M):
    '''
    We must crop the mesh, and return a new nv-array to fit the cropped mesh
    '''
    obc_elements = np.empty(0, dtype=int)

    # Identify triangles connected to the nesting-nodes
    for nodes in M.nodestrings:
        cells_at_obc = np.array([elem for elem, nv in enumerate(M.tri) if any([nv[i] in nodes for i in range(3)])])
        start, stop = nodes[[0,-1]]

        # First finds all triangles without endpoints in them. Thereafter finds triangles with an endpoint _and_ another obc-node in the triangle
        obc_identifiers = [i for (i, nv) in enumerate(M.tri[cells_at_obc, :]) if start not in nv and stop not in nv]
        obc_identifiers.extend([i for (i, nv) in enumerate(M.tri[cells_at_obc, :]) if start in nv and any([nv[i] in nodes and nv[(i+1)%3] in nodes for i in range(3)])])
        obc_identifiers.extend([i for (i, nv) in enumerate(M.tri[cells_at_obc, :]) if stop in nv and any([nv[i] in nodes and nv[(i+1)%3] in nodes for i in range(3)])])
        obc_identifiers = np.array(obc_identifiers)
        obc_elements = np.append(obc_elements, cells_at_obc[obc_identifiers.astype(int)])

    # Renumber the nodes so that we can make a triangulation independent of the mother mesh
    obc_tris     = M.tri[obc_elements,:]
    unique_nodes = np.unique(obc_tris.ravel())

    # Dump new mesh to a dict
    NEST = {}
    NEST['xn'], NEST['yn'] = M.x[unique_nodes], M.y[unique_nodes]
    obc_tree     = KDTree(np.array([NEST['xn'], NEST['yn']]).T)
    d, ind       = obc_tree.query(np.array([M.x, M.y]).T)

    # Rebuild nv, compute cell positions
    NEST['nv'] = np.array([ind[nv] for nv in obc_tris])
    NEST['xc'] = np.mean(NEST['xn'][NEST['nv']], axis = 1)
    NEST['yc'] = np.mean(NEST['yn'][NEST['nv']], axis = 1)
    NEST['lonn'], NEST['latn'] = M.Proj(NEST['xn'], NEST['yn'], inverse = True)
    NEST['lonc'], NEST['latc'] = M.Proj(NEST['xc'], NEST['yc'], inverse = True)
    return NEST

def add_depth_info(NEST, M_mother):
    '''
    Copy and paste depth info to the mother model
    '''
    NEST['h_mother']             = M_mother.h[NEST['nid'][:]]
    NEST['hc_mother']            = np.mean(M_mother.h[M_mother.tri], axis = 1)[NEST['cid']][:]
    NEST['siglev_mother']        = M_mother.siglev[NEST['nid'], :]
    NEST['siglay_mother']        = M_mother.siglay[NEST['nid'], :]
    NEST['siglev_center_mother'] = M_mother.siglev_c[NEST['cid'], :]
    NEST['siglay_center_mother'] = M_mother.siglay_c[NEST['cid'], :]
    NEST['siglayz_mother']       = M_mother.h[NEST['nid']][:, None]*NEST['siglay_mother']
    NEST['siglayz_uv_mother']    = NEST['hc_mother'][:, None]*NEST['siglay_center_mother']
    return NEST

def match_bathymetry(NEST,  M_mother, M):
    '''
    Important to ensure that mass conservation is actually met
    '''
    print('- Estimate obc mesh resolution')
    dst  = [sorted(np.sqrt((NEST['xn']-x)**2+(NEST['yn']-y)**2))[1] for x,y in zip(NEST['xn'], NEST['yn'])]
    r1, r2= 2*max(dst), 8*max(dst)

    print('- Find nodes in the experiment mesh near the OBC')
    M_distance_from_obc, _ = KDTree(np.array([NEST['xn'], NEST['yn']]).T).query(np.array([M.x, M.y]).T)

    # Identify nodes we will change the depth at
    transition_nodes = np.where(M_distance_from_obc<r2)[0]
    distance_to_obc_in_transitionzone = M_distance_from_obc[transition_nodes]

    # Find the mother nodes and depths corresponding to these
    print('- Find nodes in the mother grid corresponding to ')
    _, nearest_mother_to_transition = KDTree(np.array([M_mother.x, M_mother.y]).T).query(np.array([M.x[transition_nodes], M.y[transition_nodes]]).T)

    # Weight function in the transition zone
    weight  = np.zeros(len(transition_nodes))
    weight[np.where(distance_to_obc_in_transitionzone<=r1)[0]] = 1 # weight = 0 at nodes in/near the actual OBC
    transition            = np.where(np.logical_and(distance_to_obc_in_transitionzone>=r1, distance_to_obc_in_transitionzone<r2))[0]
    a                     = 1.0/(r1-r2)
    b                     = r2/(r2-r1)
    weight[transition]    = a*distance_to_obc_in_transitionzone[transition] + b

    # Compute the updated bathymetry
    print('- Store depth information')
    h_new     = np.copy(M.h)
    h_old = np.copy(h_new)
    h_new[transition_nodes] = M_mother.h[nearest_mother_to_transition] * weight + M.h[transition_nodes] *(1-weight)
    M.h = h_new

    #  - nodes
    _, nid          = KDTree(np.array([M.x, M.y]).T).query(np.array([NEST['xn'], NEST['yn']]).T)
    NEST['h']       = h_new[nid]
    NEST['siglev']  = M.siglev[nid, :]
    NEST['siglay']  = M.siglay[nid, :]
    NEST['siglayz'] = M.h[nid][:, None]*NEST['siglay']
    
    #  - cells
    _, cid                = KDTree(np.array([M.xc, M.yc]).T).query(np.array([NEST['xc'], NEST['yc']]).T)
    NEST['siglev_center'] = M.siglev_c[cid, :]
    NEST['siglay_center'] = M.siglay_c[cid, :]
    NEST['hc']            = np.mean(h_new[M.tri], axis = 1)[cid]
    NEST['siglayz_uv']    = NEST['hc'][:, None]*NEST['siglay_center']
    return M, NEST
