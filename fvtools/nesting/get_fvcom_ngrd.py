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

def main(mesh,
         fvcom_mother, 
         casename = None,
         debug    = True):
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
    debug:         True
    '''
    print('\nLoad mother and child grid')

    # Store the stuff we need to create a nestingfile in this dict
    # ----
    print('- child:  '+mesh)
    M_exp     = FVCOM_grid(mesh)

    print('- mother: '+fvcom_mother)
    M_mother  = FVCOM_grid(fvcom_mother, verbose = False)

    # FULL kdtree
    # ----
    FULL_tree = KDTree(np.array((M_exp.x, M_exp.y)).T)

    # Only keep data at the boundary in the first pass
    # ----
    print('\nBuild the nest-mesh')
    NEST      = build_nest_mesh(FULL_tree, M_exp)

    # Find corresponding indices (necessary for fvcom2fvcom)
    # ----
    print('- Find nearest mesh points:')
    NEST['nid'], NEST['cid'] = nearest_mother(M_mother, NEST)

    # Copy depth information from mother
    # ----
    print('- Add mother depth information to ngrd')
    NEST          = add_depth_info(NEST, M_mother)

    # Smooth topography in the nesting-zone
    # ----
    print('\nMatch the bathymetry of the mother and child model')
    M_exp.h, NEST = match_bathymetry(NEST, M_mother, M_exp, FULL_tree)
    
    # Write to .dat format
    # ----
    if casename is None:
        if mesh.split('.')[-1] == 'npy':
            M = np.load(mesh, allow_pickle = True)
            casename = './input/'+M.item()['info']['casename']
        else:
            casename = 'matched_bathymetry'
        
    fname = casename + '_dep.dat'
        
    # Plot some figures to indicate how big the success is
    # ----
    if debug:
        plt.figure()
        plt.triplot(M_mother.x, M_mother.y, M_mother.tri, c = 'g', lw = 0.2, label = 'mother')
        plt.triplot(M_exp.x, M_exp.y, M_exp.tri, c = 'b', lw = 0.2, label = 'child')
        plt.triplot(NEST['xn'], NEST['yn'], NEST['nv'], c = 'r', lw = 0.2, label = 'nest')
        plt.axis('equal')
        plt.legend(loc = 'upper right')
        plt.xlim([min(M_exp.x), max(M_exp.x)])
        plt.ylim([min(M_exp.y), max(M_exp.y)])
        
        plt.figure()
        plt.title('model bathymetry')
        plt.tricontourf(M_exp.x, M_exp.y, M_exp.tri, M_exp.h)
        plt.axis('equal')
        
    print(' ')
    write_FVCOM_bath(M_exp, filename = fname) #now just writing carthesian, can add lon/lat from M.info['reference'] if needbe
    np.save('ngrd.npy', NEST)

    plt.show()

# ------------------------------------------------------------------------------------------------------------
#                                       Subroutines
# ------------------------------------------------------------------------------------------------------------
def read_2dm(my2dm):
    '''
    Reads 2dm-file, returns basic grid info
    '''
    try:
        triangle, nodes, X, Y, Z, types, nstr = fvg.read_sms_mesh(my2dm, nodestrings=True)
    except ValueError:
        raise ValueError('Make sure to save the file with a nodestring')

    points    = np.array([X,Y]).transpose()
    return X, Y, triangle, nstr

def project(x, y, to):
    '''
    Project the positions from latlon to UTM33W
    (can be upgraded to other reference systems in the future
    '''
    WGS84    = Proj('epsg:4326')
    UTM33W   = Proj(proj='utm', zone = '33', ellps='WGS84')
    lon, lat = transform(UTM33W, WGS84, x, y, always_xy = True)
    return lon, lat

def nearest_mother(M,NEST):
    '''
    Find common indices of the nest and computation mesh
    '''
    # Store positions as point-arrays
    node_nst = np.array([NEST['xn'], NEST['yn']]).transpose()
    cell_nst = np.array([NEST['xc'], NEST['yc']]).transpose()

    node_me  = np.array([M.x, M.y]).transpose()
    cell_me  = np.array([M.xc, M.yc]).transpose()

    print('  -> node')
    tree              = KDTree(node_me)
    p,inds            = tree.query(node_nst)
    nearest_mesh_node = inds.astype(int)

    print('  -> cell')
    tree              = KDTree(cell_me)
    p,inds            = tree.query(cell_nst)
    nearest_mesh_cell = inds.astype(int)

    return nearest_mesh_node, nearest_mesh_cell

def triangulate(NEST,var):
    c = (NEST[var][NEST['nv'][:,0]] + NEST[var][NEST['nv'][:,1]] + NEST[var][NEST['nv'][:,2]])/3.0
    return c

def build_nest_mesh(FULL_tree, M_exp):
    '''
    We must crop the mesh, and return a new nv-array to fit the cropped mesh
    '''
    NEST = {}
    obc_nodes    = np.empty(0)
    obc_elements = np.empty(0)
    for i in range(M_exp.read_obc_nodes.shape[1]):
        nodes    = M_exp.read_obc_nodes[0,i][0]

        # find triangles in nest
        raw_elems = [elem for elem, nv in enumerate(M_exp.tri) if (nv[0] in nodes or nv[1] in nodes or nv[2] in nodes)]
        raw_nest  = M_exp.tri[raw_elems, :]

        # if end of nodestring, be strict
        end1 = nodes[0]
        end2 = nodes[-1]
        
        obc_tris = np.empty(0)
        obc_el = []
        for e, nv in enumerate(raw_nest):
            if end1 in nv:
                if nv[0] in nodes and nv[1] in nodes:
                    obc_tris = np.append(obc_tris,nv)
                    obc_el.append(e)
                elif nv[1] in nodes and nv[2] in nodes:
                    obc_tris = np.append(obc_tris,nv)
                    obc_el.append(e)
                elif nv[2] in nodes and nv[0] in nodes:
                    obc_tris = np.append(obc_tris,nv)
                    obc_el.append(e)

            elif end2 in nv:
                if nv[0] in nodes and nv[1] in nodes:
                    obc_tris = np.append(obc_tris,nv)
                    obc_el.append(e)
                elif nv[1] in nodes and nv[2] in nodes:
                    obc_tris = np.append(obc_tris,nv)
                    obc_el.append(e)
                elif nv[2] in nodes and nv[0] in nodes:
                    obc_tris = np.append(obc_tris,nv)
                    obc_el.append(e)
                
            else:
                obc_tris = np.append(obc_tris,nv)
                obc_el.append(e)

        # Store
        # ----
        nest_nodes   = obc_tris.ravel()
        obc_elements = np.append(obc_elements, np.array(raw_elems)[obc_el])

    # re-build the mesh
    # ----
    obc_elements = obc_elements.astype(int)
    obc_tris     = M_exp.tri[obc_elements,:]
    unique_nodes = np.unique(obc_tris.ravel())
    NEST['xn']   = M_exp.x[unique_nodes]
    NEST['yn']   = M_exp.y[unique_nodes]

    obc_tree     = KDTree(np.array([NEST['xn'], NEST['yn']]).T)
    d, ind       = obc_tree.query(np.array([M_exp.x, M_exp.y]).T)

    # Re-build nv
    # ----
    NEST['nv'] = []
    for nv in obc_tris:
        tri = [ind[nv[0]], ind[nv[1]], ind[nv[2]]]
        NEST['nv'].append(tri)
    NEST['nv'] = np.array(NEST['nv'])

    # Cell-positions
    # ----
    NEST['xc'] = triangulate(NEST,'xn')
    NEST['yc'] = triangulate(NEST,'yn')

    # Convert to get latlon
    # ----
    print('- Projecting latlon')
    NEST['lonn'], NEST['latn'] = project(NEST['xn'], NEST['yn'], M_exp.info['reference'])
    
    # Get cell values
    # ----
    NEST['lonc'], NEST['latc'] = project(NEST['xc'], NEST['yc'], M_exp.info['reference'])

    return NEST

def add_depth_info(NEST, M):
    '''
    Copy and paste depth info to the mother model
    '''
    # Mother grid
    NEST['h_mother']             = M.h[NEST['nid'][:]]
    hc                           = np.mean(M.h[M.tri], axis = 1)
    NEST['hc_mother']            = hc[NEST['cid']][:,None]
    NEST['siglev_mother']        = M.siglev[NEST['nid'], :]
    NEST['siglay_mother']        = M.siglay[NEST['nid'], :]
    NEST['siglev_center_mother'] = M.siglev_c[NEST['cid'], :]
    NEST['siglay_center_mother'] = M.siglay_c[NEST['cid'], :]
    NEST['siglayz_mother']       = M.h[NEST['nid']][:,None]*NEST['siglay_mother']
    NEST['siglayz_uv_mother']    = NEST['hc_mother']*NEST['siglay_center_mother']
    
    return NEST

def match_bathymetry(NEST,  M_mother, M_exp, FULL_tree):
    '''
    Important to ensure that mass conservation is actually met
    '''
    print('- Estimate obc mesh resolution')
    dst  = [sorted(np.sqrt((NEST['xn']-x)**2+(NEST['yn']-y)**2))[1] for x,y in zip(NEST['xn'], NEST['yn'])]
    r1   = 2*max(dst)
    r2   = 8*max(dst)

    # 1. find nodes within a distance from the nestingzone
    print('- Find nodes in the experiment mesh near the OBC')
    obc_tree         = KDTree(np.array([NEST['xn'], NEST['yn']]).T)
    d,i_exp          = obc_tree.query(np.array([M_exp.x, M_exp.y]).T)
    transition_nodes = np.where(d<r2)[0]
    d_transition     = d[transition_nodes]
    ht_exp           = M_exp.h[transition_nodes]

    # 2. Find the mother nodes and depths corresponding to these
    print('- Find nodes in the mother grid corresponding to ')
    transition_tree  = KDTree(np.array([M_mother.x,M_mother.y]).T)
    d, i_mother      = transition_tree.query(np.array([M_exp.x[transition_nodes], M_exp.y[transition_nodes]]).T)
    ht_mother        = M_mother.h[i_mother]

    # 3. Define weight function
    weight                = np.zeros(len(ht_exp))
    weight[np.where(d_transition<=r1)[0]] = 1
    transition            = np.where(np.logical_and(d_transition>=r1, d_transition<r2))[0]
    a                     = 1.0/(r1-r2)
    b                     = r2/(r2-r1)
    weight[transition]    = a*d_transition[transition]+b

    # 4. Compute the updated bathymetry
    print('- Store depth information')
    h_updated = ht_mother * weight + ht_exp *(1-weight)
    h_old     = np.copy(M_exp.h)
    h_new     = np.copy(M_exp.h)
    h_new[transition_nodes] = h_updated
    
    # 5. Return depth info to FVCOM nest
    #    - nodes
    node_tree             = KDTree(np.array([M_exp.x, M_exp.y]).T)
    d, nid                = node_tree.query(np.array([NEST['xn'], NEST['yn']]).T)
    NEST['h']             = h_new[nid]
    NEST['siglev']        = M_exp.siglev[nid, :]
    NEST['siglay']        = M_exp.siglay[nid, :]
    NEST['siglayz']       = M_exp.h[nid][:,None]*NEST['siglay']
    
    #    - cells
    cell_tree             = KDTree(np.array([M_exp.xc, M_exp.yc]).T)
    d, cid                = cell_tree.query(np.array([NEST['xc'], NEST['yc']]).T)

    NEST['siglev_center'] = M_exp.siglev_c[cid, :]
    NEST['siglay_center'] = M_exp.siglay_c[cid, :]
    NEST['hc']            = np.mean(h_new[M_exp.tri], axis = 1)[cid]
    NEST['siglayz_uv']    = NEST['hc'][:, None]*NEST['siglay_center']

    return h_new, NEST
