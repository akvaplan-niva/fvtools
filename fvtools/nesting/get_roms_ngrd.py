# ---------------------------------------------------------------------
#       Create a file containing information about the nestgrid
# ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from fvtools.grid.fvcom_grd import FVCOM_grid
from pykdtree.kdtree import KDTree

def main(mesh, R = None):
    '''
    Create a "ngrd.npy" file to be read by the routines creating nesting files

    Parameters:
    dm_file: 
    '''
    print('Computing nestzone metrics')
    # Store the stuff we need to create a nestingfile in this dict
    M = FVCOM_grid(mesh)

    print(f'- Number of nodestrings: {len(M.nodestrings)}')

    # Adjust sides
    circular = False
    if len(M.nodestrings) == 1:
        if M.nodestrings[0][0] == M.nodestrings[0][-1]:
            print('-- this nest is circular')
            circular = True
        
    print('- Cut nestzone out of mesh')
    NEST = adjust_sides(M, circular, R)
    
    # Convert to get latlon
    print('- Projecting latlon')
    NEST['lonn'], NEST['latn'] = M.Proj(NEST['xn'], NEST['yn'], inverse = True)

    # Get cell values
    NEST['lonc'], NEST['latc'] = M.Proj(NEST['xc'], NEST['yc'], inverse = True)

    # Find corresponding indices (necessary for fvcom2fvcom)
    print('- Find nearest mesh points:')
    NEST['nid'] = M.find_nearest(NEST['xn'], NEST['yn'], grid = 'node')
    NEST['cid'] = M.find_nearest(NEST['xc'], NEST['yc'], grid = 'cell')

    # Save. (Creates a structure readable by roms_nesting and fvcom2fvcom nesting)
    NEST['oend1'] = 1; NEST['oend2'] = 1
    NEST['R'] = R
    NEST['info'] = {}
    NEST['info']['reference'] = M.info['reference']
    np.save('ngrd.npy', NEST)

    if R is not None:
        plt.figure()
        plt.triplot(NEST['xn'], NEST['yn'], NEST['nv'])
        plt.axis('equal')
        plt.show()

# ------------------------------------------------------------------------------------------------------------
#                                       Subroutines
# ------------------------------------------------------------------------------------------------------------
def adjust_sides(M, circular, R):
    '''
    Cut of the sides of the nestingzone
    '''
    nstrs = len(M.nodestrings)

    print('  -> Cropping obc nodes')
    x_obc = np.empty(0); 
    y_obc = np.empty(0)
    if circular:
        x_obc = M.x[M.nodestrings[0]]
        y_obc = M.y[M.nodestrings[0]]

    else:
        for i in range(nstrs):
            x_tmp = M.x[M.nodestrings[i]]
            y_tmp = M.y[M.nodestrings[i]]
        
            # Look at the sides one-by-one
            x_tmp, y_tmp = crop_obc(x_tmp, y_tmp, x_tmp[0], y_tmp[0], R)
            x_tmp, y_tmp = crop_obc(x_tmp, y_tmp, x_tmp[-1], y_tmp[-1], R)

            x_obc = np.append(x_obc, x_tmp)
            y_obc = np.append(y_obc, y_tmp)

    # Find cells within R from x_obc and y_obc
    NEST = crop_mesh(M, x_obc, y_obc, R)

    return NEST

def crop_obc(x_obc, y_obc, xcoast, ycoast, R):
    '''
    Make cropped nestingzone
    '''
    dist  = np.sqrt((x_obc-xcoast)**2+(y_obc-ycoast)**2)
    inds  = np.where(dist >= 0.895*R)[0]

    return x_obc[inds], y_obc[inds]

def crop_mesh(M, x_obc, y_obc, R):
    '''
    Get the triangles within R from the cropped obc
    '''
    print('  -> Find the necessary cells')
    obc_tree = KDTree(np.array([x_obc, y_obc]).T)
    dst, _ = obc_tree.query(np.array([M.xc, M.yc]).T, distance_upper_bound = R)

    # Cells within search range
    cells = np.where(dst <= R)[0]

    # Store the new nodes
    inds  = np.unique(M.tri[cells,:].ravel())
    x_new = M.x[inds]
    y_new = M.y[inds]

    # Create new nv structure
    print('  -> Create nest triangulation')
    new_nv = np.nan*np.ones((len(cells), 3), dtype = int)
    new_tree = KDTree(np.array([x_new, y_new]).T)

    # Find corresponding x,y in the cells corners
    cx = M.x[M.tri[cells,:]]
    cy = M.y[M.tri[cells,:]]

    for j in range(3):
        _, index = new_tree.query(np.array([cx[:,j], cy[:,j]]).T)
        new_nv[:,j] = index.astype(int)
    new_nv = new_nv.astype(int)

    # Overwrite return nest-dict
    dNEST = {}
    dNEST['xn'] = x_new
    dNEST['yn'] = y_new
    dNEST['nv'] = new_nv.astype(int)
    dNEST['xc'] = triangulate(dNEST,'xn')
    dNEST['yc'] = triangulate(dNEST,'yn')
    return dNEST

def triangulate(NEST, var):
    c = (NEST[var][NEST['nv'][:,0]] + NEST[var][NEST['nv'][:,1]] + NEST[var][NEST['nv'][:,2]])/3.0
    return c
