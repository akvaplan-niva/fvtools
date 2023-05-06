# ---------------------------------------------------------------------
#       Create a file containing information about the nestgrid
# ---------------------------------------------------------------------
import numpy as np
import sys
import fvtools.grid.fvgrid as fvg
import matplotlib.pyplot as plt
from pyproj import Proj, transform
from fvtools.grid.fvcom_grd import FVCOM_grid
from scipy.spatial import cKDTree as KDTree

def main(mesh, 
         R = None):
    '''
    Create a "ngrd.npy" file to be read by the routines creating nesting files

    Parameters:
    dm_file: 
    '''
    print('Computing nestzone metrics')
    # Store the stuff we need to create a nestingfile in this dict
    # ----
    M    = FVCOM_grid(mesh)
    M.R  = R
    FULL = {}

    # Store the full grid for later
    # ----
    FULL['xn'] = np.copy(M.x)
    FULL['yn'] = np.copy(M.y)
    FULL['nv'] = np.copy(M.tri)
    FULL['xc'] = triangulate(FULL,'xn')
    FULL['yc'] = triangulate(FULL,'yn')

    print(f'- Number of nodestrings: {len(M.nodestrings)}')
    # "Cut corners" on the solid sides of the nestingzone
    # ----
    M.oend1 = 1; M.oend2 = 1

    # Adjust sides
    # ----
    circular = False
    if len(M.nodestrings) == 1:
        if M.nodestrings[0][0] == M.nodestrings[0][-1]:
            print('-- this nest is circular')
            circular = True
        
    print('- Cut nestzone out of mesh')
    NEST = adjust_sides(M, circular)
    
    # Convert to get latlon
    # ----
    print('- Projecting latlon')
    NEST['lonn'], NEST['latn'] = project(NEST['xn'], NEST['yn'], to = M.info['reference'])
    
    # Get cell values
    # ----
    NEST['lonc'], NEST['latc'] = project(NEST['xc'], NEST['yc'], to = M.info['reference'])

    # Find corresponding indices (necessary for fvcom2fvcom)
    # ----
    print('- Find nearest mesh points:')
    NEST['nid'], NEST['cid'] = nearest_mother(FULL, NEST)

    # Save. (Creates a structure readable by roms_nesting and fvcom2fvcom nesting)
    # ----
    NEST['oend1'] = 1; NEST['oend2'] = 1
    NEST['R'] = R
    NEST['info'] = {}
    NEST['info']['reference'] = M.info['reference']
    np.save('ngrd.npy',NEST)

    if R is not None:
        plt.figure()
        plt.triplot(NEST['xn'], NEST['yn'], NEST['nv'])
        plt.axis('equal')
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

def project(x, y, to = 'UTM33W'):
    '''
    Project the positions from latlon to UTM33W
    (can be upgraded to other reference systems in the future
    '''
    WGS84    = Proj('epsg:4326')
    UTM33W   = Proj(proj='utm', zone = '33', ellps='WGS84')
    lon, lat = transform(UTM33W, WGS84, x, y, always_xy = True)
    return lon, lat

def nearest_mother(FULL,NEST):
    '''
    Find common indices of the nest and computation mesh
    '''
    # Get coordinates
    x_me     = FULL['xn'];  y_me  = FULL['yn']
    x_nst    = NEST['xn'];  y_nst = NEST['yn']

    xc_me    = FULL['xc']; yc_me  = FULL['yc']
    xc_nst   = NEST['xc']; yc_nst = NEST['yc']

    # Store as point-arrays
    node_nst = np.array([x_nst, y_nst]).transpose()
    cell_nst = np.array([xc_nst,yc_nst]).transpose()

    node_me  = np.array([x_me, y_me]).transpose()
    cell_me  = np.array([xc_me,yc_me]).transpose()

    print('  -> node')
    tree              = KDTree(node_me)
    p,inds            = tree.query(node_nst)
    nearest_mesh_node = inds.astype(int)

    print('  -> cell')
    tree              = KDTree(cell_me)
    p,inds            = tree.query(cell_nst)
    nearest_mesh_cell = inds.astype(int)

    return nearest_mesh_node, nearest_mesh_cell

def adjust_sides(M, circular):
    '''
    Cut of the sides of the nestingzone
    '''
    nstrs = len(M.nodestrings)

    print('  -> Cropping obc nodes')
    x_obc = np.empty(0); y_obc = np.empty(0)
    if circular:
        x_obc = M.x[M.nodestrings[0]]
        y_obc = M.y[M.nodestrings[0]]

    else:
        for i in range(nstrs):
            x_tmp = M.x[M.nodestrings[i]]
            y_tmp = M.y[M.nodestrings[i]]
        
            # Look at the sides one-by-one
            x_tmp, y_tmp = crop_obc(M, x_tmp, y_tmp, x_tmp[0], y_tmp[0])
            x_tmp, y_tmp = crop_obc(M, x_tmp, y_tmp, x_tmp[-1], y_tmp[-1])

            x_obc = np.append(x_obc, x_tmp)
            y_obc = np.append(y_obc, y_tmp)

    # Find cells within R from x_obc and y_obc
    NEST = crop_mesh(M, x_obc, y_obc)

    return NEST

def crop_obc(M, x_obc, y_obc, xcoast, ycoast):
    '''
    Make cropped nestingzone
    '''
    dist  = np.sqrt((x_obc-xcoast)**2+(y_obc-ycoast)**2)
    inds  = np.where(dist >= 0.895*M.R)[0]

    return x_obc[inds], y_obc[inds]

def crop_mesh(M, x_obc, y_obc):
    '''
    Get the triangles within R from the cropped obc
    '''
    print('  -> Find the necessary cells')
    NEST_cells = []
    for i,p in enumerate(zip(M.xc, M.yc)):
        dst = np.sqrt((x_obc-p[0])**2+(y_obc-p[1])**2)
        ltR = np.where(dst<=M.R)[0]
        if len(ltR) > 0:
            NEST_cells.append(i)

    # Cells within search range
    cells = np.unique(NEST_cells).astype(int)
    
    # Store the new nodes
    inds  = np.unique(M.tri[cells,:].ravel())
    x_new = M.x[inds]
    y_new = M.y[inds]

    # Create new nv structure
    print('  -> Create nest triangulation')
    new_nv = np.nan*np.ones((len(cells), 3))
    for i,c in enumerate(cells):
        # Find corresponding x,y in the cell
        cx = M.x[M.tri[c,:]]; cy = M.y[M.tri[c,:]]
        for j,p in enumerate(zip(cx,cy)):
            dst = np.sqrt((x_new-p[0])**2+(y_new-p[1])**2)
            new_nv[i,j] = np.where(dst==dst.min())[0]

    # Overwrite return nest-dict
    dNEST        = {}
    dNEST['xn']  = x_new
    dNEST['yn']  = y_new
    dNEST['nv']  = new_nv.astype(int)
    dNEST['xc']  = triangulate(dNEST,'xn')
    dNEST['yc']  = triangulate(dNEST,'yn')
    return dNEST

def triangulate(NEST,var):
    c = (NEST[var][NEST['nv'][:,0]] + NEST[var][NEST['nv'][:,1]] + NEST[var][NEST['nv'][:,2]])/3.0
    return c
