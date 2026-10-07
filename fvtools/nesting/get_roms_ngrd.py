# ---------------------------------------------------------------------
#       Create a file containing information about the nestgrid
# ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from fvtools.grid.fvcom_grd import FVCOM_grid
from pykdtree.kdtree import KDTree

def main(mesh, nrows = 4, remove_land_squares = True):
    '''
    Create a "ngrd.npy" file to be read by the routines creating nesting files

    Parameters:
    mesh:   
        a FVCOM grid with nodestrings, either stored as .npy or .2dm
    nrows:  
        the number of rows near the obc that make up the nestingzone
    remove_land_squares: 
        remove squares in the nest zone that connect to land (True by default)
        we do this to avoid mass conservation issues near the boundary due to the
        boundary conditions, but note that it is not clear that this is necessary.
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
            remove_land_squares = False
        
    print('- Cut nestzone out of mesh')
    cells = add_rows(M, nrows = nrows, remove_land_squares = remove_land_squares)

    NEST = store_nest(M, cells)
    
    # Convert to get latlon
    print('- Projecting latlon')
    NEST['lonn'], NEST['latn'] = M.Proj(NEST['xn'], NEST['yn'], inverse = True)

    # Get cell values
    NEST['lonc'], NEST['latc'] = M.Proj(NEST['xc'], NEST['yc'], inverse = True)

    # Find corresponding indices (necessary for fvcom2fvcom)
    print('- Find nearest mesh points in the FVCOM model:')
    NEST['nid'] = M.find_nearest(NEST['xn'], NEST['yn'], grid = 'node')
    NEST['cid'] = M.find_nearest(NEST['xc'], NEST['yc'], grid = 'cell')

    # Save. (Creates a structure readable by roms_nesting and fvcom2fvcom nesting)
    NEST['oend1'] = 1; NEST['oend2'] = 1
    NEST['R'] = M.grid_res[cells].mean() # since this number is used later on in roms_nesting_fg
    NEST['info'] = {}
    NEST['info']['reference'] = M.info['reference']
    np.save('ngrd.npy', NEST)

    plt.figure()
    M.plot_grid()
    plt.triplot(NEST['xn'], NEST['yn'], NEST['nv'], c = 'r')
    plt.axis('equal')

# ------------------------------------------------------------------------------------------------------------
#                                       Subroutines
# ------------------------------------------------------------------------------------------------------------
# Add new rows to the nesting grid
def new_row(ids, triangles, open_boundary, nodes = None):
    boundary_triangles = []
    # note which nodes we already know connect to the OBC grid
    if nodes is None:
        nodes = np.unique(triangles)

    for (n, nv) in zip(ids, triangles):
        # Do not add existing cells to the list
        if nodes is None:
            if open_boundary[n] == 2:
                continue
            
        if any([True for node in nv if node in nodes]):
            # Add triangles that connect to existing open boundary nodes
            boundary_triangles.append(n)
    
            # Mark open boundary triangles so we won't have to find them again
            open_boundary[n] = 1
            
    return boundary_triangles, open_boundary

def add_rows(M, nrows = 5, remove_land_squares = True):
    '''
    Loop over all open boundaries and add new rows
    - nrows               = number of rows from the OBC to cut out of the mesh
    - remove_land_squares = if True (defeault), we remove squares that connect to FVCOM land, other than the first row
    '''
    all_obcs = []
    for obc_nodes in M.nodestrings:
        # Dummy holder for boundary triangles
        boundary_triangles = []
    
        # Copy of the open boundary identifiers
        open_boundary = np.copy(M.ISBCE)
    
        # All nearby elements to the open boundary elements
        nbse = M.nbse[open_boundary == 2]
        elements = np.unique(nbse[nbse > -1])
    
        # Build the first row
        first_row, open_boundary = new_row(elements, M.tri[elements], open_boundary, nodes = obc_nodes)

        all_obcs.extend(first_row)
        
        for i in range(nrows-1):
            if i == 0:
                nbse = M.nbse[first_row]
                elements = np.unique(nbse[nbse > -1])
                boundary_triangles, open_boundary = new_row(elements, M.tri[elements], open_boundary)
                
            else:
                nbse = M.nbse[boundary_triangles]
                elements = np.unique(nbse[nbse > -1])
                boundary_triangles, open_boundary = new_row(elements, M.tri[elements], open_boundary)

            # remove row squares that have a boundary towards land
            if remove_land_squares:
                on_land = (M.ISONB[M.tri[boundary_triangles]] == 1).any(axis=1)
                boundary_triangles = np.array(boundary_triangles)[on_land == False].tolist()
                
            # Add this open boundary
            all_obcs.extend(boundary_triangles)
    return np.unique(all_obcs)

def store_nest(M, cells):
    '''
    Store the nestingzone mesh
    '''
    print('  -> Find the necessary cells')
    # Store the new nodes
    inds  = np.unique(M.tri[cells, :].ravel())
    x_new = M.x[inds]
    y_new = M.y[inds]

    # Create new nv structure
    print('  -> Create nest triangulation')
    new_nv = np.nan*np.ones((len(cells), 3), dtype = int)
    new_tree = KDTree(np.array([x_new, y_new]).T)

    # Find corresponding x,y in the cells corners
    triangle_corner_x = M.x[M.tri[cells,:]]
    triangle_corner_y = M.y[M.tri[cells,:]]

    for j in range(3):
        _, index = new_tree.query(np.array([triangle_corner_x[:,j], triangle_corner_y[:,j]]).T)
        new_nv[:,j] = index.astype(int)
    new_nv = new_nv.astype(int)

    # Overwrite return nest-dict
    dNEST = {}
    dNEST['xn'] = x_new
    dNEST['yn'] = y_new
    dNEST['nv'] = new_nv.astype(int)
    dNEST['xc'] = M.xc[cells]
    dNEST['yc'] = M.yc[cells]
    return dNEST
