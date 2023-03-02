# ---------------------------------------------------------------------------
#               Build a obc-grid for roms2fvcom nesting
# ---------------------------------------------------------------------------
# --> Based on the obcgrid procedure in matlab
#
# Future:
# - Store nodestring
# - fix the last node so that it does not have a triangle with only one neighbour
#
# hes@akvaplan.niva.no
import sys
import numpy as np
import fvtools.grid.fvgrid as fvg
from fvtools.grid.fvcom_grd import FVCOM_grid
from scipy.spatial import cKDTree as KDTree
from operator import add

def main(meshname, res, rows):
    '''
    Build a nestzone-grid suitable for roms-fvcom nesting
    '''
    print(f'Read: {meshname}')
    t, _, x, y, _, _, nstr = fvg.read_sms_mesh(meshname, nodestrings = True)
    M = FVCOM_grid(meshname)
    
    print('\nMake the OBC mesh')
    p,tri = obcgridding(x, y, t, nstr, res, rows)
    new_name = meshname.split('.')[0]+'_with_nest'

    print(f'\nThe new mesh file is called: {new_name}')
    fvg.write_2dm(p[:,0], p[:,1], tri, name = new_name, casename = 'obcgridding')

    print('Fin.')
    
    
def obcgridding(x, y, t, nstr, res, rows):
    '''
    Builds a mesh outside of the obc
    - Assumes that SMS counts anti-clockwise (as it tends to do)

    Parameters:
    x,y :  ndarray 
        nodal coordinates
    t :    ndarray
        triangulation matrix
    nstr : list
        list containing the nodestrings from the sms grid
    '''
    
    # Supports multiple obcs.
    for i in range(len(nstr)):
        print(f'- Nodestring: {i+1} of {len(nstr)}')
        if nstr[i][0] == nstr[i][-1]:
            print('-- identified circular OBC')
            circular = True
        else:
            circular = False

        # Boundary nodes
        xb = x[nstr[i]]; yb = y[nstr[i]]
        
        # Boundary node-vector
        dx = np.array([xb[i+1]-xb[i] for i in range(len(xb)-1)])
        dy = np.array([yb[i+1]-yb[i] for i in range(len(yb)-1)])

        # The normal-vector equation
        # (a,b) * (dx,dy) = 0 if (a,b) = (-dy, dx)
        normvec = [dy, -dx]
        normvec = list(map(list, zip(*normvec)))

        # Get a normal vector for each boundary node (there's no fun working with lists...!)
        if not circular:
            normvec_node = [normvec[0]]
        else:
            normvec_node = [[0.5*normvec[0][0]+ 0.5*normvec[-1][0], 0.5*normvec[-1][1]+0.5*normvec[0][1]]]
        normvec_node_rest = [list(map(add,a,b)) for a,b in zip(normvec[1:],normvec[:-1])]
        normvec_node_half = [[a/2, b/2] for a,b in normvec_node_rest]
        normvec_node.extend(normvec_node_half)
        
        if not circular:
            normvec_node.extend([normvec[-1]])

        # Normalize the normal vector (in the future we might want variable size, then this will be obsolete.)
        dst       = [np.sqrt(a**2+b**2) for a,b in normvec_node] 
        unit_norm = np.array([[v[0]/l, v[1]/l] for v,l in zip(normvec_node, dst)])
        
        # Build the structured grid
        # - Loop over each node, add points away from the obc. create new triangles.
        if not circular:
            xr = xb
            yr = yb
        else:
            # This row
            xr = xb[:-1]
            yr = yb[:-1]

            # Update row positions
            xb = xb[:-1]
            yb = yb[:-1]

        xnew   = np.empty(0)
        ynew   = np.empty(0)
        new_nv = []

        print('- Loop to add rows and build new triangles')
        for row in range(rows):
            if row == 0:
                # Update the nestgrid-nodes
                xr     = np.append(xr,xb+res*unit_norm[:,0])
                yr     = np.append(yr,yb+res*unit_norm[:,1])

                # Update those we paste to the full mesh
                xnew   = np.append(xnew, xb+res*unit_norm[:,0])
                ynew   = np.append(ynew, yb+res*unit_norm[:,1])
                old_xr = xb+res*unit_norm[:,0]
                old_yr = yb+res*unit_norm[:,1]
                    
                # Store the triangles
                # Better to think of them one square at the time
                x_square = [xb[:-1], xb[1:], old_xr[:-1], old_xr[1:]]
                y_square = [yb[:-1], yb[1:], old_yr[:-1], old_yr[1:]]

                if circular:
                    x_square = circular_adjustment(x_square, xb, old_xr)
                    y_square = circular_adjustment(y_square, yb, old_yr)

                # Build the triangle for each square
                for i in range(len(x_square[0][:])):
                    ind = []
                    for j in range(len(x_square)):
                        dst = np.sqrt((xr-x_square[j][i])**2+(yr-y_square[j][i])**2)
                        ind.append(np.where(dst==dst.min())[0][0])

                    new_nv.append([ind[1],ind[3],ind[2]])
                    new_nv.append([ind[1],ind[0],ind[2]])
                    

            else:
                # Update the nestgrid-nodes
                xr = np.append(xr,old_xr+res*unit_norm[:,0])
                yr = np.append(yr,old_yr+res*unit_norm[:,1])

                # Update those we paste to the full mesh
                xnew   = np.append(xnew, old_xr+res*unit_norm[:,0])
                ynew   = np.append(ynew, old_yr+res*unit_norm[:,1])

                # Prepare to look for triangles
                this_xr = np.copy(old_xr)
                this_yr = np.copy(old_yr)
                old_xr += res*unit_norm[:,0]
                old_yr += res*unit_norm[:,1]

                # Cluster to squares
                x_square = [this_xr[:-1], this_xr[1:], old_xr[:-1], old_xr[1:]]
                y_square = [this_yr[:-1], this_yr[1:], old_yr[:-1], old_yr[1:]]
                if circular:
                    x_square = circular_adjustment(x_square, this_xr, old_xr)
                    y_square = circular_adjustment(y_square, this_yr, old_yr)

                # Build the triangle for each square
                for i in range(len(x_square[0][:])):
                    ind = []
                    for j in range(len(x_square)):
                        dst = np.sqrt((xr-x_square[j][i])**2+(yr-y_square[j][i])**2)
                        ind.append(np.where(dst==dst.min())[0][0])

                    new_nv.append([ind[1],ind[3],ind[2]])
                    new_nv.append([ind[1],ind[0],ind[2]])

        # Merge the nestzone with the computation mesh
        # - node location
        print('- Add the new nodes')
        x_full  = np.append(x, xnew)
        y_full  = np.append(y, ynew)
        nv_full = np.copy(t)

        # - triangulation
        updated_nv = []
        print('- Fit the nestzone triangulation to the full mesh')
        tree = KDTree(np.array([x_full, y_full]).T)
        p, ind = tree.query(np.array([xr, yr]).T)
        for nv in new_nv:
            tri = [ind[nv[0]], ind[nv[1]], ind[nv[2]]]
            updated_nv.append(tri)

        updated_nv = np.array(updated_nv).astype(int)
        nv_full = np.append(nv_full, updated_nv, 0)

        # Update the mesh metrics
        x = np.copy(x_full)
        y = np.copy(y_full)
        t = np.copy(nv_full)

        # Haven't tested it, but it should work for multiple OBCs

    p = np.array((x,y)).T
    return p,t

def circular_adjustment(square_in, pos_in, old_pos):
    """
    Connect first and last xb/yb
    """
    square_in[0] = np.append(square_in[0], pos_in[-1])
    square_in[1] = np.append(square_in[1], pos_in[0])
    square_in[2] = np.append(square_in[2], old_pos[-1])
    square_in[3] = np.append(square_in[3], old_pos[0])
    return square_in
