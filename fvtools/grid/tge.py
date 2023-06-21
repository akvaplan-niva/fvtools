'''
TGE is a function that computes grid metrics for your FVCOM grid

version 0.2:
- routines now run without failing. Some uncertainty with respect to how some
  functions work in parallell, and if the CVs are calculated correctly
  (will be tested by computing art1)

version 0.3:
- the TGE class is adjusted to work more seamlessly with fvgrad

version 0.4:
- fix a bug in handling of CV walls connecting to land

version 0.5:
- Add gradient estimation at nodes using a least-squares algorithm

version 0.6:
- Gradient of data on cells (FVCOM has routines for this internally, which
                             should probably be copied for consistency - but
                             we use the same mathematical method in this script)

version 0.6:
- Fix boundary issue with vorticity calculation

version 0.7:
- Calculate the streamfunction on an unstructured grid

'''
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from numba import jit, prange
from datetime import date
versionnr = 0.7

def main(M, verbose = False):
    '''
    Calculate the grid metrics that FVCOM stores internally when computing gradients etc.

    Input:
    - M (object from FVCOM_grid in fvcom_grd) containing
      - tri    (triangulation)
      - x,  y  (node positions)
      - xc, yc (centrod positions)
      - obc_nodes or nodestrings (reference to position of open boundary, optional)

    - verbose  (default: False - boolean that turns on progress reports)

    Output:
      - TGE_dict (dictionary containing the fields listed below)

    =======================================================================================
                           Permanent parameters in this routine:
    =======================================================================================

    Dimensions (index), positions and triangulation:
    ------------------
    - NT              (number of elements)
    - MT              (number of nodes)
    - VX, VY [MT]     (node locations)
    - XC, YC [NT]     (cell locations)
    - NV     [NT,0:2] (triangulation)

    Element neighbor information
    ------------------
    - NBE    [NT,0:2]       (element index of 1-3 neighbors of element N)
    - ISBCE  [NT]           (element on boundary check)
                             isbce = 0 element in the interior of the
                                       computational domain
                             isbce = 1 element on the solid boundary
                             isbce = 2 element on the open boundary
                             isbce = 3 element with 2 solid boundary edges
    - ISONB  [MT]           (node is on the boundary check
                             isonb = 1 on the solid boundary
                             isonb = 2 on the open boundary)
    - NTVE   [MT]            (number of neighboring elements of node M)
    - NBVE   [MT, :NTVE[MT]] (ntve elements containing node i)
    - NBVT   [MT, :NTVE(MT)] (the node number of node i in element nvbe[i,j])
    - NESE   [MT]            (number of elements surrounding a element)
    - NBSE   [MT, :NESE[MT]] (ids of elements surrounding a element)

    Edge information
    ------------------
    - NE                    (number of unique element edges)
    - IEC    [NE, 2]        (cell IDs for those connected to edge i)
    - ISBC   [NE]           (flag marking edge property: 0 - interior)
                                                         1 - boundary)
    - ienode [NE,2]         (node numbers at each end of element edge i)
    - XIJC   [NE]           (x-coordinate of mid-point of element edge i)
    - YIJC   [NE]           (y-coordinate of mid-point of element edge i)
    - DLTXYC [NE]           (length of element edge i)
    - DLTXC  [NE]           (delta_x - x-projection of dltxyc)
    - DLTYC  [NE]           (delta_y - y-projection of dltxyc)
    - SITAC  [NE]           (arctg(dltyc, dltxc) - angle of incliation of edge)

    Node neighbor information
    ------------------
    - NTVE   [MT]           (total number of the surrounding triangles connected to the given node)
    - NBVE   [M, NTVE+1]    (the identification number of a given node over each individual surrounding triangle (counted clockwise))
    - NBVT   [M, NTVE[1:M]] (the identification number of a given node over each individual surrounding triangle (counted clockwise))
    - NTSN   [M]            (total number of surrounding nodes)
    - NBSN   [M, NTSN]      (the identification number of surrounding nodes counted clockwise)

    Boundary information
    ------------------
    iobce             (number of open boundary cells)
    isbcn             (number of open boundary nodes)
    i_obc_e           (counter number of open boundary cells)
    i_obc_n           (counter number of open boundary nodes)
    '''
    if verbose:
        print('-------------------------------------------------------------------------------')
        print('                       Setting up TRIS/ELEMENTS/CVS')
        print('-------------------------------------------------------------------------------')

    # Initialize output:
    out = TGE()

    # ------------------------------------------------------------------------------------------------
    #                                       Input handling
    # ------------------------------------------------------------------------------------------------
    # Unpack obc nodes
    OBC_NODES  = get_obc(M)
    out.source = M.filepath

    # Unpack dimensions
    out.NT = len(M.xc); out.MT = len(M.x)

    # Unpack positions
    out.VX = M.x;  out.VY = M.y  # nodes
    out.XC = M.xc; out.YC = M.yc # elements/cells

    # Unpack triangulation (+ see re-arange if needbe so that triangulation is clockwise)
    out.NV = check_nv(M.tri, out.VX, out.VY, verbose = verbose)

    # -----------------------------------------------------------------------------------------------
    #                         Domain information that require some computing
    # -----------------------------------------------------------------------------------------------
    # Get nearby elements
    out.NBE                = get_NBE(out.NT, out.MT, out.NV)
    if verbose: print('- Found nearby elements')

    # Check if element is on boundary
    ISBCE, ISONB           = get_BOUNDARY(out.NT, out.MT, out.NBE, out.NV)
    if verbose: print('- Found and flagged boundary nodes and elements.')

    # Get max number of surrounding elements
    #MAXNBR                 = np.max(get_MAXNBR(out.MT, out.NT, out.NV))
    #if verbose: print(f'- Found max number of surrounding elements: {MAXNBR}')
    MAXNBR = 20

    # Get number of surrounding triangles to a given node
    NBVE, NBVT, out.NTVE   = get_NBVE_NBVT(out.MT, out.NT, out.NV, MAXNBR)
    if verbose: print('- Found number of elements surrounding nodes')

    # Find number of nodes surrounding nodes, and elements surrounding nodes
    out.NTSN, out.NBSN, out.NBVE, out.NBVT, invalid_nodes = get_NTSN_NBSN(NBVE, out.NTVE, NBVT, out.NBE,
                                                                          out.NV, ISONB, MAXNBR, out.MT)
    if verbose: print('- Reordered elements surrounding nodes, found NTSN and NBSN')

    # Check if any of the nodes are invalid, if so kill and indicate where we have a problem
    invalid_nodes = invalid_nodes.astype(bool)
    if any(invalid_nodes):
        M.plot_grid()
        plt.scatter(M.x[invalid_nodes], M.y[invalid_nodes], c = 'r')
        plt.draw()
        raise Exception(f'{len(np.where(invalid_nodes)[0])} of the boundary nodes are invalid, see figure.')

    # Control volume grid metrics
    out.NE, out.IEC, out.IENODE, out.ISBC  = get_TRI_EDGE_PARAM(out.NT, out.NBE, out.NV)
    if verbose: print('- Found CV connectivity')

    # Get element wall positions, lengths and angles (return as dict)
    out.DLTXC, out.DLTYC, out.XIJC, out.YIJC, out.DLTXYC, out.SITAC = get_element_edge_metrics(out.VX, out.VY, out.IENODE, out.NE)
    if verbose: print('- CV distances computed. ')

    # Set ISONB on open boundary nodes, determine if element on open boundary, on land or in the interior
    if len(OBC_NODES) < 1:
        OBC_NODES = None
    out.ISONB, out.ISBCE   = set_boundary(ISONB, ISBCE, out.NV, out.NBE, OBC_NODES, out.NT)
    if verbose: print('- Nodes and cells on the edge of the grid identified and tagged')

    # Get more control volume wall positions, lengths and angles (return as dict)
    out.XIJE, out.YIJE, out.NTRG, out.NIEC, out.DLTXE, out.DLTYE, out.DLTXYE, out.SITAE, out.NCV = get_CV(out.XIJC, out.YIJC, out.XC, out.YC, \
                                                                                                          out.ISBC, out.IEC, out.IENODE, out.NE, out.NT, out.MT)
    if verbose: print('- Control volume walls (indices, lengths, angles) identified and calculated.')

    # Get shape coefficients for gradient calculation
    out.A1U, out.A2U = shape_coefficients(out.NT, out.XC, out.YC, out.NBE, out.ISBCE)
    if verbose: print('- Found shape coefficients for gradient calculation')

    # Add reference to M to object if needed later?
    out.M = M

    # Compute CV area
    out.get_art1()
    if verbose: print('- Computed control volume area (art1)')

    if verbose: print('-------------------------------------------------------------------------------')
    if verbose: print('                                   Fin')
    if verbose: print('-------------------------------------------------------------------------------')

    return out

@jit(nopython = True, parallel = True)
def get_NBE(NT, MT, NV):
    '''
    Determine NBE
    '''
    NBE     = -1*np.ones((NT,3),  dtype = np.int64)
    CELLS   = -1*np.ones((NT,50), dtype = np.int64)
    CELLCNT = np.zeros((MT), dtype = np.int64)

    # Store a structure where node N is associated with the cells that connect to it
    for I in range(NT):
        for J in range(3):
            N = NV[I,J]
            CELLS[N, CELLCNT[N]] = I # Index of triangles connected to this node
            CELLCNT[N] += 1          # Number of triangles connected to this node

    # CELLS contain the CELLS associated with any NODE.
    # In this loop we loop over cells associated with node N for all nodes in a triangle.
    # If the cells in two neighboring nodes are the same, they must be at each others boundary.
    for I in range(NT):
        N1 = NV[I,0]
        N2 = NV[I,1]
        N3 = NV[I,2]

        # If two cells connected to these nodes are the same, and
        # not the cell we're considering - then these share a side with cell I

        # First nodes connected to the line segment N1, N2
        for J1 in range(CELLCNT[N1]):         # Loop over number of cells connected to node 1
            for J2 in range(CELLCNT[N2]):     # Loop over number of cells connected to node 2
                if (CELLS[N1, J1] == CELLS[N2,J2]) and (CELLS[N1, J1] != I):
                    NBE[I,2] = CELLS[N1, J1]

        for J2 in range(CELLCNT[N2]):
            for J3 in range(CELLCNT[N3]):
                if (CELLS[N2, J2] == CELLS[N3,J3]) and (CELLS[N2, J2] != I):
                    NBE[I,0] = CELLS[N2, J2]

        for J1 in range(CELLCNT[N1]):
            for J3 in range(CELLCNT[N3]):
                if (CELLS[N3, J3] == CELLS[N1,J1]) and (CELLS[N3, J3] != I):
                    NBE[I,1] = CELLS[N3, J3]

    return NBE

@jit(nopython = True, parallel = True)
def get_BOUNDARY(NT, MT, NBE, NV):
    '''
    Flag boundary nodes and cells
        - set boundary elements and nodes to 1, 0 otherwise
    '''
    ISBCE = np.zeros((NT), np.int32)
    ISONB = np.zeros((MT), np.int32)

    # Determine the index of the three surrounding elementsa
    # Note that it is important that the nodes are arranged CLOCKWISE in this routine
    # ----
    for I in prange(NT):
        JJB = 0
        if np.min(NBE[I,:]) == -1:
            ISBCE[I] = 1
            if NBE[I,0] == -1:
                ISONB[NV[I,1]] = 1
                ISONB[NV[I,2]] = 1

            if NBE[I,1] == -1:
                ISONB[NV[I,0]] = 1
                ISONB[NV[I,2]] = 1

            if NBE[I,2] == -1:
                ISONB[NV[I,0]] = 1
                ISONB[NV[I,1]] = 1

    return ISBCE, ISONB

@jit(nopython = True, parallel = True)
def get_MAXNBR(MT, NT, NV):
    '''
    Determines the maximum number of surrounding elements
    '''
    elems_with_this_node = np.zeros((MT), dtype = np.int32) # Elements around each node
    for I in prange(MT):
        NCNT = 0
        for J in range(NT):
            if float(NV[J,0] - I)*float(NV[J,1] - I)*float(NV[J,2] - I) == 0:
                NCNT += 1
        elems_with_this_node[I] = NCNT

    return elems_with_this_node

# Try to rewrite this in a more pythonic way to speed it up
@jit(nopython = True, parallel = True)
def get_NBVE_NBVT(MT, NT, NV, MXNBR_ELEMS):
    '''
    Determine number of surrounding elements for node I = NTVE(I)
    '''
    NBVE = -1*np.ones((MT, MXNBR_ELEMS+1), np.int32) # Indices of neighboring elements of node I
    NBVT = -1*np.ones((MT, MXNBR_ELEMS+2), np.int32) # index of node I in neighboring element
    NTVE = np.zeros((MT), np.int32)                  # Counting array, indexes indicate number of elements around each node

    for I in prange(MT):
        NCNT = -1      # nodecount, negative by default
        for J in range(NT):
            if float(NV[J,0] - I)*float(NV[J,1] - I)*float(NV[J,2] - I) == 0:
                NCNT         += 1
                NBVE[I,NCNT]  = J
                for K in range(3):
                    if NV[J,K]-I == 0:
                        NBVT[I,NCNT] = K

        NTVE[I]=NCNT+1 # +1 to adjust for the negative initial nodecount

    return NBVE, NBVT, NTVE

@jit(nopython = True)
def get_NTSN_NBSN(NBVE, NTVE, NBVT, NBE, NV, ISONB, MXNBR, MT):
    '''
    Returns NTSN and NBSN, and reorders elements surrounding a node to go in a cyclical procession
    '''
    # Temporary storage for use inside the loop
    NBSN   = -1*np.ones((MT, MXNBR+3), dtype = np.int64)      # Number of nodes surrounding a node (+1)
    NTSN   = -1*np.ones((MT), dtype = np.int64)               # Node indicies of nodes surrounding a node
    nearby_elements = np.zeros((MXNBR+1,2), dtype = np.int64) # Temporary storage of the two above (since we are rearranging)
    invalid_nodes = np.zeros((MT,), dtype = np.int64)
    for I in range(MT):                                # Loop over all nodes (as far as I can see, this is safe to paralellize)
        if ISONB[I] == 0:                              # If we are in the interior
            nearby_elements[0,0] = NBVE[I,0]           # Indicies of neighboring elements of node I
            nearby_elements[0,1] = NBVT[I,0]           # Index (0,1,2) of node I in the neighboring element

            for J in range(1, NTVE[I]+1):              # Loop over triangles connected to node
                element    = nearby_elements[J-1, 0]   # II are thus indicies of neighboring elements to node I)
                tri_corner = nearby_elements[J-1, 1]   # JJ is thus the index of nodes in the neighboring element)
                nearby_elements[J,0] = NBE[element, int(tri_corner+1-np.floor((tri_corner+2)/4)*3)] # first element clockwise of start

                # This was even sloppier coding for a while...
                element = nearby_elements[J,0]

                # Store the corner of triangle JJ connecting to the node we are considering
                for K in range(3):
                    if NV[element,K] == I:
                        nearby_elements[J,1] = K
                        break

            for J in range(1,NTVE[I]+1):
                NBVE[I,J] = nearby_elements[J,0]
                NBVT[I,J] = nearby_elements[J,1]

            # Update number of triangles surrounding node
            NTSN[I] = NTVE[I]

            for J in range(NTSN[I]):
                element    = NBVE[I,J]
                tri_corner = NBVT[I,J]
                NBSN[I,J]  = NV[element, int(tri_corner+1-np.floor((tri_corner+2)/4)*3)]

            NTSN[I]          += 1
            NBSN[I,NTSN[I]-1] = NBSN[I,0]

        else:
            JJB = 0

            # We first identify the triangle facing land
            for J in range(NTVE[I]):
                tri_corner = NBVT[I,J]

                # check to find the boundary side of triangle
                if NBE[NBVE[I,J], int(tri_corner+2-np.floor((tri_corner+3)/4)*3)] == -1: # if cell in counterclockwise direction is boundary cell
                    JJB +=1
                    nearby_elements[0, 0] = NBVE[I,J] # Store the triangle next to the boundary
                    nearby_elements[0, 1] = NBVT[I,J] # And the corner counter-clockwise to land

            if JJB != 1:
                print('--> Invalid boundary at node '+str(I))
                invalid_nodes[I] = 1

            # And loop over the other triangles around the node
            for J in range(1, NTVE[I]):
                element    = nearby_elements[J-1,0]
                tri_corner = nearby_elements[J-1,1]
                nearby_elements[J,0] = NBE[element, int(tri_corner+1-np.floor((tri_corner+2)/4)*3)] # next node in clockwise direction
                element    = nearby_elements[J,0]

                for K in range(3):
                    if NV[element, K] == I:
                        nearby_elements[J,1] = K
                        break

            # Update nearest element information with what we have just learned
            for J in range(NTVE[I]):
                NBVE[I,J] = nearby_elements[J,0]
                NBVT[I,J] = nearby_elements[J,1]

            NBVE[I,NTVE[I]] = -1
            NTSN[I]   = NTVE[I]+1
            NBSN[I,0] = I

            for J in range(NTSN[I]-1):
                element = NBVE[I,J]
                tri_corner = NBVT[I,J]
                NBSN[I,J+1] = NV[element, int(tri_corner+1-np.floor((tri_corner+2)/4)*3)]

            J  = NTSN[I]
            element    = NBVE[I,J-2]
            tri_corner = NBVT[I,J-2]
            shift_counter_clockwise = int(tri_corner+2-np.floor((tri_corner+3)/4)*3)
            NBSN[I,J]  = NV[element,shift_counter_clockwise]
            NTSN[I]   += 2
            NBSN[I,NTSN[I]-1] = I

    return NTSN, NBSN, NBVE, NBVT, invalid_nodes

@jit(nopython = True, parallel = True)
def get_TRI_EDGE_PARAM(NT, NBE, NV):
    '''
    Get number of unique element edges (ie. no duplicates), and the cell ids that connect to given edge
    '''
    # Allocate
    NE     = 0                                       # number of unique edges
    ISET   = np.zeros((NT, 3),  dtype = np.int32)    # temporary storage to check wether this edge has been considered
    TEMP   = -1*np.zeros((3*NT,2), dtype = np.int32) # temporary storage for identification number for to connected cells
    TEMP2  = -1*np.zeros((3*NT,2), dtype = np.int32) # temporary storage for identification number of nodes on
                                                     # edge between connected cells (left and right node corner)

    for I in range(NT):
        for J in range(3):
            if ISET[I,J] == 0:
                element_other_side = NBE[I,J]  # Elements that connect to this element on this wall
                ISET[I,J] = 1                  # Mark to show that this edge is no longer unique

                if element_other_side != -1:
                    for JN in range(3):
                        if I == NBE[element_other_side, JN]: # If this edge does not connect to land
                            ISET[element_other_side, JN] = 1

                # Update nearest element on either side of the interface
                TEMP[NE,0]  = I
                TEMP[NE,1]  = element_other_side

                # Update the nodes spanning the wall
                TEMP2[NE,0] = NV[I, int(J+1-np.floor((J+2)/4)*3)]
                TEMP2[NE,1] = NV[I, int(J+2-np.floor((J+3)/4)*3)]

                NE         += 1                # Count number of unique edges

    # Allocate array for these
    IEC         = np.zeros((NE,2), dtype = np.int32)
    IENODE      = np.zeros((NE,2), dtype = np.int32)

    # Dump temporary data to permanent storage
    IEC[:,0]    = TEMP[:NE,0]
    IEC[:,1]    = TEMP[:NE,1]
    IENODE[:,0] = TEMP2[:NE,0]
    IENODE[:,1] = TEMP2[:NE,1]

    # Mark element edges that are on the boundary
    ISBC   = np.zeros((NE), dtype = np.int32)
    for I in range(NE):
        if (IEC[I,0] == -1) or (IEC[I,1] == -1):
            ISBC[I] = 1

    return NE, IEC, IENODE, ISBC

def get_element_edge_metrics(VX, VY, IENODE, NE):
    '''
    For lengths of edges and angles relative to west-east axis
    '''
    DLTXC  = VX[IENODE[:,1]]  - VX[IENODE[:,0]]
    DLTYC  = VY[IENODE[:,1]]  - VY[IENODE[:,0]]
    XIJC   = (VX[IENODE[:,1]] + VX[IENODE[:,0]])/2.0
    YIJC   = (VY[IENODE[:,1]] + VY[IENODE[:,0]])/2.0
    DLTXYC = np.sqrt(DLTXC*DLTXC + DLTYC*DLTYC)
    SITAC  = np.arctan2(DLTYC, DLTXC)

    return DLTXC, DLTYC, XIJC, YIJC, DLTXYC, SITAC

@jit(nopython = True, parallel = True)
def set_boundary(ISONB, ISBCE, NV, NBE, OBC_NODES, NT):
    '''
    Marks open boundary nodes and open boundary elements
    '''
    if OBC_NODES is not None:
        for I in prange(len(OBC_NODES)):
            ISONB[OBC_NODES[I]] = 2

    for I in prange(NT):
        ITMP1 = ISONB[NV[I,0]]
        ITMP2 = ISONB[NV[I,1]]
        ITMP3 = ISONB[NV[I,2]]

        if np.sum(ISONB[NV[I,:]]) == 4:
            ISBCE[I] = 2

    for I in prange(NT):
        if NBE[I,0]+NBE[I,1]+NBE[I,2] == 0 and ISBCE[I] != 2:
            ISBCE[I] = 3

        if NBE[I,0]+NBE[I,1] == 0 and ISBCE[I] != 2:
            ISBCE[I] = 3

        if NBE[I,1]+NBE[I,2] == 0 and ISBCE[I] != 2:
            ISBCE[I] = 3

        if NBE[I,0]+NBE[I,2] == 0 and ISBCE[I] != 2:
            ISBCE[I] = 3

    return ISONB, ISBCE

@jit(nopython = True)
def get_CV(XIJC, YIJC, XC, YC, ISBC, IEC, IENODE, NE, NT, MT):
    '''
    Input:
    - XIJC, YIJC - Grid coordinate on mid triangle walls
    - XC, YC     - Element/cell centre position
    - ISBC       - Test for element on/off boundary
    - NE         - Number of unique element edges
    - IENODE     - Index of nodes on each element edge
    - NT         - Number of elements

    Computes
    - x, y-coordinates of start/end position of Control Volume (CV) edges
    - niec
        --> counting number of left and right nodes connected to this edge from start to end.
    - dltxe, dltye, dltxye
        --> distance of each subsegment the CVs form
    - ntrg
        --> element associated with htis CV edge

    --> This routine builds a set of CV polygons within the grid
    '''
    # Initialize data structures
    NCTMP  = -1 # For a seemless translation to python indexing
    NCETMP = -1

    # Floats (since their units are grid coordinates)
    XIJE   = np.zeros((NT*3, 2), np.float64)
    YIJE   = np.zeros((NT*3, 2), np.float64)
    DLTXE  = np.zeros((NT*3), np.float64)
    DLTYE  = np.zeros((NT*3), np.float64)
    DLTXYE = np.zeros((NT*3), np.float64)
    SITAE  = np.zeros((NT*3), np.float64)

    # Integers, since they are indices
    NIEC   = np.zeros((NT*3, 2), np.int64)
    NTRG   = np.zeros((NT*3), np.int64)

    for I in range(NE):
        if ISBC[I] == 0: # Loop over the interior of the domain first
            if IEC[I,0] <= (NT-1):
                NCTMP   += 1
                NPT      = NCTMP # python always copies scalars (and thus numba should too)
            else:
                NCETMP  += 1
                NPT      = NCETMP+3*NT

            # Corners of each CV
            XIJE[NPT,0]  = XC[IEC[I,0]]
            YIJE[NPT,0]  = YC[IEC[I,0]]
            XIJE[NPT,1]  = XIJC[I]
            YIJE[NPT,1]  = YIJC[I]

            # Indicies connecting the nodes from left to right
            NIEC[NPT, 0] = IENODE[I,0]
            NIEC[NPT, 1] = IENODE[I,1]

            # Triangle these two CV edges are within
            NTRG[NPT]    = IEC[I,0]

            # Length of control volume wall
            DLTXE[NPT]   = XIJE[NPT,1]-XIJE[NPT,0]
            DLTYE[NPT]   = YIJE[NPT,1]-YIJE[NPT,0]
            DLTXYE[NPT]  = np.sqrt(DLTXE[NPT]*DLTXE[NPT] + DLTYE[NPT]*DLTYE[NPT])

            # Angle of wall
            SITAE[NPT]   = np.arctan2(DLTYE[NPT], DLTXE[NPT])

            if IEC[I,1] <= (NT-1):
                NCTMP   += 1
                NPT      = NCTMP

            else:
                NCETMP  += 1
                NPT      = NCETMP+3*NT

            XIJE[NPT,0]  = XC[IEC[I,1]]
            YIJE[NPT,0]  = YC[IEC[I,1]]
            XIJE[NPT,1]  = XIJC[I]
            YIJE[NPT,1]  = YIJC[I]
            NIEC[NPT,0]  = IENODE[I,1]
            NIEC[NPT,1]  = IENODE[I,0]
            NTRG[NPT]    = IEC[I,1]
            DLTXE[NPT]   = XIJE[NPT,1] - XIJE[NPT,0]
            DLTYE[NPT]   = YIJE[NPT,1] - YIJE[NPT,0]
            DLTXYE[NPT]  = np.sqrt(DLTXE[NPT]*DLTXE[NPT]+DLTYE[NPT]*DLTYE[NPT])
            SITAE[NPT]   = np.arctan2(DLTYE[NPT], DLTXE[NPT])

        elif ISBC[I] == 1:
            if IEC[I,0] <= (NT-1):
                NCTMP   += 1
                NPT      = NCTMP
            else:
                NCETMP  += 1
                NPT      = NCETMP+3*NT

            # Corners of each CV
            XIJE[NPT,0]  = XC[IEC[I,0]]
            YIJE[NPT,0]  = YC[IEC[I,0]]
            XIJE[NPT,1]  = XIJC[I]
            YIJE[NPT,1]  = YIJC[I]

            # Indicies connecting the nodes from left to right
            NIEC[NPT, 0] = IENODE[I,0]
            NIEC[NPT, 1] = IENODE[I,1]

            # Triangle these two CV edges are within
            NTRG[NPT]    = IEC[I,0]

            # Length of control volume wall
            DLTXE[NPT]   = XIJE[NPT,1]-XIJE[NPT,0]
            DLTYE[NPT]   = YIJE[NPT,1]-YIJE[NPT,0]
            DLTXYE[NPT]  = np.sqrt(DLTXE[NPT]*DLTXE[NPT] + DLTYE[NPT]*DLTYE[NPT])

            # Angle of wall
            SITAE[NPT]   = np.arctan2(DLTYE[NPT], DLTXE[NPT])

    # Now then
    NCV_I = NCTMP+1
    NCV   = NCETMP + NCTMP + 2

    return XIJE, YIJE, NTRG, NIEC, DLTXE, DLTYE, DLTXYE, SITAE, NCV

# To find elements surrounding elements
# ----
@jit(nopython = True, parallel = True)
def get_NBSE_NESE(NBVE, NTVE, NV, NT):
    '''
    Returns NESE and NBSE, and reorders elements surrounding a cell to go in a cyclical procession
    '''
    # Temporary storage for use inside the loop
    NBSE   = -1*np.ones((NT, len(NBVE[0,:])+10), dtype = np.int64) # Number of cells surrounding a cell (+10) (neighboring surrounding)
    NESE   = -1*np.ones((NT), dtype = np.int64)                   # Node indicies of nodes surrounding a node

    for I in range(NT):   # Loop over all cells
        count  = -1
        for J in NV[I,:]: # Check each node in each cell
            for K in NBVE[J,:NTVE[J]]:
                if K in NBSE[I,:count+1]:
                    continue
                elif K == I:
                    continue
                count += 1
                NBSE[I,count] = K
                NESE[I]       = count

    return NBSE, NESE

# =======================================================================================================================
#                                           Cell-based gradients
# =======================================================================================================================
@jit(nopython = True)
def shape_coefficients(NT, XC, YC, NBE, ISBCE):
    '''
    Calculates the coefficiet for a linear function on the x-y plane, ie:
    r(x,y;phai)=phai_c+cofa1*x+cofa2*y
      innc(i)=0    cells on the boundary
      innc(i)=1    cells in the interior

    In this routine, we only care about the gradients we need to make an estimate of vorticity, hence some variables
    in the fortran files are left out (namely aw0, awx, awy)

    This routine will just be executed once
    '''
    a1u = np.zeros((NT, 4), dtype = np.float64)
    a2u = np.zeros((NT, 4), dtype = np.float64)
    for I in range(NT):
        # Interior cells
        # -------------------------------------------------------------------------------------
        if ISBCE[I] == 0:
            # Distance from surrounding elements to center element
            # ----
            # dy
            Y1 = YC[NBE[I,0]]-YC[I]
            Y2 = YC[NBE[I,1]]-YC[I]
            Y3 = YC[NBE[I,2]]-YC[I]

            # dx
            X1 = XC[NBE[I,0]]-XC[I]
            X2 = XC[NBE[I,1]]-XC[I]
            X3 = XC[NBE[I,2]]-XC[I]

            # Convert meters to kilometers
            # ----
            X1 = X1/1000.0
            X2 = X2/1000.0
            X3 = X3/1000.0

            Y1 = Y1/1000.0
            Y2 = Y2/1000.0
            Y3 = Y3/1000.0

            delt = (X1*Y2-X2*Y1)**2+(X1*Y3-X3*Y1)**2+(X2*Y3-X3*Y2)**2
            delt = delt*1000.0

            # This seems to be equivalent to the inverse matrix in the cell_gradient routine
            #
            # A*grad_f = df
            #
            # A      = [(x_1-x_0), (y_1-y_0)]
            #          [(x_2-x_0), (y_2-y_0)]
            #          [(x_3-x_0), (y_3-y_0)]
            # grad_f = [df/dx, df/dy]
            # df     = [(f_1-f_0)]
            #          [(f_2-f_0)]
            #          [(f_3-f_0)]
            #
            # This is solved for grad_f:
            # grad_f = (A^T*A)^(-1) * df
            #
            # a1u and a2u represents the (A^T*A)^(-1) matrix.
            # ----
            # - x components
            a1u[I,0] = (Y1+Y2+Y3)*(X1*Y1+X2*Y2+X3*Y3) - (X1+X2+X3)*(Y1**2+Y2**2+Y3**2)
            a1u[I,1] = (Y1**2+Y2**2+Y3**2)*X1-(X1*Y1+X2*Y2+X3*Y3)*Y1
            a1u[I,2] = (Y1**2+Y2**2+Y3**2)*X2-(X1*Y1+X2*Y2+X3*Y3)*Y2
            a1u[I,3] = (Y1**2+Y2**2+Y3**2)*X3-(X1*Y1+X2*Y2+X3*Y3)*Y3

            # - y components
            a2u[I,0] = (X1+X2+X3)*(X1*Y1+X2*Y2+X3*Y3) - (Y1+Y2+Y3)*(X1**2+X2**2+X3**2)
            a2u[I,1] = (X1**2+X2**2+X3**2)*Y1-(X1*Y1+X2*Y2+X3*Y3)*X1
            a2u[I,2] = (X1**2+X2**2+X3**2)*Y2-(X1*Y1+X2*Y2+X3*Y3)*X2
            a2u[I,3] = (X1**2+X2**2+X3**2)*Y3-(X1*Y1+X2*Y2+X3*Y3)*X3

            # Convert back to meters
            # ----
            a1u[I,0] = a1u[I,0]/delt
            a1u[I,1] = a1u[I,1]/delt
            a1u[I,2] = a1u[I,2]/delt
            a1u[I,3] = a1u[I,3]/delt

            a2u[I,0] = a2u[I,0]/delt
            a2u[I,1] = a2u[I,1]/delt
            a2u[I,2] = a2u[I,2]/delt
            a2u[I,3] = a2u[I,3]/delt

        else:
            a1u[I,:] = 0.0
            a2u[I,:] = 0.0

    return a1u, a2u

# =======================================================================================================================
#                                     Bonus scripts (not copied from TGE)
# =======================================================================================================================
# Check if this M object has OBC nodes
def get_obc(M):
    '''
    Simple check to see if obc_nodes is available
    '''
    if any(M.obc_nodes):
        obc_nodes = M.obc_nodes

    else:
        print(f'{M.filepath}\ndoes not contain OBC information.')
        obc_nodes = np.empty(0, dtype = np.int64)

    return obc_nodes

# Ensure that the triangulation we feed to TGE is oriented clockwise
def check_nv(nv, x, y, verbose = False):
    '''
    Based on code from stackoverflow:
    https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    '''
    neg = test_nv(x,y,nv)

    if neg.size == 0:
        if verbose: print('- Triangulation from source file is clockwise')

    elif neg.size>0:
        if len(neg) != len(nv[:,0]):
            raise ValueError('The triangulation direction is inconsistent, either edit tge.py to\n'+\
                             'fix this, or fix the triangulation. TGE will not fix it for now, since\n'+\
                             'I am a bit qurious to see if triangulations can have inconsistent directions.')

        nv  = np.array([nv[:,0], nv[:,2], nv[:,1]], dtype = np.int64).T
        neg = test_nv(x,y,nv)
        if neg.size > 0:
            raise ValueError('*Something* went wrong when trying to re-arrange the triangulation to clockwise direction :(')
        else:
            if verbose: print('- Triangulation from source was anti-clockwise. It is now clockwise')
    return nv

def test_nv(x,y,nv):
    # Triangle corners
    xpts  = x[nv]
    ypts  = y[nv]

    # Edges
    e1    = (xpts[:,1]-xpts[:,0])*(ypts[:,1]+ypts[:,0])
    e2    = (xpts[:,2]-xpts[:,1])*(ypts[:,2]+ypts[:,1])
    e3    = (xpts[:,0]-xpts[:,2])*(ypts[:,0]+ypts[:,2])

    # Direction test
    loop  = e1+e2+e3
    neg   = np.where(loop < 0)[0]

    return neg

@jit(nopython = True)
def get_art(VX, VY, NV, NT):
    ART = np.zeros((NT), dtype = np.float64)
    for I in range(NT):
        ART[I] = (VX[NV[I,1]]-VX[NV[I,0]])*(VY[NV[I,2]]-VY[NV[I,0]]) - (VX[NV[I,2]]-VX[NV[I,0]])*(VY[NV[I,1]]-VY[NV[I,0]])

    return np.abs(0.5*ART)

@jit(nopython = True)
def get_art1(NT, MT, VX, VY, XC, YC, NV, ISONB, NTVE, NBVE, NBVT, NBSN):
    '''
    Computes the area of a control volume

    Gives results that are very similar to output from FVCOM, but ~1-2% off - however seems to cancel
    when integrated over the entire domain. This could suggest that something is wrong, but where?
    '''
    # Set aside some storage
    ART1 = np.zeros((MT), dtype = np.float64)
    XX   = np.zeros((2*np.max(NTVE)+2), dtype = np.float64)
    YY   = np.zeros((2*np.max(NTVE)+2), dtype = np.float64)

    # Calculate area of control volume
    for I in range(MT):
        if ISONB[I] == 0: # for nodes that are not connected to the boundary
            for J in range(1,NTVE[I]+1):
                II = NBVE[I,J-1] # Start at the first listed cell nearby
                J1 = NBVT[I,J-1] # Triangle nearby

                # Cycle clockwise
                J2 = int(J1+1-np.floor((J1+2)/4)*3)

                XX[2*J-2] = (VX[NV[II,J1]]+VX[NV[II,J2]])*0.5-VX[I] # dx to midpoint on triangle wall
                YY[2*J-2] = (VY[NV[II,J1]]+VY[NV[II,J2]])*0.5-VY[I] # dy to midpoint on triangle wall
                XX[2*J-1] = XC[II]-VX[I] # dx from node to cell
                YY[2*J-1] = YC[II]-VY[I] # dy from node to cell

            XX[2*NTVE[I]] = XX[0]
            YY[2*NTVE[I]] = YY[0]

            for J in range(1,2*NTVE[I]+1):
                ART1[I] += 0.5*(XX[J]*YY[J-1]-XX[J-1]*YY[J])

            ART1[I] = np.abs(ART1[I])

        else: # for nodes that are connected to the boundary
            for J in range(1,NTVE[I]+1):
                II = NBVE[I,J-1]
                J1 = NBVT[I,J-1]
                J2 = int(J1+1-np.floor((J1+2)/4)*3)
                XX[2*J-2] = (VX[NV[II,J1]]+VX[NV[II,J2]])*0.5-VX[I]
                YY[2*J-2] = (VY[NV[II,J1]]+VY[NV[II,J2]])*0.5-VY[I]
                XX[2*J-1] = XC[II]-VX[I]
                YY[2*J-1] = YC[II]-VY[I]

            J  = NTVE[I]+1
            II = NBVE[I,J-2]
            J1 = NBVT[I, NTVE[I]-1]
            J2 = int(J1+2-np.floor((J1+3)/4)*3) # cycle counter-clockwise

            XX[2*J-2] = (VX[NV[II,J1]]+VX[NV[II,J2]])*0.5-VX[I]
            YY[2*J-2] = (VY[NV[II,J1]]+VY[NV[II,J2]])*0.5-VY[I]

            # this increment must be zero, I guess we just try to keep to the same routine as for CVs that don't connect to land
            XX[2*J-1] = 0 # VX(I)-VX(I)
            YY[2*J-1] = 0 # VY(I)-VY(I)

            XX[2*J] = XX[0]
            YY[2*J] = YY[0]

            for J in range(1,2*NTVE[I]+3):
                ART1[I] += 0.5*(XX[J]*YY[J-1]-XX[J-1]*YY[J])

            ART1[I] = np.abs(ART1[I])
    return ART1

def get_art2(ART1, MT, NBVE, NTVE):
    '''
    Area of triangles surrounding a node, get back to it later
    '''
    raise NotImplementedError('Not yet included in tge')

# Get vorticity
@jit(nopython = True, parallel = True)
def vorticity_node(U, V, NCV, MT, NTRG, NIEC, DLTXE, DLTYE, ART1, ISONB, NTSN, NBSN):
    '''
    Calculate vorticity using the control volume, based on calc_vort.F

    Procedure:
    In a point:   vort = nabla x u = dv/dx - du/dy
    Over an area: iint(vort) = --> greens theorem --> = int(udx + vdy)

    The direction of integration is by convention in the counter-clockwise
    direction, but FVCOM go clockwise - so we must use the negative value
    of the summation.
    '''
    VORT = np.zeros((MT), dtype = np.float32)
    LIST = np.zeros((MT), dtype = np.float32)
    AVE  = np.zeros((MT), dtype = np.float32)
    for I in prange(NCV):
        # Triangle I1 is conncedted to CV wall I
        I1 = NTRG[I]

        # The nodes that connect to this wall (NCV are the number of independent edges)
        IA  = NIEC[I,0]
        IB  = NIEC[I,1]

        # Vorticity flux through this wall
        EXFLUX = -(U[I1]*DLTXE[I] + V[I1]*DLTYE[I])

        # "add vorticity" to this wall
        VORT[IA] -= EXFLUX
        VORT[IB] += EXFLUX

    VORT = VORT/ART1

    # Correction at boundaries (how does this really work?)
    # ----
    for I in range(MT):
        if ISONB[I] > 0:
            AVE  = np.float32(0)
            CNT  = np.float32(0)
            for J in range(NTSN[I]):
                JNODE = NBSN[I,J]
                if JNODE != I and ISONB[JNODE] == 0:
                    AVE += VORT[JNODE]
                    CNT += np.float32(1)
            if not CNT:
                LIST[I] = np.float32(1)
            else:
                VORT[I] = AVE/CNT

    for I in range(MT):
        if LIST[I] > 0:
            AVE = np.float32(0)
            CNT = np.float32(0)
            for J in range(NTSN[I]):
                JNODE = NBSN[I,J]
                if JNODE != I and LIST[JNODE] == 0:
                    AVE += VORT[JNODE]
                    CNT += 1
            VORT[I] = AVE/CNT
    return VORT

# --------------------------------------------------------------------------------------------------------
#                                             Get gradients
# --------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def node_gradient(FIELD, VX, VY, MT, NBSN, NTSN, ISONB):
    '''
    When calculating node gradients, we first use least squares to fit a plane to the data
    surrounding our node, and then we return the best-fit approximation for the gradient
    evaluated at node I.

    Let: A       = [[(X1-x0), (Y1-y0)], [(X2-x0), (Y2-y0)], ...]
         grad(f) = [df/dx, df/dy] (evaluated at (x0, y0))
         delta_f = [(f1-f0), (f2-f0), ...]

    A*grad(f)     = delta_f
    A^T*A*grad(f) = A^T * delta_f
    grad(f)       = (A^T*A)^(-1)*A^T * delta_f

    hes@akvaplan.niva.no
    '''
    grad_f = np.zeros((MT, 2), dtype = np.float32)
    for I in range(MT):
        # Find the surrounding nodes (using list of indices used to find control volume)
        if ISONB[I] > 0:
            surrounding_nodes = NBSN[I, 1:NTSN[I]-1] # Since we don't need to look at the same node twice

        else:
            # And we need some boundary treatment
            surrounding_nodes = NBSN[I, :NTSN[I]-1] # Since we don't need to look at the same node twice

        # Then find distance from node I to each surrounding node and store as matrix
        X_surrounding = VX[surrounding_nodes]
        Y_surrounding = VY[surrounding_nodes]

        # Find distance matrix
        XDST = X_surrounding-VX[I]
        YDST = Y_surrounding-VY[I]
        A    = np.column_stack((XDST, YDST))

        # Then do some matrix multiplication
        ATA         = np.dot(A.T, A)
        ATA_inv     = np.linalg.inv(ATA)

        # Next we compute the right hand side
        data        = FIELD[surrounding_nodes]-FIELD[I]
        ATFIELD     = np.dot(A.T, data)

        # And compute the gradient
        grad_f[I,:] = np.dot(ATA_inv, ATFIELD)

    return grad_f

@jit(nopython=True)
def cell_gradient(FIELD, XC, YC, NT, NBE, ISBCE, grid_points):
    '''
    Calculate gradient of cell-data using a least-squares approach (come back to the FVCOM method
    on a later time, if needbe)
    '''
    grad_f = np.zeros((len(XC), 2), dtype = np.float32)
    for ind, I in enumerate(grid_points):
        # Find the surrounding nodes (using list of indices used to find control volume)
        if ISBCE[I] > 0:
            for J in range(3):
                if NBE[I,J] == -1:
                    surrounding_cells = np.zeros((2,), dtype = np.int64)
                    # Cycle to the non-negative indices
                    J1 = int(J+2-np.floor((J+3)/4)*3)
                    J2 = int(J+1-np.floor((J+2)/4)*3)
                    surrounding_cells[0] = NBE[I,J1]
                    surrounding_cells[1] = NBE[I,J2]
                    break

        else:
            # And we need some boundary treatment
            surrounding_cells = NBE[I,:] # Since we don't need to look at the same node twice

        # Then find distance from node I to each surrounding node and store as matrix
        X_surrounding = XC[surrounding_cells]
        Y_surrounding = YC[surrounding_cells]

        # Find distance matrix
        XDST = X_surrounding-XC[I]
        YDST = Y_surrounding-YC[I]
        A    = np.column_stack((XDST, YDST))

        # Then do some matrix multiplication
        ATA         = np.dot(A.T, A)
        ATA_inv     = np.linalg.inv(ATA)

        # Next we compute the right hand side
        data        = FIELD[surrounding_cells]-FIELD[I]
        ATFIELD     = np.dot(A.T, data)

        # And compute the gradient
        grad_f[I,:] = np.dot(ATA_inv, ATFIELD)

    return grad_f

@jit(nopython = True)
def fvcom_cell_gradients_speed(NT, NBE, a1u, a2u, U, V, grid_points):
    '''
    Calculate and return cell gradients using the same routines as FVCOM
    '''
    grad_u = np.zeros((NT,2), dtype = np.float64)
    grad_v = np.zeros((NT,2), dtype = np.float64)
    for I in grid_points:
        if NBE[I,0] == -1:
            u1k1 = 0; u2k1 = 0
            v1k1 = 0; v2k1 = 0
        else:
            u1k1 = a1u[I,1]*U[NBE[I,0]]; u2k1 = a2u[I,1]*U[NBE[I,0]]
            v1k1 = a1u[I,1]*V[NBE[I,0]]; v2k1 = a2u[I,1]*V[NBE[I,0]]

        if NBE[I,1] == -1:
            u1k2 = 0; u2k2 = 0
            v1k2 = 0; v2k2 = 0
        else:
            u1k2 = a1u[I,2]*U[NBE[I,1]]; u2k2 = a2u[I,2]*U[NBE[I,1]]
            v1k2 = a1u[I,2]*V[NBE[I,1]]; v2k2 = a2u[I,2]*V[NBE[I,1]]

        if NBE[I,2] == -1:
            u1k3 = 0; u2k3 = 0
            v1k3 = 0; v2k3 = 0
        else:
            u1k3 = a1u[I,3]*U[NBE[I,2]]; u2k3 = a2u[I,3]*U[NBE[I,2]]
            v1k3 = a1u[I,3]*V[NBE[I,2]]; v2k3 = a2u[I,3]*V[NBE[I,2]]

        # u-vel
        # ----
        grad_u[I,0] = a1u[I,0]*U[I] + u1k1 + u1k2 + u1k3
        grad_u[I,1] = a2u[I,0]*U[I] + u2k1 + u2k2 + u2k3

        # v-vel
        # ----
        grad_v[I,0] = a1u[I,0]*V[I] + v1k1 + v1k2 + v1k3
        grad_v[I,1] = a2u[I,0]*V[I] + v2k1 + v2k2 + v2k3

    return grad_u, grad_v

@jit(nopython = True)
def fvcom_cell_gradients_scalar(NT, NBE, a1u, a2u, F, grid_points):
    '''
    Calculate and return cell gradients using the same routines as FVCOM
    '''
    grad = np.zeros((NT,2), dtype = np.float64)
    for I in grid_points:
        if NBE[I,0] == -1:
            u1k1  = 0; u2k1 = 0
        else:
            u1k1  = a1u[I,1]*F[NBE[I,0]]; u2k1 = a2u[I,1]*F[NBE[I,0]]

        if NBE[I,1] == -1:
            u1k2  = 0; u2k2 = 0
        else:
            u1k2  = a1u[I,2]*F[NBE[I,1]]; u2k2 = a2u[I,2]*F[NBE[I,1]]

        if NBE[I,2] == -1:
            u1k3  = 0; u2k3 = 0
            v1k3  = 0; v2k3 = 0
        else:
            u1k3  = a1u[I,3]*F[NBE[I,2]]; u2k3 = a2u[I,3]*F[NBE[I,2]]

        # field gradient
        grad[I,0] = a1u[I,0]*F[I] + u1k1 + u1k2 + u1k3
        grad[I,1] = a2u[I,0]*F[I] + u2k1 + u2k2 + u2k3

    return grad

def vertical_gradient(f, h, sigma):
    '''
    Calculate the simplest possible df/dz

    f     = [lay, grid]
    h     = [1,   grid]
    sigma = [lay]

    ==> let s be sigma layers
    df/dz = df/dz ds/dz. We know that ds/dz is 1/H (where H is depth), since both are monotinicly increasing functions

    Then we can use the very accessible numpy library to estimate the vertical gradient

    Note that this routine will ONLY be valid for grids where the sigma-steps are identical at
    every grid point.
    '''
    return np.gradient(f, sigma, axis = 0)/h

@jit(nopython=True)
def streamfunction(MT, VX, VY, NBSN, NTVE, NTSN, NBVE, NBVT, ISONB, u, v, threshold = 1.0, boundary_conditions = False):
    '''
    Compute the streamfunction for a given velocity-field using a "least squares-like" approach
    Boundary condition: psi = 0 at nodes connected to land

    Iterate until convergence (not quite sure what a good criteria is)

    Does not work well at the moment :)
    '''
    # Streamfunction (actually an initialization)
    derivative = np.zeros((MT), dtype = np.float32)

    # Find boundary nodes
    boundary   = np.where(ISONB==1)[0]

    # First solve the derivative
    for I in range(MT):
        if ISONB[I] == 0: # If node is in the interior
            for J in range(NTSN[I]-1):
                dx = VX[NBSN[I,J]] - VX[I]
                dy = VY[NBSN[I,J]] - VY[I]

                # Make sure to use the right velocity...
                # --> Find left and right triangle
                if J == 0:
                    left  = NBVE[I,NTVE[I]-1]
                    right = NBVE[I,J]

                elif J == NTVE[I]:
                    left  = NBVE[I,J-1]
                    right = NBVE[I,0]

                else:
                    left  = NBVE[I,J-1]
                    right = NBVE[I,J]

                derivative[I] += (0.5*(u[left]+u[right])*dy - 0.5*(v[left]+v[right])*dx)/(NTSN[I]-1)

        elif ISONB[I]>0:
            for J in range(1,NTSN[I]-1):
                # Need special treatment, but still a gridpoint where the streamfunction can be /= 0
                dx = VX[NBSN[I,J]] - VX[I]
                dy = VY[NBSN[I,J]] - VY[I]

                # Make sure to use the right velocity...
                # --> Find left and right triangle
                if J == 1:
                    left  = NBVE[I,J-1]
                    right = NBVE[I,J-1]

                elif J == NTVE[I]:
                    left  = NBVE[I,J-2] # Dobbeltsjekk at dette blir rett (er jo dønn likt den andre...)
                    right = NBVE[I,J-1]

                else:
                    left  = NBVE[I,J-2]
                    right = NBVE[I,J-1]

                derivative[I] += (0.5*(u[left]+u[right])*dy - 0.5*(v[left]+v[right])*dx)/(NTSN[I]-2)

    # Now that we have the derivative matrix, we can prepare the successive over-relaxation iteration step
    # Ax = b
    # ----
    psi          = np.zeros((MT), dtype = np.float32)
    iterations   = 0
    while True:
        psi_old  = np.copy(psi)
        for I in range(MT):
            if ISONB[I] == 0:
                psi[I] = np.mean(psi[NBSN[I,:NTSN[I]-1]])  - derivative[I]

            elif ISONB[I] >= 1:
                psi[I] = np.mean(psi[NBSN[I,1:NTSN[I]-1]]) - derivative[I]

        # Stronger homogenization for boundary nodes
        # --------------------------------------------------
        if boundary_conditions:
            if iterations > 100: # Enforce them later on in the calculation
                for N in range(30):
                    for K in boundary:
                        if ISONB[K] == 1:
                            for J in range(1, NTSN[K]-1):
                                if ISONB[NBSN[K,J]] == 1:
                                    #psi[K] = 0.5*(psi[K]+psi[NBSN[K,J]])
                                    psi[K] = 0


        # Stop the iteration if it has converged sufficiently, or if it has been running for too long
        # -------------------------------------------------
        if iterations > 100:
            if np.nanmax(np.abs(psi-psi_old))<threshold :#np.nanmax(np.abs(psi_old-psi)/psi) < threshold:
                break

        iterations += 1
        if iterations>2000:
            print('Slow convergence, we return this result as a "best guess"')
            break

    return psi

# -------------------------------------------------------------------------------------------------------
#     Class that will function as a return of the function, and that can be used to load data into
# -------------------------------------------------------------------------------------------------------
class TGE():
    def __init__(self, fname = None):
        '''
        TGE is the output from the function tge.main. This class can also be loaded individually from tge
        and read a .npy file with TGE data.
        '''
        if fname:
            if fname.split('.')[-1] == 'npy':
                self.load_data_npy(fname)
            elif fname.split('.')[-1] == 'mat':
                self.load_data_mat(fname)
            self.get_art1()

    def load_data_npy(self, fname):
        tge    = np.load(fname, allow_pickle=True)
        fields = ['VX', 'VY', 'XC', 'YC', 'NV', 'MT', 'NT', 'NCV', 'NBE', 'NBSN', 'NTSN', 'ISONB', 'ISBCE',
                  'NBVE','NBVT', 'NTVE', 'NE', 'IEC', 'IENODE', 'ISBC', 'XIJE', 'YIJE', 'NTRG',
                  'NIEC', 'DLTXE', 'DLTYE', 'DLTXYE', 'SITAE', 'DLTXC', 'DLTYC', 'XIJC', 'YIJC',
                  'DLTXYC', 'SITAC', 'A1U', 'A2U', 'source', 'author', 'date']

        for key in fields:
            setattr(self, key, tge.item()[key])


    def load_data_mat(self, fname):
        from scipy.io import loadmat
        mat_tge = loadmat(fname)
        fields  = ['NBE', 'NBSN', 'NTSN', 'ISONB', 'ISBCE', 'NBVE', 'NBVT', 'NTVE', 'NE', 'IEC', 'ISBC',
                   'IENODE', 'NIEC', 'XIJC', 'YIJC', 'DLTXYC', 'DLTXC', 'DLTYC', 'NTRG', 'DLTXE', 'DLTYE']
        m1inds  = ['NBE', 'NBSN', 'NBVE', 'NBVT']
        for key in fields:
            shape = mat_tge[key.lower()].shape
            if max(shape) > 1:
                if min(shape) > 1:
                    if key in m1inds:
                        setattr(self, key, mat_tge[key.lower()].astype(np.int32)-1)
                    else:
                        setattr(self, key, mat_tge[key.lower()].astype(np.int32)-1)
                else:
                    setattr(self, key, mat_tge[key.lower()].ravel().astype(np.int32))

            else:
                setattr(self, key, mat_tge[key.lower()][0,0].astype(np.int32))

    def write_data(self, fname = 'tge.npy'):
        try:
            self.author = os.getlogin()
        except:
            self.author = 'compute node'
        self.date   = date.today()
        datakeys = ['VX', 'VY', 'XC', 'YC', 'NV', 'MT', 'NT', 'NBE', 'NBSN', 'NTSN', 'ISONB', 'ISBCE',
                    'NBVE','NBVT', 'NTVE', 'NE', 'IEC', 'IENODE', 'ISBC', 'XIJE', 'YIJE', 'NTRG',
                    'NIEC', 'DLTXE', 'DLTYE', 'DLTXYE', 'SITAE', 'DLTXC', 'DLTYC', 'XIJC', 'YIJC',
                    'DLTXYC', 'SITAC', 'NCV', 'A1U', 'A2U','source', 'author', 'date']
        outdict  = {}
        for key in datakeys:
            outdict[key] = getattr(self, key)
        np.save(fname, outdict)

    def get_art1(self, verbose = True):
        '''
        Get the area of each CV
        '''
        self.art1 = get_art1(self.NT, self.MT, self.VX, self.VY, self.XC, self.YC,
                             self.NV, self.ISONB, self.NTVE, self.NBVE, self.NBVT, self.NBSN)

    def qc_tge(self, I):
        plot_CV(self.M, self.NBSN, self.NTSN, self.NBVE, self.NTVE, I)

    def qc_tge_dir(self, I):
        plot_CV_points(self.M, self.NBSN, self.NTSN, self.NBVE, self.NTVE, I)

    def vorticity(self, u, v):
        return vorticity_node(u, v, self.NCV, self.MT, self.NTRG, self.NIEC,
                              self.DLTXE, self.DLTYE, self.art1, self.ISONB,
                              self.NTSN, self.NBSN)

    def vorticity_3D(self, u, v, h, sigma, grid_points = None, verbose = False):
        '''
        Calculate vorticity in all sigma layers, convert to z-coordinate
        u, v  = [siglay, grid]
        h     = [grid]
        sigma = [siglay, grid]

        You can specify grid_points to reduce the size of the grid you wish to do the calculation on.
        '''
        # If grid is not cropped
        # ----
        if grid_points is None:
            grid_points = np.arange(len(u[0,:]), dtype = np.int32)

        # Prepare vorticity and gradient storage
        # ----
        vorticity     = np.zeros(u[:,grid_points].shape).T

        # Vertical gradients
        # ----
        if verbose: print('  - Vertical gradient')
        dudz        = self.vertical_gradient(u[:,grid_points], h[grid_points][None,:], sigma[:,0])
        dvdz        = self.vertical_gradient(v[:,grid_points], h[grid_points][None,:], sigma[:,0])
        z           = h*sigma

        # Then horizontal gradients in z-coordinates
        # ----
        if verbose: print('  - Horizontal gradient')
        for i in range(len(sigma)):
            if verbose: print('   - in sigma layer: '+str(i))

            # Find grad u, v and depth in constant sigma layers
            grad_uv    = self.fvcom_cell_gradient(u[i,:], v = v[i,:], grid_points = grid_points)
            grad_z     = self.fvcom_cell_gradient(z[i,:], grid_points = grid_points)

            # Transform from sigma to z coordinates
            grad_uy = grad_uv['u'][:,1][grid_points] - dudz[i,:] * grad_z[:,1][grid_points]
            grad_vx = grad_uv['v'][:,0][grid_points] - dvdz[i,:] * grad_z[:,0][grid_points]

            # Compute vorticity in this sigma layer
            vorticity[:, i] = grad_vx - grad_uy

        return vorticity

    # --------------------------------------------------------------------------------------------------------------
    #                                               Find gradients
    # --------------------------------------------------------------------------------------------------------------
    def vertical_gradient(self, f, h, sigma):
        '''
        Vertical difference, centers the data to a new set of depth levels (for convenience)
        '''
        # -> Should not need other stuff than these
        return vertical_gradient(f, h, sigma)

    def node_gradient(self, f):
        '''
        Returns gradient of a node-based field.

        The routine is using a least squares method to fit a surface to the data around a node, the
        inclination of this surface is the gradient of the node based field.
        '''
        return node_gradient(f, self.VX, self.VY, self.MT, self.NBSN, self.NTSN, self.ISONB)

    def cell_gradient(self, f, grid_points = None):
        '''
        Returns gradient of cell-based data

        In linear algebra terms: A*grad_f = delta_f
                                 grad_f = (A_transpose * A)_inverse * A_transpose * delta_f

        where A = (delta_x, delta_y) from given triangle to surrounding ones, and delta_f is the difference in cell-
        based value between the surrounding triangles and the one considered.
        '''
        if grid_points is None:
            grid_points = np.arange(len(f), dtype = np.int32)
        return cell_gradient(f, self.XC, self.YC, self.NT, self.NBE, self.ISBCE, grid_points)

    def fvcom_cell_gradient(self, u, v = None, grid_points = None):
        '''
        Using the same script as FVCOM uses internally
        '''
        if grid_points is None:
            grid_points = np.arange(u.shape[0], dtype = np.int32)

        if v is None:
            grad = fvcom_cell_gradients_scalar(self.NT, self.NBE, self.A1U, self.A2U, u, grid_points = grid_points)

        else:
            grad = {}
            grad['u'], grad['v'] = fvcom_cell_gradients_speed(self.NT, self.NBE, self.A1U, self.A2U, u, v, grid_points = grid_points)

        return grad

    def okubo_weiss(self, u, v):
        '''
        Returns the okubo-weiss parameter as defined and described in Wekerle et. al. 2020
        https://doi.org/10.5194/os-16-1225-2020
        '''
        grad_uv = self.fvcom_cell_gradient(u, v = v, grid_points=None)

        # Okubo-weiss
        OW = (grad_uv['u'][:,0] - grad_uv['v'][:,1])**2 + \
             (grad_uv['v'][:,0] + grad_uv['u'][:,1])**2 - \
             (grad_uv['v'][:,0] - grad_uv['u'][:,1])**2

        # This would be "the spot" to put a limitation for the search

        # Standard deviation (how to weight this for area bias?)
        OW_0 = -0.2*np.std(OW)

        # FVCOM cells where eddy boolean vector satisfies the criteria
        eddies = OW < OW_0

        # Return dict
        out       = {}
        out['OW'] = OW
        out['eddies'] = eddies

        return out

    def streamfunction(self, u, v, threshold = 1, boundary_conditions = True):
        '''
        Compute the streamfunction for a dataset
        '''
        return streamfunction(self.MT, self.VX, self.VY,
                              self.NBSN, self.NTVE, self.NTSN, self.NBVE, self.NBVT,
                              self.ISONB, u, v, threshold = threshold, boundary_conditions = boundary_conditions)

    # Needed to compute standard deviation:
    # --------------------------
    def standard_deviation(self, data):
        '''
        cell area weighted standard deviation
        '''
        weight    = self.tri_area/self.tot_tri_area
        wmean     = np.sum(weight*data)
        deviation = data-wmean
        sum1      = np.sum(weight*(data-wmean)**2)
        return np.sqrt(len(data)*sum1/(len(data)-1))


    def tri_area(self):
        AB    = np.array([self.VX[self.NV[:,1]]-self.VX[self.NV[:,0]],
                          self.VY[self.NV[:,1]]-self.VY[self.NV[:,0]]]).transpose()
        AC    = np.array([self.VX[self.NV[:,2]]-self.VX[self.NV[:,0]],
                          self.VY[self.NV[:,2]]-self.VY[self.NV[:,0]]]).transpose()

        # Find their cross product
        ABxAC = np.cross(AB,AC)

        # Find the area
        self.tri_area  = 0.5*np.linalg.norm(ABxAC, axis=1)
        self.tot_tri_area = np.sum(self.tri_area)

# ------------------------------------------------------------------------------------------------------------------
#                                             Debug routines
# ------------------------------------------------------------------------------------------------------------------
# --> To plot positions of CV corners
def plot_CV_points(M, NBSN, NTSN, NBVE, NTVE, I):
    plt.scatter(M.x[I,0], M.y[I,0], c = 'r', label = 'centre node', zorder = 1001)
    for J in range(NTVE[I]):
        plt.scatter(M.xc[NBVE[I,J], 0], M.yc[NBVE[I,J], 0], c = 'g', label = 'cell', zorder = 1000)
        plt.pause(0.1)
    for J in range(NTSN[I]):
        plt.scatter(M.x[NBSN[I,J],0], M.y[NBSN[I,J],0], c = 'b', label = 'node', zorder = 1000)
        plt.pause(0.1)
    plt.legend()

def plot_CV(M, NBSN, NTSN, NBVE, NTVE, I):
    M.plot_grid()
    plt.scatter(M.x[NBSN[I,:NTSN[I]-1],0], M.y[NBSN[I,:NTSN[I]-1],0],
                c = 'b', label = 'CV nodes', zorder = 1000)
    plt.scatter(M.xc[NBVE[I,:NTVE[I]]], M.yc[NBVE[I, :NTVE[I]]],
                c = 'g', label = 'CV cells', zorder = 1000)
    plt.scatter(M.x[I,0], M.y[I,0], c = 'r', label = 'centre node', zorder = 1001)
    plt.legend()

class NotImplementedError(Exception):
    pass
