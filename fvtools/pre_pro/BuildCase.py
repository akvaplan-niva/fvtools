"""
--------------------------------------------------------------------------------------------------------
                            In development - Status: Testing (beta)
--------------------------------------------------------------------------------------------------------
BuildCase - Creates all the input files needed to initialize an FVCOM experiment
          - Works together with get_ngrd.py, obcgridding.py (obcgridding should be run before BC,
            get_ngrd should be run after)
--------------------------------------------------------------------------------------------------------
"""
global version_number
version_number = 2.2

# Modules:
# -------------------
import os
import numpy as np
import progressbar as pb
import fvtools.grid.fvgrid as fvg
import matplotlib.pyplot as plt

from pyproj import Proj, transform
from scipy.interpolate import LinearNDInterpolator
from matplotlib.tri import Triangulation
from datetime import datetime
from numba import jit, njit

# Routines:
# -------------------
def main(dmfile,
         dptfile,
         casename  = os.getcwd().split('/')[-1],
         reference = 'epsg:32633',
         nesting   = 'fvcom',
         rx0max    = 0.2,
         min_depth = 5.0,
         sponge_radius = 8000.0,
         sponge_factor = 0.001,
         make_dpt   = False):
    """
    Reads in the SMS unstructured grid formatted 2dm file.
    Prepares FVCOM input files

    Parameters
    -----------
    dmfile:    .2dm file containing the mesh info (type: string)
    dptfile:   .txt or .npy file containing the raw bathymetry (type: string)
               or just a number for constant depth
    nesting:   'fvcom' or 'roms'
    rx0max:    smoothing target
    min_depth: minimum depth to be fed to FVCOM
    casename:  Name of the _sigma.dat file in the input folder
    reference: Projection reference (UTM 33 by default, 'epsg:32633' or 'latlon')
               This will _only_ affect how the routine interprets the .2dm file,
               make sure to check that the bathymetry file is on the same format.
    make_dpt:  True if you don't intend to nest into another model (default: False)

    Returns:
    ----------
    - FVCOM input files
    - M.npy

    hes@akvaplan.niva.no
    """
    print('-------------------------------------------------------------------------')
    print(f'                   BuildCase: {casename}, {dmfile}')
    print('-------------------------------------------------------------------------')
    print('Prepare grid information')
    print('- Read '+dmfile)
    M = read_2dmfile(dmfile, casename)

    if reference == 'latlon':
        M['lon'] = M['x']; M['lat'] = M['y']
    else:
        print('- Project to lat-lon')
        M['lon'], M['lat'] = project(M['x'], M['y'], reference)

    print('- Determine sponge nodes')
    M['sponges']       = return_sponge_nodes(M['read_obc_nodes'])
    M['sponge_radius'] = sponge_radius
    M['sponge_fact']   = sponge_factor

    M['xc']     = triangulate(M, 'x')
    M['yc']     = triangulate(M, 'y')
    M['lonc']   = triangulate(M, 'lon')
    M['latc']   = triangulate(M, 'lat')

    print('- Read sigma coordinate file')
    M['siglev'], M['siglay'] = read_sigma(f'./input/{casename}_sigma.dat')

    print('- Find surrounding nodes and cells')
    M['nbsn'], M['ntsn'], M['edge'] = setup_metrics(M)

    print('\nPrepare the bathymetry')
    print('- Read the depth')
    M['h_raw'] = make_depth(M, dptfile, min_depth)

    print('- Smooth the topography')
    M['h']     = np.copy(M['h_raw'])
    M['h']     = smooth_topo(M, rx0max = rx0max, SmoothFactor = 0.2)
    show_depth(M)
    M['ts']    = get_cfl(M)

    print(f'\nWrite the {casename}_*.dat files')
    write_grd(M, casename)
    write_obc(M, casename)
    write_sponge(M, casename)
    write_cor(M, casename)
    if make_dpt:
        write_bath(M, casename)

    print('\nAdd documentation about this grid')
    M = add_dict_info(M, casename, dmfile, dptfile, reference)

    M['siglay']  = np.tile(M['siglay'],(len(M['x']), 1))
    M['siglayz'] = (M['h']*M['siglay'].T).T
    M['siglev']  = np.tile(M['siglev'],(len(M['x']), 1))
    M['siglevz'] = (M['h']*M['siglev'].T).T

    print('\nSave the M-dict')
    np.save('M.npy',M)
    if make_dpt:
        print('\nFin')
    else:
        print('\nFin.\n- Run get_ngrd to get a bathymetry file and nest-grid info.')

# ============================================================================
#                     Visualize the smoothed bathymetry
# ============================================================================
def show_depth(M):
    """
    Plot the raw, smoothed and difference between the two of them

    Parameters:
    ----
    A dict (M) with h_raw, h and grid info in it.

    Out:
    ----
    Returns a plot of the raw bathymetry
    """
    increment = np.int(max(M['h_raw'])/100)*10
    if increment == 0:
        increment = 2

    levels = np.arange(0,max(M['h_raw']),increment)
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15,8))

    cdat = ax[0].tricontourf(M['x'], M['y'], M['nv'], M['h_raw'], 100, cmap = 'terrain')
    ax[0].tricontour(M['x'], M['y'], M['nv'], M['h_raw'], levels = levels, colors = 'k')
    ax[0].tricontourf
    ax[0].axis('equal')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('raw bathymetry')
    plt.colorbar(cdat, ax = ax[0])

    cdat = ax[1].tricontourf(M['x'], M['y'], M['nv'], M['h'], 100, cmap = 'terrain')
    ax[1].tricontour(M['x'], M['y'], M['nv'], M['h'], levels = levels, colors = 'k')
    ax[1].axis('equal')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('smoothed bathymetry')
    plt.colorbar(cdat, ax = ax[1])

    cdat = ax[2].tricontourf(M['x'], M['y'], M['nv'], M['h_raw']-M['h'], 100, cmap = 'terrain')
    ax[2].axis('equal')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title('raw-smoothed bathymetry')
    plt.colorbar(cdat, ax = ax[2])

# ===================================================================================
#                              Reading the mesh file
# ===================================================================================
def project(x, y, reference):
    '''
    Project the positions from latlon to UTM(reference)
    (can be upgraded to other reference systems in the future
    '''
    WGS84 = Proj('epsg:4326')
    UTM   = Proj(reference)
    lon, lat = transform(UTM, WGS84, x, y, always_xy = True)
    return lon, lat

def read_2dmfile(dmfile, casename):
    M = {}
    M['nv'], _, M['x'], M['y'], _, _, M['read_obc_nodes'] = fvg.read_sms_mesh(dmfile, nodestrings = True)

    # Make sure that read_obc_nodes has the same format as the data from matlab
    # ----
    ron = np.ndarray((1,len(M['read_obc_nodes'])), dtype = object)
    for n in range(len(M['read_obc_nodes'])):
        ron[0,n] = np.array([M['read_obc_nodes'][n]], dtype = np.int32)

    # This is equivalent to loading a cell-array using scip.io.loadmat
    # ----
    M['read_obc_nodes'] = ron

    # Check the mesh-boundary quality
    # ----
    print('- Check boundary triangles')
    M, update_obc = check_obc(M)
    if update_obc:
        not_plotted = False
    else:
        not_plotted = True

    M, update_coast = check_coast(M, not_plotted)
    if update_obc or update_coast:
        update_grid(M, casename)

    # Count the number of obc nodes
    # ----
    M['num_obc'] = 0
    for i in range(len(M['read_obc_nodes'][0,:])):
        M['num_obc'] += len(M['read_obc_nodes'][0,i][0])

    return M

def return_sponge_nodes(obc_nodes):
    '''
    Sponge from the obc
    '''
    sponges = []
    for i in range(obc_nodes.shape[1]):
        for node in obc_nodes[0,i][0]:
            sponges.append(node)
    return np.array(sponges)

def add_dict_info(M, case, dm, depth, reference):
    """
    Add some info so that it is easy to tell which version of M this is
    """
    info = {}
    info['created']    = datetime.now().strftime('%Y-%m-%d at %H:%M h')
    info['2dm file']   = dm
    info['depth file'] = depth
    info['author']     = os.getlogin()
    info['directory']  = os.getcwd()
    info['scipt version'] = version_number
    info['casename']   = case
    info['reference']  = reference
    M['info'] = info
    return M

def check_obc(M):
    '''
    Updates the grid and writes a new casename.2dm file if there are illegal boundary triangles.
    ________
    \/\/\/\/
       \/ <- illegal, would be removed.
    '''
    read_obc_nodes = np.copy(M['read_obc_nodes'])
    x  = np.copy(M['x']); y = np.copy(M['y']); nv = np.copy(M['nv'])
    already_checked = False; update = False

    # Loops over the grid until we get one that satisfies the no-double-boundary-triangle criteria
    while True:
        # Get a numpy array with all obc nodes in one dimension
        # ----
        obc_nodes = np.empty(0, dtype = np.int32)
        for i in range(len(read_obc_nodes[0,:])):
            obc_nodes = np.append(obc_nodes, read_obc_nodes[0,i])

        # Tagg all obc nodes with 1
        # ----
        node_tag = np.zeros(len(x))
        node_tag[obc_nodes] = 1

        # Check how many nodes each triangle has on the open boundary
        # ----
        n_obc    = np.sum(node_tag[nv], axis = 1)
        bad_tri, good_tri, good, triangle = identify_problems_in_grid(n_obc, x, y, nv, already_checked, 2, 'OBC')
        if good:
            break

        # Need to remove the bad triangle and the bad node and create a new .2dm file
        # ----
        xc = np.mean(x[nv],axis=1); yc = np.mean(y[nv],axis=1)
        plt.scatter(xc[bad_tri], yc[bad_tri], label = 'bad obc '+triangle, zorder = 10)
        x, y, nv, read_obc_nodes = remap(x, y, nv, read_obc_nodes, good_tri)

        # Prepare for next loop
        # ---
        already_checked = True
        update = True

    # Update the mesh info if we had to change something
    # ----
    if update:
        M['x'] = x; M['y'] = y; M['nv'] = nv; M['read_obc_nodes'] = read_obc_nodes

    return M, update

def check_coast(M, not_plotted = True):
    '''
    Remove bad triangles along the coast (these often appear in smeshing results)
    --> Normally very isolated errors, I wouldn't be too conserned if these
    '''
    read_obc_nodes = np.copy(M['read_obc_nodes'])
    x  = np.copy(M['x']); y = np.copy(M['y']); nv = np.copy(M['nv'])
    already_checked = False; update = False

    # Loops over the grid until we get one that satisfies the no-double-boundary-triangle criteria
    while True:
        # Get a numpy array with all obc nodes in one dimension
        # ----
        NBE       = get_NBE(len(nv[:,0]), len(x), nv)
        land      = np.zeros(NBE.shape)
        land[NBE == -1] = 1
        sides     = np.sum(land, axis = 1)
        bad_tri, good_tri, good, triangle = identify_problems_in_grid(sides, x, y, nv, already_checked, 1, 'coast', not_plotted)
        if good:
            break

        # Need to remove the bad triangle and the bad node and create a new .2dm file
        # ----
        xc = np.mean(x[nv],axis=1); yc = np.mean(y[nv],axis=1)
        plt.scatter(xc[bad_tri], yc[bad_tri], label = 'bad coast '+triangle, zorder = 10)

        # Remove bad triangles (and nodes), return cropped grid and new triangulation
        x, y, nv, read_obc_nodes = remap(x, y, nv, read_obc_nodes, good_tri)

        # Prepare for next loop
        # ---
        already_checked = True
        update          = True
        not_plotted     = False

    # Update mesh info if there were problems
    # ----
    if update:
        M['x'] = x; M['y'] = y; M['nv'] = nv; M['read_obc_nodes'] = read_obc_nodes

    return M, update

def identify_problems_in_grid(sides, x, y, nv, already_checked, critval, name, not_plotted = True):
    '''
    Identify triangles we need to change
    '''
    bad_tri   = np.where(sides > critval)[0]
    good_tri  = np.where(sides < critval+1)[0]
    triangle  = ''

    # Check if there are bad triangles in this mesh:
    # ----
    if bad_tri.shape[0] > 0:
        good = False
        # then we need to remove a triangle from the triangulation
        # -----
        n_gt3 = len(bad_tri)
        if not already_checked:
            if not_plotted:
                plt.figure()
                plt.triplot(x, y, nv, label='before')
                plt.axis('equal')
            if n_gt3 > 1:
                triangle = 'triangles'
                tring = triangle+' have'
            else:
                triangle = 'triangle'
                tring = triangle+' has'
            print(f'  - {n_gt3} {tring} more than one side connected to the {name}.\n'+\
                   '    Removing illegal triangles and moving on.')

    else:
        good = True
        if not already_checked:
            print(f'  - No triangles with more than one side toward {name}')

    return bad_tri, good_tri, good, triangle

def remap(x, y, nv, read_obc_nodes, good_tri):
    '''
    Returns a remapped version of the mesh
    '''
    # See which nodes to keep (node_ind) and define new node indices
    # ----
    node_ind     = np.unique(nv[good_tri])
    new_node_ind = np.arange(len(node_ind))

    # Create a map from old indexing to new
    # ----
    all_nodes    = np.nan*np.ones((len(x)), dtype = np.int32)
    all_nodes[node_ind] = new_node_ind

    # Remap the triangles and the positions
    # ----
    nv  = all_nodes[nv[good_tri]].astype(np.int32)
    x   = x[node_ind]
    y   = y[node_ind]

    # Update index of read_obc_nodes
    # ----
    new_read_obc_nodes = np.ndarray((1,len(read_obc_nodes[0,:])), dtype = object)
    for i in range(len(read_obc_nodes[0,:])):
        tmp = all_nodes[read_obc_nodes[0,i]]
        new_read_obc_nodes[0,i] = tmp[~np.isnan(tmp)][None,:].astype(np.int32)

    # Prepare for next loop
    # ----
    read_obc_nodes  = np.copy(new_read_obc_nodes)

    # Look over to check that everything went fine
    # ----
    if np.max(nv) == np.nan:
        raise RemapError('Something went wrong when re-mapping. Check that the grid indexing was ok to start with.')

    for i in range(len(read_obc_nodes[0,:])):
        tmp = read_obc_nodes[0,i]
        if np.max(tmp) == np.nan:
            raise RemapError('Something went wrong when re-mapping. Check that the grid indexing was ok to start with.')

    return x, y, nv, read_obc_nodes

def update_grid(M, casename = 'BuildCase'):
    '''
    Update the .2dm file if the grid has been changed
    '''
    # Visualize:
    # ----
    filename = casename+'_corrected'
    plt.triplot(M['x'],M['y'],M['nv'],label='after')
    plt.title('before and after removing bad triangles')
    plt.legend(loc = 'upper right')

    # Then we write it!
    # ----
    print(f'  --> Writing corrected FVCOM grid to {filename}.2dm')
    fvg.write_2dm(M['x'], M['y'], M['nv'], read_obc_nodes = M['read_obc_nodes'], name = filename, casename = casename)

# ===================================================================================
#          Handle bathymetry (loading, interpolation and smoothing)
# ===================================================================================
def make_depth(M, dptfile, min_depth):
    """
    Interpolate data from a "depth.txt" file to a FVCOM mesh

    Parameters:
    -----------
    - M:       Dict containing x- and y positions (type: ndarray)
    - dptfile: Text file containing depths  (type: x,y,z,info)
               or just a number for constant depth
    - min_depth: Sets values less than min_depth equal to min_depth

    Returns:
    -----------
    - interpolated depth
    - Parameters giving measures of the mesh quality
    """

    # Load the depth data
    if type(dptfile) == int or type(dptfile) == float:
        h_raw = dptfile*np.ones((M['x'].shape))

    else:
        if dptfile.split('.')[-1] == 'npy':
            depth = load_numpybath(dptfile)
        else:
            depth = load_textbath(dptfile)

        # Crop the data so that we only have data covering the FVCOM domain
        depth = crop(M, depth)

        # Interpolate the bathymetry to the mesh
        h_raw = interpolate_to_mesh(depth, M, min_depth = min_depth)

    if np.argwhere(np.isnan(h_raw)).size >0:
        print('Nan found in h_raw. Nans are set to min_depth value. '+\
              'Consider using a better bathymetry that has good coverage of the entire domain.'+\
              'Number of nans found: ',np.argwhere(np.isnan(h_raw)).size)
        h_raw[np.argwhere(np.isnan(h_raw))]=min_depth
    return h_raw

def load_numpybath(dptfile):
    """
    Load the bathymetry from a big numpy file
    """
    try:
        depth_data = np.load(dptfile)
    except:
        depth_data = np.load(dptfile, allow_pickle = True)

    return depth_data

def load_textbath(dptfile):
    """
    Load the depth file
    - .txt files are "raw", ie. has not been processed by any BuildCase runs before. (slow to load)
    - .npy files are already in a BuildCase friendly format, and don't take too long to load
    """
    # Load raw data
    print('-> Load: '+dptfile)
    try:
        depth_data = np.loadtxt(dptfile, delimiter=',')
    except:
      try:
          depth_data = np.loadtxt(dptfile)
      except:
        try:
            depth_data = np.loadtxt(dptfile, delimiter=' ')
        except:
            depth_data = np.loadtxt(dptfile, skiprows = 1, delimiter = ',', usecols = [0,1,2])

    numpy_bath_name = dptfile.split('.txt')[0]+'.npy'
    print('- Storing the full bathymetry in: ' + numpy_bath_name)
    np.save(numpy_bath_name, depth_data)

    return depth_data

def crop(M, depth_data):
    """
    Crop the depth file to your domain
    """
    if max(depth_data[:,0]) < max(M['x']) or min(depth_data[:,0]) > min(M['x']):
        raise ValueError('The bathymetry file does not cover the model domain!')

    if max(depth_data[:,1]) < max(M['y']) or min(depth_data[:,1]) > min(M['y']):
        raise ValueError('The bathymetry file does not cover the model domain!')

    print('- Crop the bathymetry data')
    ind1 = np.logical_and(depth_data[:,0] >= min(M['x'])-5000.0, depth_data[:,0] <= max(M['x'])+5000.0)
    ind2 = np.logical_and(depth_data[:,1] >= min(M['y'])-5000.0, depth_data[:,1] <= max(M['y'])+5000.0)
    ind  = np.logical_and(ind1, ind2)

    # Store it and return
    depth = {}
    depth['x'] = depth_data[ind,0]
    depth['y'] = depth_data[ind,1]
    depth['h'] = depth_data[ind,2]

    return depth

def interpolate_to_mesh(depth, M,  min_depth = 5.0):
    """
    Interpolate data from the unstructured data array (data) to the unstructured mesh (M)
    """
    print('- Prepare depth data for interpolation')
    point     = np.array([depth['x'], depth['y']]).T
    interpolant = LinearNDInterpolator(point, depth['h'])

    print('- Interpolate topography to nodes')
    h_raw       = interpolant(M['x'], M['y'])

    print(f'- Force the depth to be greater than {min_depth} m')
    i           = np.where(h_raw[:] < min_depth)[0]
    h_raw[i]    = min_depth

    return h_raw

def smooth_topo(M, rx0max = 0.2, SmoothFactor = 0.2, min_depth = 5.0):
    """
    Smooth the topography where the smoothness is bad
    """
    # Initial pass to do the initial smoothing (reduce dataset noise)
    print('- Reduce noise')
    print('-- Laplacian filter')
    M['h'] = laplacian_filter(M, SmoothFactor = SmoothFactor)

    print('- Adjust slope')
    print('-- Mellor, Ezer and Oey scheme')
    i = 0
    while True:
        i += 1
        M['h'], rx0_max, corrected = mellor_ezer_oey(M['h'], M['ntsn'], M['nbsn'], nodes = None, rx0max = rx0max-0.02)
        print(str(i)+': Max rx0: ' + str(np.round(rx0_max,3)) + \
              ' - Number of adjustments: '+str(corrected))
        if abs(rx0_max - rx0max) < rx0max*0.01: # hard to read syntax :)
            print('- Bathymetry smoothed.')
            break
        elif rx0_max == 0:
            print('- Bathymetry smoothed.')
            break

    return M['h']

def get_rx0(M, nodes = None):
    """
    Get the rx0 number
    """
    if nodes is None:
        nodes = np.arange(len(M['x']))

    edge = []
    dh   = []
    ph   = []
    for i in nodes:
        diff = M['h'][i]-M['h'][M['nbsn'][i,:M['ntsn'][i]]]
        dh.append(np.abs(np.max(diff)))
        ind = np.where(np.abs(diff) == dh[-1])[0][0]
        ph.append(M['h'][i] + M['h'][M['nbsn'][i,ind]])
        edge.append([i,M['nbsn'][i,ind]])

    rx0  = np.array(dh)/np.array(ph)

    return rx0, np.array(dh), np.array(ph), np.array(edge)

@njit
def get_rx1_direct(nbsn, ntsn, siglevz, sig):
    '''
    Computes the rx1 number directly from the definition
    - Not in an abnormally efficient way though, better make use of those edges at some point
    '''
    rx1 = np.zeros(ntsn.shape)
    for n, (nb, nt) in enumerate(zip(nbsn, ntsn)):
        print(n)
        nodes_surrounding_at_n = nb[:nt]
        depths_around_here     = siglevz[nodes_surrounding_at_n, :]
        depths_here            = siglevz[n,:]
        abs_dh = np.zeros((depths_around_here.shape[0], sig))
        for i in range(sig-1):
            abs_dh = np.abs(depths_here[i] - depths_around_here[:,i] + depths_here[i] - depths_around_here[:,i])
            dh_v   = depths_here[i] + depths_around_here[:,i] - (depths_here[i+1] + depths_around_here[:,i+1])
            rx1[n] = np.max(abs_dh / dh_v)
    return rx1

def get_rx1(M, nodes = None, rx1max = None):
    '''
    Estimate the rx1 number

    Parameters
     -----------
    M:      FVCOM mesh, connectivity data
    rx1max: Maximum acceptable rx1 value

    Output:
    ----------
    Estimate of the pressure-gradient-error estimate rx2, and nodes that need smoothing
    '''
    if nodes is None:
        nodes = np.arange(len(M['h']))

    # Get rx0:
    rx0, dh, ph, e  = get_rx0(M, nodes = nodes)

    # Get rx1:
    print('-- Estimate rx1')
    upper = (M['siglev'][-1]+M['siglev'][-2])*dh
    lower = (M['siglev'][-1]-M['siglev'][-2])*ph
    #rx1   = rx0*M['rx0_to_rx1'] #(upper/lower)
    rx1    = np.abs(upper)/np.abs(lower)
    bad_nodes = np.where(rx1>rx1max)[0]
    return rx1, nodes[bad_nodes]

def read_sigma(sigmafile=None):
    '''
    Generate a tanh sigma coordinate distribution

    Parameters:
    ----------
    sigmafile:   A casename_sigma.dat file containing the tanh parameters

    Out:
    ----------
    lev, lay:    Sigma coordinate lev and lay (valid for the entire domain)
    '''
    if sigmafile is None:
        raise ValueError('You must provide a file determining the sigma layer distribution!\n')

    # Import data from a .tex document
    data = np.loadtxt(sigmafile,delimiter = '=', dtype = str)

    # Read the input parameters from the casename_sigma.dat file
    if data[1,1] == ' TANH ':
        nlev = int(data[0,1])
        du   = float(data[2,1])
        dl   = float(data[3,1])
        lev  = np.zeros((nlev))

        for k in np.arange(nlev-1):
            x1 = dl + du
            x1 = x1 * (nlev - 2 - k) / (nlev - 1)
            x1 = x1 - dl
            x1 = np.tanh(x1)
            x2 = np.tanh(dl)
            x3 = x2 + np.tanh(du)
            lev[k+1] = (x1 + x2) / x3 - 1.0
    elif data[1,1] == ' UNIFORM' or data[1,1] == 'UNIFORM':
        nlev = int(data[0,1])
        lev = np.zeros(nlev)
        for k in np.arange(1,nlev+1):
            lev[k -1] = -((k - 1)/(nlev - 1))

    elif data[1,1] == ' GEOMETRIC':
        nlev = int(data[0,1])
        lev = np.zeros(nlev)
        p_sigma = np.double(data[2,1])
        for k in range(1,np.int(np.floor((nlev+1)/2)+1)):
            lev[k-1]=-((k-1)/((nlev+1)/2 - 1))**p_sigma/2
        for k in range(np.int(np.floor((nlev+1)/2))+1,nlev+1):
            lev[k-1]=((nlev-k)/((nlev+1)/2-1))**p_sigma/2-1

    else:
        raise ValueError(f'BuildCase supports tanh-, geometric- and uniform-coordinates at the moment. {data[1,1]} is invalid.')

    # Siglay
    lay = [(lev[k]+lev[k+1])/2 for k in range(len(lev)-1)]

    return lev, np.array(lay)

# ======================================================================================================
#                                      Smoothing procedures
# ======================================================================================================
def laplacian_filter(M, bad_nodes = None, SmoothFactor = None):
    """
    A simple lapacian filter to smooth the topography

    Parameters:
    -----------
    M:            Mesh dict
    SmoothFactor: The degree of smoothing
    """
    if bad_nodes is None:
        bad_nodes = np.arange(len(M['x']))

    h_smooth = np.copy(M['h'])
    for node in bad_nodes:
        nodes  = M['nbsn'][node,:M['ntsn'][node]]
        smooth = np.mean(M['h'][nodes])
        h_smooth[node] = (1-SmoothFactor)*h_smooth[node] + SmoothFactor*smooth
    return h_smooth

def martinho_batten(M, rx0max = 0.2):
    """
    Lifts the bathymetry if it is too steep. Not implemented yet...

    Parameters:
    -----------
    M: Mesh dict

    Returns:
    -----------
    Smoothed bathymetry

    ===================================================================================
    Batteen, Mary L., et al.
    A process-oriented modelling study of the coastal Canary and Iberian Current system
    Ocean Modelling, Volume 18 (2007)
    https://doi.org/10.1016/j.ocemod.2007.02.006
    ===================================================================================
    """
    bath = np.copy(M['h'])
    rmax = 0
    corrected = 0
    for edge in M['edge']:
        r = (bath[edge[0]]-bath[edge[1]])/(bath[edge[0]]+bath[edge[1]])
        rmax = max(r,rmax)
        if r>rx0max:
            bath[edge[1]] = ((1-r)/(1+r))*bath[edge[0]]
            corrected +=1
            continue

    return bath, rmax, corrected

@jit(nopython = True)
def mellor_ezer_oey(raw_bath, ntsn, nbsn, rx0max = 0.2, nodes = None):
    """
    Parameters:
    -----------
    bathymetry, number of surrounding nodes, index of surrounding nodes, smoothing goal and
    what nodes to inspect

    Returns:
    -----------
    Smoothed bathymetry

    ===================================================================================
    Mellor, Ezer and Oey:
    The Pressure Gradient Conundrum of Sigma Coordinate Ocean Models
    J. Atmos. Oceanic Technol. (1994)
    https://doi.org/10.1175/1520-0426(1994)011%3C1126:TPGCOS%3E2.0.CO;2
    ===================================================================================

    Future:
    - Add support for over-relaxation to improve convergence.
    """

    # initialize
    if nodes is None:
        nodes = np.arange(len(raw_bath))

    # Iterate the mesh
    rmax = 0
    bath      = np.copy(raw_bath)
    corrected = 0
    for i in nodes:
        diff = bath[i]-bath[nbsn[i,:ntsn[i]]]
        nbsn_ind  = np.where(diff == np.max(diff))[0][0]
        ind  = nbsn[i,nbsn_ind]
        difference = diff[nbsn_ind]
        if difference < 0:
            continue

        ph = bath[i] + bath[ind]

        r  = diff[nbsn_ind]/ph
        if r > rx0max:
            corrected += 1
            rarray     = np.array((r,rmax))
            rmax       = np.max(rarray)
            delta      = 0.5*(bath[i]-bath[ind]-rx0max*(bath[i]+bath[ind]))
            bath[i]   -= delta
            bath[ind] += delta

    return bath, rmax, corrected

# ======================================================================================================
#                                      Triangulation stuff
# ======================================================================================================
def triangulate(M,var):
    c = (M[var][M['nv'][:,0]] + M[var][M['nv'][:,1]] + M[var][M['nv'][:,2]])/3.0
    return c

def setup_metrics(M):
    """
    Setup metrics for secondary connectivity (nodes surrounding nodes)
    """
    # Find triangle edges
    tri    = Triangulation(M['x'], M['y'], M['nv'])
    edges  = tri.edges

    # Connect nodes around each triangle
    # ----------------------------------------------------------------
    # Allocate storage
    ntsn = np.zeros((len(M['x']))).astype(int)
    nbsn = -1*np.ones((len(M['x']),12)).astype(int)

    # This is a slow one, nice to keep track
    widget = ['  - Identify nodes surrounding nodes: ', pb.Percentage(), ' ', pb.Bar()]
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

# =====================================================================================================
#                                       Estimate stability criteria
# =====================================================================================================
def get_cfl(M, u = 3.0, zeta = 1.0):
    '''
    Estimate the CDF determined maximum timestep
    '''
    g  = 9.81
    x = np.array([M['x'][M['nv'][:,0]], M['x'][M['nv'][:,1]], M['x'][M['nv'][:,2]]])
    y = np.array([M['y'][M['nv'][:,0]], M['y'][M['nv'][:,1]], M['y'][M['nv'][:,2]]])

    # Store each side as vectors
    AB    = np.array([x[0,:]-x[1,:], y[0,:]-y[1,:]])
    BC    = np.array([x[1,:]-x[2,:], y[1,:]-y[2,:]])
    CA    = np.array([x[2,:]-x[0,:], y[2,:]-y[0,:]])

    # Length of each triangle side
    lAB   = np.sqrt(AB[0,:]**2+AB[1,:]**2)
    lBC   = np.sqrt(BC[0,:]**2+BC[1,:]**2)
    lCA   = np.sqrt(CA[0,:]**2+CA[1,:]**2)

    # Timestep for each wall
    dpt      = np.max(M['h'][M['nv']], axis = 1)+zeta
    cg_speed = np.sqrt(dpt*g)+u
    ts_AB    = np.array(lAB/cg_speed)
    ts_BC    = np.array(lBC/cg_speed)
    ts_CA    = np.array(lCA/cg_speed)
    ts_walls = np.array([ts_AB, ts_BC, ts_CA])
    ts_min   = np.min(ts_walls, axis = 0)/np.sqrt(2) # to adjust for transverse wave propagation

    # Display it! :)
    print('Required timestep: '+str(min(ts_min))+' s')
    plt.figure()
    plt.tripcolor(M['x'], M['y'], M['nv'], ts_min, cmap = 'jet')
    plt.colorbar()
    plt.clim([min(ts_min), 6*min(ts_min)])
    plt.axis('equal')
    plt.show(block = False)

    return ts_min

# =====================================================================================================
#                                       Write inputfiles
# =====================================================================================================
def write_bath(M, casename):
    '''
    Write a FVCOM readable batymetry file
    -----------
    - Generates an ascii FVCOM 4.x format bathymetry from a Mesh dict

    Parameters
    -----------
    - M:         Any dictionary containing lat, lon, x, y and h variables.
    - casename:  Casename of the given experiment
    '''
    filename = casename + '_dep.dat'

    f = open('input/'+filename, 'w')
    f.write('Node Number = ' + str(len(M['x']))+'\n')

    for x, y, h in zip(M['x'], M['y'], M['h']):
        line = '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(h)+'\n'
        f.write(line)
    f.close()

    print('- Wrote : '+filename)

def write_grd(M, casename):
    '''
    Write a FVCOM readable grid file
    -----------
    - Generates an ascii FVCOM 4.x format grid file from a Mesh dict

    Parameters
    -----------
    - M:         Any dictionary containing x, y  and tri.
    - casename:  Casename of the given experiment
    '''
    filename = casename + '_grd.dat'

    f = open('input/'+filename, 'w')
    f.write('Node Number = ' + str(len(M['x']))+'\n')
    f.write('Cell Number = ' + str(len(M['xc']))+'\n')

    # Write triangulation
    # ----
    i = 0
    for t1, t2, t3 in M['nv']:
        i += 1
        f.write(str(i)+' '+str(t1+1) + ' ' + str(t2+1) + ' ' + str(t3+1)+ ' ' + str(i) + '\n')

    # Write nodes
    # ----

    for i, (x, y) in enumerate(zip(M['x'], M['y'])):
        line = str(i+1) +' '+ '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(0.0)+'\n'
        f.write(line)
    f.close()
    print('- Wrote : '+filename)

def write_sponge(M, casename):
    '''
    Write a FVCOM readable grid file
    -----------
    - Generates an ascii FVCOM 4.x format grid file from a Mesh dict

    Parameters
    -----------
    - M:         Any dictionary containing lat, lon, x, y and h variables.
    - casename:  Casename of the given experiment
    '''
    filename = casename + '_spg.dat'
    f = open('input/'+filename, 'w')
    f.write('Sponge Node Number = ' + str(len(M['sponges']))+'\n')

    # Write the sponge nodes and properties
    # ----
    for node in M['sponges']:
        line = str(node+1) + ' ' + '{0:.6f}'.format(M['sponge_radius']) + ' ' + '{0:.6f}'.format(M['sponge_fact'])+'\n'
        f.write(line)
    f.close()

    print('- Wrote : '+filename)

def write_obc(M, casename):
    '''
    Write a FVCOM readable grid file
    -----------
    - Generates an ascii FVCOM 4.x format grid file from a Mesh dict

    Parameters
    -----------
    - M:         Any dictionary containing lat, lon, x, y and h variables.
    - casename:  Casename of the given experiment
    '''
    filename = casename + '_obc.dat'
    obc_nodes = M['read_obc_nodes']

    f = open('input/'+filename, 'w')
    f.write('OBC Node Number = ' + str(M['num_obc'])+'\n')

    # Write obc nodes
    # ----
    i = 0
    for j in range(M['read_obc_nodes'].shape[1]):
        for obc_node in obc_nodes[0,j][0]:
            i += 1
            f.write(str(i) + ' ' + str(obc_node+1) + ' ' + str(1)+'\n')
    f.close()
    print('- Wrote : '+filename)

def write_cor(M, casename):
    '''
    Write a FVCOM readable grid file
    -----------
    - Generates an ascii FVCOM 4.x format grid file from a Mesh dict

    Parameters
    -----------
    - M:         Any dictionary containing lat, lon, x, y and h variables.
    - casename:  Casename of the given experiment
    '''
    filename = casename + '_cor.dat'

    f = open('input/'+filename, 'w')
    f.write('Node Number = ' + str(len(M['x']))+'\n')

    # Write latitude to nodes
    # ----
    for x, y, lat in zip(M['x'], M['y'], M['lat']):
        line = '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(lat)+'\n'
        f.write(line)
    f.close()
    print('- Wrote : '+filename)

class RemapError(Exception):
    pass
