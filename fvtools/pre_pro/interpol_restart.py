"""
Will only work for python versions >= 3.5
"""
import netCDF4
import sys
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time
import fvtools.nesting.vertical_interpolation as vi
import seawater as sw
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.tools import Filelist
from scipy.spatial import cKDTree
from netCDF4 import Dataset

global versionnr
versionnr = 1.2

# versionnr 1.0: Adds support for different #sigmalayers in the mother and child
# versionnr 1.1: Writes full field to the restart file
# versionnt 1.2: Crops the full mother grid to fit the smaller one (to speed it up significantly, smaller KDTrees)

def main(childfn       = None,
         result_folder = None,
         name          = None,
         motherfn      = None,
         filelist      = None,
         mother_grid   = None,
         child_grid    = None,
         speed         = False):
    '''
    Interpolate all restart fields from one- to another FVCOM model.

    Replace temperature, salinity and velocity data in the restart file
    with data from another model covering that domain.

    We strongly recommend using a casename_restart_xxxx.nc file when interpolating, as this
    initializes the model with all the fields it needs. Normal output files should just be used
    if restart files are not available!

    Mandatory:
    ---
    childfn        - path to child restart
    
    Three methods to specify the restart file:
    ---
    motherfn       - path to mother source file (given explicit)
    filelist       - filelist make by fvcom_make_filelist.py
    results_folder - folders where the mother grid is located (textfile with paths to the folders)
    --> name       - Also provide the name of the numerical experiment (ie. 'Tr6'). This will force 
                     Python to look for the mother file 

    Optional:
    ---
    mother_grid    - path to mother grid file (if mother grid info is not in the source file)
    child_grid     - path to child grid file (if child grid info is not in the restart file)
    speed          - sometimes works, many times not =) (False by default)
    '''
    if childfn is None:
        raise InputError('You must specify the name of the restartfile.')

    # Open the child data
    # -------------------
    child     = Dataset(childfn,'r+', format='NETCDF4')
    if len(child['time'][:]) > 1:
        raise InputError(f'The time dimension in {childfn} is longer than 1, this can not be a restartfile.')

    # Find mother results
    # -------------------
    if result_folder is not None:
        motherfn  = find_mother(name, result_folder, child)

    if filelist is not None:
        fl = Filelist(filelist)
        try:
            ind = np.where(child['time'][:] == fl.time)[0][0]
        except:
            raise InputError('The filelist has no time corresponding to the restart time.')
        motherfn = fl.path[ind]

    # Load mother & child grid information
    # -------------------
    print('Load grid files:\n-------------')
    
    M, M_ch   = get_mobj(motherfn, childfn, \
                         mother_grid, child_grid)
    print(f'Child grid:  {childfn}')
    print(f'Mother grid: {motherfn}\n')

    # Crop mother grid
    # -------------------
    M.subgrid([np.min(M_ch.x)-10000, np.max(M_ch.x)+10000], [np.min(M_ch.y)-10000, np.max(M_ch.y)+10000], full = True)

    # Load output
    # -------------------
    print('Searching for correct time index, prepare grid metrics')
    print('-------------')
    mother    = Dataset(motherfn,'r', format='NETCDF4')
    ind       = check_time(mother['time'],child['time'])

    # Define fields we want in the restart file
    # Fields
    # -----
    if speed:
        nodefield = ['zeta', 'salinity', 'temp', 'viscofh', 'km',
                     'kh', 'kq','q2', 'q2l', 'l', 'omega', 'et',
                     'tmean1', 'smean1', 'rho1', 'rmean1', 'zice']
        cellfield = ['u','v','ua','va','ww','tauc','viscofm']

    else:
        nodefield = ['zeta', 'salinity', 'temp','et',
                     'tmean1', 'smean1', 'rho1', 'rmean1', 'zice']
        cellfield = None

    # These fields are not stored in the normal output files, but will be available in restart files. 
    # -----
    alias     = {}
    alias['tmean1'] = 'temp'
    alias['smean1'] = 'salinity'
    alias['et']     = 'zeta'

    # -----------------------------------------------------------------------------------------------
    #                                  Interpolation routines
    # -----------------------------------------------------------------------------------------------
    print('\nInterpolate data')
    print('-------------')

    # Horizontal
    print('- Horizontal interpolation (nearest neighbor):')
    data, dpt = nearest_neighbor(mother, M, M_ch, ind,
                                  nodefield, cellfield, alias)

    # Vertical
    print('- Vertical interpolation') 
    data = vertical_interpolation(data, child, dpt) 

    # Dump to netCDF
    child = dump_data(data, child)

    # Done!
    child.mother           = motherfn
    child.data_age         = f'data dumped to this restart file by interpol_restart.py {time.ctime(time.time())}'
    child.interpol_version = versionnr
    child.interp_folder    = os.getcwd()
    mother.close()
    child.close()

    # Show results (now that the data are safely stored :])
    child  = Dataset(childfn)
    mother = Dataset(motherfn)
    plot_interps(mother, child, M, ind)
    print('Fin.')


# ---------------------------------------------------------------------------------------------------------------------

# Load grid
# ----
def get_mobj(mother_fn,child_fn,mother_grid,child_grid):
    '''
    load the Mesh object from a .nc- or .mat-file
    '''
    if mother_grid is not None:
        M         = FVCOM_grid(mother_grid, verbose = False)
    else:
        M         = FVCOM_grid(mother_fn, verbose = False)

    if child_grid is not None:
        M_ch      = FVCOM_grid(child_grid, verbose = False)
    else:
        M_ch      = FVCOM_grid(child_fn, verbose = False)
    return M, M_ch

def check_time(tm,tch):
    '''
    make sure that we interpolate data from the correct timestep
    '''
    try:
        ind     = np.argwhere(tm[:]==tch[0])[0][0]
        print('- The restart file starts: '+\
              netCDF4.num2date(tch[0],tch.units).strftime('%d. %b %Y at %H:%M')+" o'clock")

    except:
        raise InputError(f'The mother file does not include: {netCDF4.num2date(tch[0],tch.units).strftime("%d. %b %Y - %H:%M")}\n'+\
                         f'This mother file starts: {netCDF4.num2date(tm[0],tm.units).strftime("%d. %b %Y - %H:%M")}\n'+\
                         f'and ends: {netCDF4.num2date(tm[-1],tm.units).strftime("%d. %b %Y - %H:%M")}')
    return ind

def find_nearest(M,M_ch):
    '''
    Find the nearest node in the mother grid using a scipy KDTree
    '''
    # Reference grids as point arrays
    # ----
    node_ch   = np.array([M_ch.x,  M_ch.y]).transpose()
    cell_ch   = np.array([M_ch.xc, M_ch.yc]).transpose()

    node_mo   = np.array([M.x,  M.y]).transpose()
    cell_mo   = np.array([M.xc, M.yc]).transpose()

    # Set up KDTrees, find index of nearest mother
    # ----
    print('  - Finding nearest node in mother')
    tree        = cKDTree(node_mo)
    p,inds_node = tree.query(node_ch)

    print('  - Finding nearest cell in mother')
    tree        = cKDTree(cell_mo)
    p,inds_cell = tree.query(cell_ch)

    return inds_node.astype(int), inds_cell.astype(int)


def find_file(name, data_directories, time):
    ''' 
    Make lists that link a point in time to fvcom result file 
    and index (in corresponding file). Three lists a returned:
    1: list with point in time (fvcom time: days since 1858-11-17 00:00:00)
    2: list with path to files
    3: list with indices
    '''
    # Go through data directories and identify relevant data files
    for directory in data_directories:
        print(f'\n{directory}')
       
        files = [elem for elem in os.listdir(directory) if os.path.isfile(os.path.join(directory,elem))]
        files = [elem for elem in files if ((name in elem) and (len(elem) == len(name) + 8))]
        
        files.sort()
         
        for file in files:
            nc     = Dataset(os.path.join(directory, file), 'r')
            t      = nc.variables['time'][:]
            if time in t:
                print(f'\n--> Found the time in: {file}\n')
                path = os.path.join(directory, file)
                return path
            print(f'- Not in {file}')
    return

def find_mother(name, result_folder, child):
    if result_folder is not None:
        # Read the name of the experiment
        if name is None:
            raise InputError('You need to provide the name of the experiment!')

        # Read the names of the result folders (should be written in "with f as open(folder)" kanskje?)
        file = open(result_folder, 'r')
        results = []
        for line in file:
            if len(line) > 1:
                results.append(line.rstrip('\n'))
            else:
                pass

        # Get the filename
        motherfn = find_file(name, results, child['time'][0])

    else:
        raise InputError('You must provide a file, a result folder or a filelist.')
    file.close() # I think so?
    return motherfn

# ------------------------------------------------------------------------------------
#                           Interpolation schemes
# ------------------------------------------------------------------------------------
def nearest_neighbor(mother, M, M_ch, ind, nodefield, cellfield, alias):
    '''
    Interpolate from mother grid using the nearest neighbor interpolation scheme
    '''

    # Find the nearest node/cell in the mother grid
    # ---------------------------
    nearest_mother_node, nearest_mother_cell = find_nearest(M,M_ch)
    horizontal = {}

    # Loop over nodes in child
    # ---------------------------
    print('- node data')
    for varname in nodefield:
        try:
            if len(mother.variables[varname].shape) == 2:
                tmp  = mother[varname][:][ind, M.cropped_nodes[nearest_mother_node]]
                horizontal[varname] = tmp
                
            elif len(mother.variables[varname].shape) == 3:
                tmp = mother[varname][:][ind,:, M.cropped_nodes[nearest_mother_node]]
                horizontal[varname] = tmp.transpose()
            print(' -> '+ varname)
        except:
            if varname in alias.keys():
                tmp  = horizontal[alias[varname]]
                horizontal[varname] = tmp
                print(' -> '+ varname)
                
            elif varname in ['rho1', 'rmean1']:
                rho = sw.dens0(horizontal['salinity'], horizontal['temp'])
                horizontal[varname] = rho
                print(' -> '+ varname)

            elif varname == 'zice':
                horizontal[varname] = np.zeros(horizontal['zeta'].shape) # zeta for å få rett dimensjon
                print(' -> '+ varname)

            else:
                print(f'{varname} could not be interpolated to restart')
            
    # Loop over elements in child
    # ---------------------------
    if cellfield is not None:
        print('- cell data')
        for varname in cellfield:
            print(' -> ' + varname)
            try:
                if len(mother.variables[varname].shape) == 2:
                    # 2D-fields
                    tmp = mother[varname][:][ind, M.cropped_cells[nearest_mother_cell]]
                    horizontal[varname] = tmp
                
                elif len(mother.variables[varname].shape) == 3:
                    # 3D-fields
                    tmp = mother[varname][:][ind,:, M.cropped_cells[nearest_mother_cell]]
                    horizontal[varname] = tmp.transpose()

            except:
                print(f'{varname} could not be interpolated to restart')
                
    # Store info needed for vertical interpolation
    # ----
    class grid_info: pass
    hc                = (M.h[M.tri[:,0]]+M.h[M.tri[:,1]]+M.h[M.tri[:,0]])/3
    grid_info.h       = M.h[nearest_mother_node]
    grid_info.siglay  = M.siglay[0,:]
    grid_info.hc      = np.sum(M.h[M.tri[nearest_mother_cell,:]],axis=1)/3
    grid_info.siglev  = M.siglev[:,0]

    return horizontal, grid_info


def vertical_interpolation(data, child, dpt):
    '''
    Linear vertical interpolation of ROMS data to FVCOM-depths.
    '''
    var = [*data]
    vertical_data = {}

    # Get the depths to interpolate to and from
    # ----
    node_dpt_mother = dpt.h[:]*dpt.siglay
    cell_dpt_mother = dpt.hc[:]*dpt.siglay
    node_dpt_child  = child['h'][:][:,None]*child['siglay'][:,0]
    cell_dpt_child  = child['h_center'][:][:,None]*child['siglay_center'][:,0]
    
    # Calculate the interpolation coefficients and get data indices
    # ----
    print('- Calculate vertical node weights')
    n_ind1, n_ind2, n_weigths1, n_weigths2 = vi.calc_interp_matrices(-node_dpt_mother.T, -node_dpt_child.T)

    print('- Calculate vertical cell weights')
    c_ind1, c_ind2, c_weigths1, c_weigths2 = vi.calc_interp_matrices(-cell_dpt_mother.T, -cell_dpt_child.T)

    print('- Interpolate vertical data')
    # Do the interpolation
    # ----
    for field in var:
        if len(data[field].shape) == 1:
            vertical_data[field] = data[field]
            continue

        if len(data[field][0,:]) == len(child['h'][:]):
            vertical_data[field] = data[field][n_ind1, range(0, data[field].shape[1])] * n_weigths1 + \
                                   data[field][n_ind2, range(0, data[field].shape[1])] * n_weigths2 
        else:
            vertical_data[field]= data[field][c_ind1, range(0, data[field].shape[1])] * c_weigths1 + \
                                  data[field][c_ind2, range(0, data[field].shape[1])] * c_weigths2 

    return vertical_data

def dump_data(data, child):
    var = [*data]
    for field in var:
        try:
            child[field][0,:] = data[field]
        except:
            print(f'- {field} could not be dumped to child')
    return child

# -----------------------------------------------------------------------------------------
#                                  Visualization
# -----------------------------------------------------------------------------------------
def plot_interps(mother, child, M, ind):
    # Visualize:
    # -----

    # Get triangulation
    triangles_ch = child.variables['nv'][:].transpose() - 1

    # SSE
    # ----------------------------------------------------------
    levels  = np.linspace(child.variables['zeta'][0,:].min(), child.variables['zeta'][0,:].max(), 30)
    fig, ax = plt.subplots(1,2,figsize = (20,10))
    tp      = ax[0].tricontourf(child['x'][:], child['y'][:], triangles_ch, child.variables['zeta'][0,:], levels = levels, extend = 'both')
    ax[0].set_title('child, sea surface elevation')
    ax[0].set_aspect('equal')

    tp      = ax[1].tricontourf(M.x, M.y, M.tri, mother.variables['zeta'][ind, M.cropped_nodes], levels = levels, extend = 'both')
    ax[1].set_title('mother, sea surface elevation')
    ax[1].set_xlim(child['x'][:].min(), child['x'][:].max())
    ax[1].set_ylim(child['y'][:].min(), child['y'][:].max())
    ax[1].set_aspect('equal')

    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.675])
    cb = fig.colorbar(tp, cax = cbar_ax, )
    cb.set_label('surface elevation [m]')

    # Temperature
    # ----------------------------------------------------------
    fig, ax = plt.subplots(1,2,figsize = (20,10))
    levels  = np.linspace(child.variables['temp'][0,-1,:].min(), child.variables['temp'][0,0,:].max(), 30)
    tp      = ax[0].tricontourf(child['x'][:], child['y'][:], triangles_ch, child.variables['temp'][0,-1,:], levels = levels, extend = 'both')
    ax[0].set_title('child, temperature (bottom)')
    ax[0].set_aspect('equal')

    tp      = ax[1].tricontourf(M.x, M.y, M.tri, mother.variables['temp'][ind, -1, M.cropped_nodes], levels = levels, extend = 'both')
    ax[1].set_title('mother, temperature (bottom)')
    ax[1].set_xlim(child['x'][:].min(), child['x'][:].max())
    ax[1].set_ylim(child['y'][:].min(), child['y'][:].max())
    ax[1].set_aspect('equal')

    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.675])
    cb = fig.colorbar(tp, cax = cbar_ax, )
    cb.set_label('degrees [*C]')


    # Speed
    # ----------------------------------------------------------
    ch_sp  = np.sqrt(child.variables['ua'][0,:]**2+child.variables['va'][0,:]**2)*100
    mo_sp  = np.sqrt(mother.variables['ua'][ind, M.cropped_cells]**2+mother.variables['va'][ind, M.cropped_cells]**2)*100
    levels = np.linspace(0, ch_sp.max(), 30)

    fig, ax      = plt.subplots(1,2,figsize = (20,10))
    tp = ax[0].tripcolor(child['x'][:], child['y'][:], triangles_ch, ch_sp, vmin = np.min(levels), vmax = np.max(levels))
    ax[0].set_title('child, speed (depth average)')
    ax[0].set_aspect('equal')

    tp = ax[1].tripcolor(M.x, M.y, M.tri, mo_sp, vmin = np.min(levels), vmax = np.max(levels))
    ax[1].set_title('mother, speed (depth average)')
    ax[1].set_xlim(child['x'][:].min(), child['x'][:].max())
    ax[1].set_ylim(child['y'][:].min(), child['y'][:].max())
    ax[1].set_aspect('equal')

    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.675])
    cb = fig.colorbar(tp, cax = cbar_ax)
    cb.set_label('speed [cm/s]')

    # Done!
    # ------------------------------------------------------------------------------------------------
    plt.show()

class InputError(Exception): pass