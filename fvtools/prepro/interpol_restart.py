"""
Will only work for python versions >= 3.5
"""
import netCDF4
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import fvtools.nesting.vertical_interpolation as vi # This script will be moved to another folder in due time
import seawater as sw
import cmocean as cmo
import datetime

from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.tools import Filelist

from .initial.empty_initial import make_initial_file
from .initial.plot_restart import plot_interps


global versionnr
versionnr = 1.4

# versionnr 1.0: Adds support for different #sigmalayers in the mother and child
# versionnr 1.1: Writes full field to the restart file
# versionnr 1.2: Crops the full mother grid to fit the smaller one (to speed it up significantly, smaller KDTrees)
# versionnr 1.3: basically just string fixes, adding metadata to the restart file
# versionnr 1.4: Make an empty restart file, optionally use this instead of childfn -- seems like we still have some field to fill in....


def main(
        child_fn      = None,
        startdate     = None,
        child_grid    = None,
        result_folder = None,
        name          = None,
        filelist      = None,
        speed         = False
    ):
    '''
    Two options for child specification:
    child_fn:      - path to an existing FVCOM restart file
    startdate      - date ("yyyy-mm-dd-hh")
                     - require that you specify child_grid (anything that can be passed to FVCOM_grid)
                     - note: this feature is still not properly tested...
    
    
    Two methods to specify the restart file:
    filelist       - filelist made using fvcom_make_filelist.py
    results_folder - folders where the mother grid is located (textfile with paths to the folders)
        --> name   - Then you must also provide the name of the numerical experiment (ie. 'PO12')

    Optional:
    speed          - Initialize the model with velocities as well. Sometimes works, many times not - hence False by default.
    '''
    # Load (or create) child restart file
    child_fn = get_child_file(child_fn, startdate, child_grid)
    nodefield, cellfield, alias = interpolation_fields(speed)

    with netCDF4.Dataset(child_fn,'r+', format='NETCDF4') as child:
        mother_fn = get_mother_file(result_folder, filelist, name, child)

        print('Load grid files:')
        M, M_ch = FVCOM_grid(mother_fn), FVCOM_grid(child_fn)
        print(f'  Child:  {M_ch.casename}')
        print(f'  Mother: {M.casename}\n')

        # Crop mother grid to just cover the child grid
        M = M.subgrid([np.min(M_ch.x)-10000, np.max(M_ch.x)+10000], 
                      [np.min(M_ch.y)-10000, np.max(M_ch.y)+10000])

        print('Searching for correct time index, prepare grid metrics')
        with netCDF4.Dataset(mother_fn,'r', format='NETCDF4') as mother:
            ind = check_time(mother['time'],child['time'])

            print('\nInterpolate data')
            print('- Horizontal interpolation (nearest neighbor):')
            data, dpt = nearest_neighbor(mother, M, M_ch, ind, nodefield, cellfield, alias)

        # Update depth with zeta for mother and child prior to vertical interpolation
        M.zeta    = M.load_netCDF(mother_fn, 'zeta', ind)
        M_ch.zeta = data['zeta']

        print('\n- Vertical interpolation') 
        data = vertical_interpolation(data, child, dpt) 

        # Dump to netCDF
        child = dump_data(data, child)

        # Done!
        child.mother           = mother_fn
        child.data_age         = f'Data dumped to this restart file by interpol_restart.py ({time.ctime(time.time())})'
        child.interpol_version = versionnr
        child.interp_folder    = os.getcwd()

    # Show results (now that the data are safely stored :])
    plot_interps(mother_fn, child_fn, M, M_ch, ind)
    print('Fin.')


#                                                      Routines
# --------------------------------------------------------------------------------------------------------------------------

def get_child_file(childfn, startdate, child_grid):
    assert childfn is not None or startdate is not None, 'You must specify the name of the restartfile or a restart date.'
    if startdate is not None:
        childfn = make_initial_file(FVCOM_grid(child_grid), startdate, 1)
    return childfn

def get_mother_file(result_folder, filelist, name, child):
    if result_folder is not None:
        return find_mother(name, result_folder, child)

    if filelist is not None:
        fl = Filelist(filelist)
        try:
            ind = np.where(child['time'][:] == fl.time)[0][0]
        except:
            raise InputError('The filelist has no time corresponding to the restart time.')
        return fl.path[ind]

def interpolation_fields(speed):
    # Define fields we want in the restart file
    if speed:
        nodefield = ['zeta', 'salinity', 'temp', 'viscofh', 'km', 'kh', 'kq','q2', 'q2l', 'l', 'omega', 'et',
                     'tmean1', 'smean1', 'rho1', 'rmean1', 'zice']
        cellfield = ['u','v','ua','va','ww','tauc','viscofm']
    else:
        nodefield = ['zeta', 'salinity', 'temp','et', 'tmean1', 'smean1', 'rho1', 'rmean1', 'zice']
        cellfield = None

    # Define aliases
    alias     = {'tmean1': 'temp',
                 'smean1': 'salinity',
                 'et': 'zeta'}
    return nodefield, cellfield, alias

def check_time(tm,tch):
    '''
    make sure that we interpolate data from the correct timestep
    '''
    try:
        ind = np.argwhere(tm[:]==tch[0])[0][0]
        print('- The restart file starts: ' + netCDF4.num2date(tch[0],tch.units).strftime('%d. %b %Y at %H:%M')+" o'clock")

    except:
        raise InputError(f'{netCDF4.num2date(tch[0],tch.units).strftime("%d. %b %Y - %H:%M")} is not available in the mother model.\n'+\
                         f'- This mother file starts: {netCDF4.num2date(tm[0],tm.units).strftime("%d. %b %Y - %H:%M")}\n'+\
                         f'                 and ends: {netCDF4.num2date(tm[-1],tm.units).strftime("%d. %b %Y - %H:%M")}')
    return ind

def find_file(name, data_directories, time):
    ''' 
    Make lists that link a point in time to fvcom result file and index (in corresponding file). Three lists a returned:
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
            with netCDF4.Dataset(os.path.join(directory, file), 'r') as nc:
                t  = nc.variables['time'][:]
                if time in t:
                    print(f'\n--> Found the time in: {file}\n')
                    path = os.path.join(directory, file)
                    return path
                print(f'- Not in {file}')
    raise InputError('Could not find any files in you search period')

def find_mother(name, result_folder, child):
    assert result_folder is not None, 'You must provide a file, a result folder or a filelist.'
    assert name is not None, 'You need to provide the name of the experiment!'

    # Read the names of the result folders
    with open(result_folder, 'r') as file:
        results = []
        for line in file:
            if len(line) > 1:
                results.append(line.rstrip('\n'))
            else:
                pass
    # Get the filename
    mother_fn = find_file(name, results, child['time'][0])
    return mother_fn

# ------------------------------------------------------------------------------------
#                           Interpolation schemes
# ------------------------------------------------------------------------------------
def nearest_neighbor(mother, M, M_ch, ind, nodefield, cellfield, alias):
    '''
    Interpolate from mother grid using the nearest neighbor interpolation scheme
    '''
    # Find the nearest node/cell in the mother grid
    nearest_mother_node = M.find_nearest(M_ch.x,  M_ch.y,  grid = 'node')
    nearest_mother_cell = M.find_nearest(M_ch.xc, M_ch.yc, grid = 'cell')
    horizontal = {}

    # Loop over nodes in child
    print('  - node data')
    for varname in nodefield:
        try:
            if len(mother.variables[varname].shape) == 2:
                horizontal[varname] = mother[varname][:][ind, M.cropped_nodes[nearest_mother_node]]
            elif len(mother.variables[varname].shape) == 3:
                horizontal[varname] = mother[varname][:][ind,:, M.cropped_nodes[nearest_mother_node]].transpose()
        except:
            if varname in alias.keys():
                horizontal[varname] = horizontal[alias[varname]]
            elif varname in ['rho1', 'rmean1']:
                horizontal[varname] = sw.dens0(horizontal['salinity'], horizontal['temp'])
            elif varname == 'zice':
                horizontal[varname] = np.zeros(horizontal['zeta'].shape) # zeta for å få rett dimensjon
            else:
                print(f'    - {varname} could not be interpolated to restart')
                continue
        print(f'    - {varname} interpolated')

    if cellfield is not None:
        print('\n  - cell data')
        for varname in cellfield:
            try:
                if len(mother.variables[varname].shape) == 2:
                    horizontal[varname] = mother[varname][:][ind, M.cropped_cells[nearest_mother_cell]]
                elif len(mother.variables[varname].shape) == 3:
                    horizontal[varname] = mother[varname][:][ind,:, M.cropped_cells[nearest_mother_cell]].transpose()
            except:
                print(f'    - {varname} could not be interpolated to restart')
                continue
            print(f'    - {varname} interpolated')

    # Store info needed for vertical interpolation
    class grid_info: pass
    grid_info.z_node_siglay_mother = M.h[nearest_mother_node, None] * M.siglay[nearest_mother_node, :] # mother siglay depth-levels at child nodes
    grid_info.z_cell_siglay_mother = np.mean(grid_info.z_node_siglay_mother[M_ch.tri], axis=1)         # mother siglay depth-levels at child cells

    grid_info.z_node_siglev_mother = M.h[nearest_mother_node, None] * M.siglev[nearest_mother_node, :] # mother siglay depth-levels at child nodes
    grid_info.z_cell_siglev_mother = np.mean(grid_info.z_node_siglev_mother[M_ch.tri], axis=1)         # mother siglay depth-levels at child cells

    return horizontal, grid_info


def vertical_interpolation(data, child, dpt):
    '''
    Linear vertical interpolation of ROMS data to FVCOM-depths.
    '''
    var = [*data]
    vertical_data = {}

    # Load depth and sigma layer information
    h = np.array(child['h'][:])
    tri = np.array(child['nv'][:].T) - 1
    siglay = np.array(child['siglay'][:].T)
    siglev = np.array(child['siglev'][:].T)

    # Get depths to interpolate to and from
    # Child center depth
    h_center = np.mean(h[tri], axis = 1)[:]
    siglay_center = np.mean(siglay[tri], axis = 1)
    siglev_center = np.mean(siglev[tri], axis = 1)

    # Sigma
    node_dpt_siglay_child  = h[:, None] * siglay
    cell_dpt_siglay_child  = h_center[:, None] * siglay_center

    # Siglev
    node_dpt_siglev_child  = h[:, None] * siglev
    cell_dpt_siglev_child  = h_center[:, None] * siglev_center
    
    # Get interpolation coefficients and data indices
    print('  - Calculate vertical weights')
    nlay_ind1, nlay_ind2, nlay_weigths1, nlay_weigths2 = vi.calc_interp_matrices(-dpt.z_node_siglay_mother.T, -node_dpt_siglay_child.T)
    clay_ind1, clay_ind2, clay_weigths1, clay_weigths2 = vi.calc_interp_matrices(-dpt.z_cell_siglay_mother.T, -cell_dpt_siglay_child.T)
    nlev_ind1, nlev_ind2, nlev_weigths1, nlev_weigths2 = vi.calc_interp_matrices(-dpt.z_node_siglev_mother.T, -node_dpt_siglev_child.T)
    clev_ind1, clev_ind2, clev_weigths1, clev_weigths2 = vi.calc_interp_matrices(-dpt.z_cell_siglev_mother.T, -cell_dpt_siglev_child.T)

    print('  - Interpolate vertical data to the child')
    for field in var:
        if len(data[field].shape) == 1:
            vertical_data[field] = data[field]
            continue

        if data[field].shape == siglay.T.shape:
            vertical_data[field] = data[field][nlay_ind1, range(0, data[field].shape[1])] * nlay_weigths1 + \
                                   data[field][nlay_ind2, range(0, data[field].shape[1])] * nlay_weigths2 
            
        elif data[field].shape == siglay_center.T.shape:
            vertical_data[field] = data[field][clay_ind1, range(0, data[field].shape[1])] * clay_weigths1 + \
                                   data[field][clay_ind2, range(0, data[field].shape[1])] * clay_weigths2 

        if data[field].shape == siglev.T.shape:
            vertical_data[field] = data[field][nlev_ind1, range(0, data[field].shape[1])] * nlev_weigths1 + \
                                   data[field][nlev_ind2, range(0, data[field].shape[1])] * nlev_weigths2 

        elif data[field].shape == siglev_center.T.shape:
            vertical_data[field] = data[field][clev_ind1, range(0, data[field].shape[1])] * clev_weigths1 + \
                                   data[field][clev_ind2, range(0, data[field].shape[1])] * clev_weigths2 
    return vertical_data

def dump_data(data, child):
    '''
    Write interpolated data from mother model to the child model restart file
    '''
    print('\nDump data to restart/initial file')
    var = [*data]
    for field in var:
        try:
            child[field][0,:] = data[field]
        except:
            print(f'    - {field} could not be dumped to child')
    return child

class InputError(Exception): pass