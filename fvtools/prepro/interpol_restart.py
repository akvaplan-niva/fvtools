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
from fvtools.grid.tools import Filelist, num2date, date2num

global versionnr
versionnr = 1.4

# versionnr 1.0: Adds support for different #sigmalayers in the mother and child
# versionnr 1.1: Writes full field to the restart file
# versionnr 1.2: Crops the full mother grid to fit the smaller one (to speed it up significantly, smaller KDTrees)
# versionnr 1.3: basically just string fixes, adding metadata to the restart file
# versionnr 1.4: Make an empty restart file, optionally use this instead of childfn

def main(child_fn      = None,
         startdate     = None,
         child_grid    = None,
         result_folder = None,
         name          = None,
         motherfn      = None,
         filelist      = None,
         speed         = False):
    '''
    Two options for child specification:
    child_fn:      - path to an existing FVCOM restart file
    startdate      - date ("yyyy-mm-dd-hh")
    
    
    Two methods to specify the restart file:
    filelist       - filelist made using fvcom_make_filelist.py
    results_folder - folders where the mother grid is located (textfile with paths to the folders)
        --> name   - Then you must also provide the name of the numerical experiment (ie. 'PO12')

    Optional:
    speed          - sometimes works, many times not - hence False by default
    '''
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
        M.zeta    = M.load_netCDF(M.filepath, 'zeta', ind)
        M_ch.zeta = data['zeta']

        print('\n- Vertical interpolation') 
        data = vertical_interpolation(data, child, dpt) 

        # Dump to netCDF
        child = dump_data(data, child)

        # Done!
        child.mother           = mother_fn
        child.data_age         = f'data dumped to this restart file by interpol_restart.py at {time.ctime(time.time())}'
        child.interpol_version = versionnr
        child.interp_folder    = os.getcwd()

    # Show results (now that the data are safely stored :])
    plot_interps(mother_fn, child_fn, M, M_ch, ind)
    print('Fin.')

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
    grid_info.z_cell_siglay_mother = np.mean(grid_info.z_node_siglay_mother[M_ch.tri], axis=1)                # mother siglay depth-levels at child cells

    grid_info.z_node_siglev_mother = M.h[nearest_mother_node, None] * M.siglev[nearest_mother_node, :] # mother siglay depth-levels at child nodes
    grid_info.z_cell_siglev_mother = np.mean(grid_info.z_node_siglev_mother[M_ch.tri], axis=1)                # mother siglay depth-levels at child cells

    return horizontal, grid_info


def vertical_interpolation(data, child, dpt):
    '''
    Linear vertical interpolation of ROMS data to FVCOM-depths.
    '''
    var = [*data]
    vertical_data = {}

    # Get depths to interpolate to and from
    # Sigma
    node_dpt_siglay_child  = child['h'][:][:, None] * child['siglay'][:].T
    cell_dpt_siglay_child  = child['h_center'][:][:, None] * child['siglay_center'][:].T

    # Siglev
    node_dpt_siglev_child  = child['h'][:][:, None] * child['siglev'][:].T
    cell_dpt_siglev_child  = child['h_center'][:][:, None] * child['siglev_center'][:].T
    
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

        if data[field].shape == child['siglay'].shape:
            vertical_data[field] = data[field][nlay_ind1, range(0, data[field].shape[1])] * nlay_weigths1 + \
                                   data[field][nlay_ind2, range(0, data[field].shape[1])] * nlay_weigths2 
        elif data[field].shape == child['siglay_center'].shape:
            vertical_data[field] = data[field][clay_ind1, range(0, data[field].shape[1])] * clay_weigths1 + \
                                   data[field][clay_ind2, range(0, data[field].shape[1])] * clay_weigths2 

        if data[field].shape == child['siglev'].shape:
            vertical_data[field] = data[field][nlev_ind1, range(0, data[field].shape[1])] * nlev_weigths1 + \
                                   data[field][nlev_ind2, range(0, data[field].shape[1])] * nlev_weigths2 

        elif data[field].shape == child['siglev_center'].shape:
            vertical_data[field] = data[field][clev_ind1, range(0, data[field].shape[1])] * clev_weigths1 + \
                                   data[field][clev_ind2, range(0, data[field].shape[1])] * clev_weigths2 
    return vertical_data

def dump_data(data, child):
    print('\nDump data to restart/initial file')
    var = [*data]
    for field in var:
        try:
            child[field][0,:] = data[field]
        except:
            print(f'    - {field} could not be dumped to child')
    return child

def plot_interps(mother_file, child_file, M_mother, M_child, ind):
    '''
    Visualize results from the interpolation in an attempt at building confidence in the quality of the interpolation
    '''
    def make_comparison_figure(M_mother, mother_field, M_child, child_field, levels, cmap, title):
        fig, ax = plt.subplots(1,2,figsize = (20,10))
        tp = ax[0].tricontourf(M_child.x, M_child.y, M_child.tri, child_field, levels = levels, extend = 'both')
        ax[0].set_title(f'child {title}')
        ax[0].set_aspect('equal')

        tp = ax[1].tricontourf(M_mother.x, M_mother.y, M_mother.tri, mother_field, levels = levels, extend = 'both')
        ax[1].set_title(f'mother {title}')
        ax[1].set_xlim(M_child.x.min(), M_child.x.max())
        ax[1].set_ylim(M_child.y.min(), M_child.y.max())
        ax[1].set_aspect('equal')

        fig.subplots_adjust(right = 0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.675])
        cb = fig.colorbar(tp, cax = cbar_ax)
        cb.set_label(f'mother {title}')

    with netCDF4.Dataset(mother_file, 'r') as mother:
        with netCDF4.Dataset(child_file, 'r') as child:
            levels  = np.linspace(child.variables['zeta'][0, :].min(), child.variables['zeta'][0, :].max(), 30)
            make_comparison_figure(M_mother, mother['zeta'][ind, M_mother.cropped_nodes], 
                                   M_child, child['zeta'][0,:], 
                                   levels, cmo.cm.amp, 'sea surface elevation [m]')

            levels  = np.linspace(child.variables['temp'][0, -1, :].min(), child.variables['temp'][0, 0, :].max(), 30)
            make_comparison_figure(M_mother, mother['temp'][ind, 10, M_mother.cropped_nodes], 
                                   M_child, child['temp'][0, 10, :], 
                                   levels, cmo.cm.thermal, 'sigma = 10 temperature [C]')

            levels  = np.linspace(29, 35, 30)
            make_comparison_figure(M_mother, mother['salinity'][ind, 0, M_mother.cropped_nodes], 
                                   M_child, child['salinity'][0, 0, :], 
                                   levels, cmo.cm.haline, 'surface salinity [psu]]')
    plt.show(block = False)

# Make an empty restartfile
def make_initial_file(M, initial_time, obc_type = 0):
    '''
    Empty restartfile for this experiment
    - M:            FVCOM_grid
    - initial_time: Date string 'yyyy-mm-dd-hh'
    - obc_type:     1 (fvcom2fvcom), 2 (??), 3 (??)
    '''
    print(f"- Create {M.casename}_initial.nc\n")
    nums   = [int(number) for number in initial_time.split('-')]
    fvtime = date2num([datetime.datetime(nums[0], nums[1], nums[2], nums[3], tzinfo = datetime.timezone.utc)])

    requested_by_restart = {
        'zeta': (('time', 'node'), 'single'),
        'salinity': (('time', 'siglay', 'node'), 'single'),
        'temp': (('time', 'siglay', 'node'), 'single'),
        'iint': (('time', ), 'int32'),
        'ua': (('time', 'nele'), 'single'),
        'va': (('time', 'nele'), 'single'),
        'u': (('time', 'siglay', 'nele'), 'single'),
        'v': (('time', 'siglay', 'nele'), 'single'),
        'tauc': (('time', 'nele'), 'single'),
        'omega': (('time', 'siglev', 'node'), 'single'),
        'ww': (('time', 'siglay', 'nele'), 'single'),
        'viscofm': (('time', 'siglay', 'nele'), 'single'),
        'viscofh': (('time', 'siglay', 'node'), 'single'),
        'km': (('time', 'siglev', 'node'), 'single'),
        'kh': (('time', 'siglev', 'node'), 'single'),
        'kq': (('time', 'siglev', 'node'), 'single'),
        'q2': (('time', 'siglev', 'node'), 'single'),
        'q2l': (('time', 'siglev', 'node'), 'single'),
        'l': (('time', 'siglev', 'node'), 'single'),
        'cor': (('nele', ), 'single'),
        'cc_sponge': (('nele', ), 'single'),
        'et': (('time', 'node', ), 'single'),
        'tmean1': (('siglay', 'node'), 'single'),
        'smean1': (('siglay', 'node'), 'single'),
        'obc_nodes': (('nobc', ), 'int32'),
        'obc_type': (('nobc', ), 'int32'),
        'rho1': (('time', 'siglay', 'node'), 'single'),
        'rmean1': (('siglay', 'node'), 'single'),
        'zice': (('siglay', 'node'), 'single')
        }

    grid_fields = {'x': (('node',), 'single'),
                   'y': (('node',), 'single'),
                   'xc': (('nele',), 'single'),
                   'yc': (('nele',), 'single'),
                   'lat': (('node',), 'single'),
                   'lon': (('node', ), 'single'),
                   'latc': (('nele',), 'single'),
                   'lonc': (('nele',), 'single'),
                   'h': (('node',), 'single'),
                   'h_center': (('nele',), 'single'),
                   'nv': (('three', 'nele'), 'int32'),
                   'siglay': (('siglay', 'node'), 'single'),
                   'siglev': (('siglev', 'node'), 'single'),
                   'siglay_center': (('siglay', 'nele'), 'single'),
                   'siglev_center': (('siglev', 'nele'), 'single')
                   }

    aliases = {'nv': 'tri',
               'h_center': 'hc'}

    with netCDF4.Dataset(f'{M.casename}_initial.nc', 'w') as initial:
        timedim  = initial.createDimension('time', 0)
        nodedim  = initial.createDimension('node', len(M.x))
        celldim  = initial.createDimension('nele', len(M.xc))
        threedim = initial.createDimension('three', 3)
        levdim   = initial.createDimension('siglev', M.siglev.shape[-1])
        laydim   = initial.createDimension('siglay', M.siglev.shape[-1]-1)
        datestr  = initial.createDimension('DateStrLen', 26)
        nobc     = initial.createDimension('nobc', len(M.obc_nodes))

        time             = initial.createVariable('time', 'single', ('time',))
        time.units       = 'days since 1858-11-17 00:00:00'
        time.format      = 'modified julian day (MJD)'
        time.time_zone   = 'UTC'

        Itime            = initial.createVariable('Itime', 'int32', ('time',))
        Itime.units      = 'days since 1858-11-17 00:00:00'
        Itime.format     = 'modified julian day (MJD)'
        Itime.time_zone  = 'UTC'

        Itime2           = initial.createVariable('Itime2', 'int32', ('time',))
        Itime2.units     = 'msec since 00:00:00'
        Itime2.time_zone = 'UTC'

        # Create variables in the netCDF file
        for key in requested_by_restart.keys():
            initial.createVariable(key, requested_by_restart[key][1], requested_by_restart[key][0])

        for key in grid_fields.keys():
            initial.createVariable(key, grid_fields[key][1], grid_fields[key][0])

        # Dump grid info to the netCDF file
        for key in grid_fields.keys():
            if key not in aliases.keys():
                if key in ['siglev', 'siglay', 'siglev_center', 'siglay_center']:
                    initial[key][:] = getattr(M, key).T
                else:
                    initial[key][:] = getattr(M, key)

            else:
                if key =='nv':
                    initial[key][:] = getattr(M, aliases[key]).T+1
                else:
                    initial[key][:] = getattr(M, aliases[key])

        # Set initial values to zero, give obc_nodes and set OBC type to desired value
        for key in requested_by_restart.keys():
            if key == 'obc_type':
                initial[key][:] = 1

            elif key == 'obc_nodes':
                initial[key][:] = M.obc_nodes

            else:
                initial[key][:] = obc_type

        # Set initial time
        initial['time'][:] = fvtime[0]
        initial['Itime'][:] = int(fvtime[0])
        initial['Itime2'][:] = int((fvtime[0]-int(fvtime[0]))*24*60*60*1000)
    return f'{M.casename}_initial.nc'

class InputError(Exception): pass