# -----------------------------------------------------------------------------------------
#                             Create nest from ROMS to FVCOM
# -----------------------------------------------------------------------------------------
import getpass
import numpy as np
import pandas as pd
import progressbar as pb
import pyproj
import netCDF4
import matplotlib.pyplot as plt
import os
import sys
import time as time_mod
from ..nesting import vertical_interpolation as vi

from glob import glob
from datetime import datetime, timedelta
from ..grid.fvcom_grd import FVCOM_grid
from scipy.io import loadmat
from pyproj import Proj
from time import gmtime, strftime
from netCDF4 import Dataset
from ..gridding.prepare_inputfiles import write_FVCOM_bath
from scipy.spatial import KDTree

def main(fvcom_grd,
         nest_grd,
         outfile,
         start_time,
         stop_time,
         mother  = None,
         weights = [2.5e-4, 2.5e-5],
         R       = None,
         avg     = True):
    '''
    Nest from NorKyst-800 or NorShelf-2.4km (Support for other ROMS models should be easy
    to implement)
    ------
    - Reads a FVCOM grid, identifies the nestingzone, creates interpolation coefficients,
      interpolates data to the nestingzone and stores to a FVCOM readable NetCDF file.

    input:
    --
    fvcom_grd  - FVCOM grid file         (M.mat, M.npy)
    nest_grd   - FVCOM nesting grid file (ngrd.mat, ngrd.npy)
    outfile    - name of the output file
    start_time - yyyy-mm-dd
    stop_time  - yyyy-mm-dd

    optional:
    --
    mother     - 'NK' for NorKyst-800 or 'NS' for NorShelf-2.4km
                 (Should be easy to add other ROMS versions as well)

    weights    - tuple giving the weight interval for the nest nodes and cells.
                 By default, weights = [2.5e-4, 2.5e-5].

    R          - Width of nestingzone

    avg        - uses daily average data if True
    '''
    if mother is None:
        raise InputError('You must specify which ROMS model you want to nest to.\n'+\
                         'More info: roms_nesting_fg.main?')

    # Get grid info
    # -----------------------------------------------------------------------------
    print('Load mesh info:')
    print('  - Read the FVCOM grid')
    M = FVCOM_grid(fvcom_grd)         # The full fvcom grid. Needed to get OBC nodes and vertical coordinates.

    print('  - Read nest grid info')
    NEST = NEST_grid(nest_grd, M)     # The grid that receives ROMS data (ngrd)

    print('  - Calculate weight coefficients')
    NEST.calcWeights(M, w1 = max(weights), w2 = min(weights))

    # Create a filelist
    # -----------------------------------------------------------------------------
    print('\nPrepare the filelist')
    time, path, index = make_fileList(start_time, stop_time, mother, avg)

    # Get the ROMS grid info
    # -----------------------------------------------------------------------------
    HI = []; MET = []
    print('\nCreate a smooth ROMS-FVCOM bathymetry transition')
    if mother == 'NS':
        ROMS_grd = ROMS_grid(path[0])

    else:
        # Check if we have to load two grids
        HI = [fil for fil in path if 'cluster' in fil]
        MET = [fil for fil in path if 'thredds.met' in fil]
        if any(HI) and any(MET):
            print('  - We are using two different ROMS setups, and thus need to load two grids...')
            # Find the thredds path...
            thredds_path = [fil for fil in path if 'thredds' in fil][0]
            local_path   = [fil for fil in path if 'cluster' in fil][0]
            # use a date I know is available on thredds
            # ----
            ROMS_grd_thredds = ROMS_grid(thredds_path)
            ROMS_grd_local   = ROMS_grid(local_path)

        else:
            ROMS_grd = ROMS_grid(path[0])

    if R is None:
        R = NEST.R[0][0]

    # Smooth the bathymetry in the nesting zone (This should be the same for both NK grids)
    # Overwrite the nest topo from matlab with topo from ROMS
    # -----------------------------------------------------------------------------
    print('- Set the bathymetry near the nestingzone equal to that of ROMS')
    if any(HI) and(MET):
        M, NEST = nestingtopo(ROMS_grd_local, M, NEST, R)
    else:
        M, NEST = nestingtopo(ROMS_grd, M, NEST, R)


    # Create the netcdf forcing file
    # -----------------------------------------------------------------------------
    print(f'\nThe nest forcing file will be: {outfile}')
    create_nc_forcing_file(outfile, NEST)

    # Find The nearest 4 and add vertical interpolation matrices
    # -----------------------------------------------------------------------------
    print('\nFind the nearest 4 interpolation coefficients for the nestingzone')
    if any(HI) and any(MET):
        # Prepare the thredds files
        N4 = nearest4(NEST, ROMS_grd_thredds)
        N4 = vi.add_vertical_interpolation2N4(N4)

        # Prepare the local files
        N4_local = nearest4(NEST, ROMS_grd_local)
        N4_local = vi.add_vertical_interpolation2N4(N4_local)

    else:
        N4 = nearest4(NEST,ROMS_grd)
        N4 = vi.add_vertical_interpolation2N4(N4)

    # Read the data, interpolate to FVCOM nest and store to forcing file
    # -----------------------------------------------------------------------------
    print('\nCreate the roms2fvcom nesting file')
    if any(HI) and any(MET):
        roms2fvcom(outfile, time, path, index, N4, N4_local = N4_local)
    else:
        roms2fvcom(outfile, time, path, index, N4)

    vi.calc_uv_bar(outfile, NEST, M)

    # Check if there is missing data
    # ----
    data = Dataset(outfile)
    diagnose_time(data['Itime'][:], data['Itime2'][:])
    print('\n--> Fin.')


# -----------------------------------------------------------------------------------------------
#                                     fileList stuff
# -----------------------------------------------------------------------------------------------
def make_fileList(start_time, stop_time, mother, avg):
    '''
    Go through the met office thredds server and link points in time to files.
    format: yyyy-mm-dd-hh
    '''
    dates   = prepare_dates(start_time, stop_time)
    time    = np.empty(0)
    path    = []
    index   = []
    local_file    = []; thredds_file = []
    missing_dates = []
    for date in dates:
        # See where the files are available
        # ----
        try:
            # NorKyst
            if mother == 'NK':
                thredds_file = test_thredds_path(date, mother)
                file         = thredds_file

            # NorShelf
            elif mother == 'NS':
                thredds_file = test_norshelf_path(date, avg)
                file         = thredds_file

            else:
                raise InputError(f'{mother} is not a valid option\n Add it! :)')

        # OSErrors will occur when the thredds server is offline/the requested file is not available
        except:
            if mother == 'NS':
                print(f'- warning: {date} not found')
                continue

            # If NorKyst was not available on thredds, check locally
            if mother == 'NK':
                try:
                    local_file = test_local_path(date)
                    file       = local_file

                    #thredds_file = test_thredds_path(date, mother)
                    #file         = thredds_file

                except:
                    print('We did not find NorKyst-800 data for ' + str(date) + 'on thredds or in the specified folders.'+\
                          '--> Run "mend_gaps.py" to fill the gaps.')
                    continue

        # We will come in problems the day before thredds lacks norkyst data since the
        # files are not stored with the same datum. We avoid that problem by storing both.
        # ----
        if any(local_file) and any(thredds_file):
            print(f'- checking: {thredds_file} \n- checking: {local_file}')
            d                = Dataset(thredds_file, 'r')
            dlocal           = Dataset(local_file, 'r')

            # Load the timevectors
            # ----
            t_thredds_roms   = netCDF4.num2date(d.variables['ocean_time'][:],units = d.variables['ocean_time'].units)
            if type(t_thredds_roms) is np.ndarray:
                ttroms = t_thredds_roms.data
            else:
                ttroms = t_thredds_roms.data

            t_thredds_fvcom  = netCDF4.date2num(ttroms, units = 'days since 1858-11-17 00:00:00')
            t_local_roms     = netCDF4.num2date(dlocal.variables['ocean_time'][:],units = dlocal.variables['ocean_time'].units)

            if type(t_local_roms) is np.ndarray:
                tlroms = t_local_roms
            else:
                tlroms = t_local_roms.data
            t_local_fvcom    = netCDF4.date2num(tlroms, units = 'days since 1858-11-17 00:00:00')

            # Expand the total time, path and index vectors
            # ----
            time             = np.append(time, t_thredds_fvcom)
            path             = path + [thredds_file]*len(t_thredds_fvcom)
            index.extend(list(range(len(t_thredds_fvcom))))

            time             = np.append(time, t_local_fvcom)
            path             = path + [local_file]*len(t_local_fvcom)
            index.extend(list(range(len(t_local_fvcom))))

        # if we just have thredds or a local copy
        # -----
        else:
            print(f'- checking: {file}')
            d        = Dataset(file)

            # Convert from ROMS units to datetime
            # ----
            t_roms   = netCDF4.num2date(d.variables['ocean_time'][:], units = d.variables['ocean_time'].units)

            # Convert from datetime to FVCOM units
            # ----
            if type(t_roms) is np.ndarray:
                trom = t_roms
            else:
                trom = t_roms.data

            t_fvcom  = netCDF4.date2num(trom, units = 'days since 1858-11-17 00:00:00')

            # Append the timesteps, paths and indices to a fileList
            # ----
            time     = np.append(time, t_fvcom)
            path     = path + [file]*len(t_fvcom)
            index.extend(list(range(len(t_fvcom))))

        local_file = []; thredds_file = []; file = []

    # --------------------------------------------------------------------------------------------
    #     Remove overlap
    # --------------------------------------------------------------------------------------------
    time_no_overlap     = [time[-1]]
    path_no_overlap     = [path[-1]]
    index_no_overlap    = [index[-1]]

    for n in range(len(time)-1, 0, -1):
        if time[n-1] < time_no_overlap[0]:
            time_no_overlap.insert(0, time[n-1])
            path_no_overlap.insert(0, path[n-1])
            index_no_overlap.insert(0, index[n-1])

    return np.array(time_no_overlap), path_no_overlap, index_no_overlap

def test_thredds_path(date, mother):
    '''
    Check if the file exists that day, and that it has enough data
    '''

    # Get the URLs
    # ----
    file = get_norkyst_url(date)

    # Test if the data is good
    # ----
    d    = Dataset(file, 'r')

    if len(d.variables['ocean_time'][:])<24:       # Check to see if empty
        print(f'{date} does not have a complete timeseries')
        thredds_file = get_norkyst_local(date-timedelta(days=1))

    file_tomorrow = get_norkyst_url(date+timedelta(days=1))
    tmp           = Dataset(file_tomorrow, 'r')
    tmp.close()

    d.close()

    return file

def test_local_path(date):
    '''
    See if the local file exists that day, and has enough data
    '''
    file         = get_norkyst_local(date)
    d            = Dataset(file, 'r')

    if len(d.variables['ocean_time'][:])<24: # Check to see if empty (Might be too strict to crash the code)
        raise ValueError(f'{file} does not have a complete timeseries')

    d.close()
    return file

def test_norshelf_path(date, avg):
    '''
    Load norshelf path
    '''
    file = get_norshelf_day_url(date, avg)

    # Test if the data is good
    # ----
    forecast_nr = 0
    while True:
        try:
            d    = Dataset(file, 'r')
            if avg:
                if len(d.variables['ocean_time'][:])<1:  # Check to see if empty
                    print('- check forecast')
                    raise OSError
                else:
                    break

            else:
                if len(d.variables['ocean_time'][:])<24:  # Check to see if empty
                    print('- check forecast')
                    raise OSError
                else:
                    break
        except:
            file = get_norshelf_fc_url(date-timedelta(days=forecast_nr), avg)
            forecast_nr +=1

        if forecast_nr > 3:
            print(f'-> no forecast available for {date}')
            raise OSError

    d.close()

    return file

# -----------------------------------------------------------------------------------------------
#                                   Download procedures
# -----------------------------------------------------------------------------------------------
def prepare_dates(start_time,stop_time):
    '''
    returns pandas array of dates needed
    '''
    start       = datetime(int(start_time.split('-')[0]), int(start_time.split('-')[1]),\
                           int(start_time.split('-')[2]))
    stop        = datetime(int(stop_time.split('-')[0]), int(stop_time.split('-')[1]),\
                           int(stop_time.split('-')[2]))

    return pd.date_range(start,stop)

def get_roms_data(d, index, N4):
    '''
    Dumps roms data from the netcdf file and prepares for interpolation
    '''

    # Initialize storage
    class dumped: pass

    # Selective load (load the domain within the given limits)
    # ----
    dumped.salt = d.variables.get('salt')[index, :, N4.m_ri:(N4.x_ri+1), N4.m_rj:(N4.x_rj+1)][:,N4.cropped_rho_mask].transpose()
    dumped.temp = d.variables.get('temp')[index, :, N4.m_ri:(N4.x_ri+1), N4.m_rj:(N4.x_rj+1)][:,N4.cropped_rho_mask].transpose()
    dumped.zeta = d.variables.get('zeta')[index,    N4.m_ri:(N4.x_ri+1), N4.m_rj:(N4.x_rj+1)][N4.cropped_rho_mask]

    dumped.u    = d.variables.get('u')[index, :, N4.m_ui:(N4.x_ui+1), N4.m_uj:(N4.x_uj+1)][:,N4.cropped_u_mask].transpose()

    dumped.v    = d.variables.get('v')[index, :, N4.m_vi:(N4.x_vi+1), N4.m_vj:(N4.x_vj+1)][:,N4.cropped_v_mask].transpose()

    return dumped

# ------------------------------------------------------------------------------------------------------------------------
#                     Find the locations/whereabouts of the files we need for each day
# ------------------------------------------------------------------------------------------------------------------------
def get_norshelf_fc_url(date, avg):
    '''
    Give it a date, and you will get the corresponding url in return
    '''
    date        = date-timedelta(days=1) # The times in norshelf files is delayed by one day for some reason?
    if avg:
        https       = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_avg_fc_"
    else:
        https       = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_qck_fc_"
    return https+ "{0.year}{0.month:02}{0.day:02}".format(date) + "T00Z.nc"

def get_norshelf_day_url(date, avg):
    '''
    Give it a date, and you will get the corresponding url in return
    '''
    date        = date
    if avg:
        https       = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_avg_an_"
    else:
        https       = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_qck_an_"
    return https+ "{0.year}{0.month:02}{0.day:02}".format(date) + "T00Z.nc"

def get_norkyst_url(date):
    '''
    Give it a date, and you will get the corresponding url in return
    '''
    https       = 'https://thredds.met.no/thredds/dodsC/fou-hi/new_norkyst800m/his/ocean_his.an.'
    year        = str(date.year)
    month       = '{:02d}'.format(date.month)
    day         = '{:02d}'.format(date.day)
    return https+year+month+day+'.nc'

def get_norkyst_local(date):
    '''
    Looks for NorKyst data in the predefined folders.
    '''

    # Some of these will need other interpolation coefficients than the ones we have calculated
    # based on THREDDS data

    folders     = ['/cluster/shared/NS9067K/apn_backup/ROMS/NK800_2016-2017',\
                   '/cluster/shared/NS9067K/apn_backup/ROMS/NK800_2017',\
                   '/cluster/shared/NS9067K/apn_backup/ROMS/NK800_2018',\
                   '/cluster/shared/NS9067K/apn_backup/ROMS/NK800_2019',\
                   '/cluster/shared/NS9067K/apn_backup/ROMS/NK800_20194']

    # look for subfolders in parents
    # ----
    subfolders  = bottom_folders(folders)

    # extract the .nc files in those folders
    # ----
    all_ncfiles = list_ncfiles(subfolders)

    # return the file with the correct date
    return connect_date_to_file(all_ncfiles, date)

def list_ncfiles(dirs):
    '''
    returns list of all files in directories (or in one single directory)
    '''
    ncfiles = []
    for dr in dirs:
        stuff   = os.listdir(dr)
        ncfiles.extend([dr+'/'+fil for fil in stuff if '.nc' in fil])
    return ncfiles

def connect_date_to_file(all_ncfiles, date):
    '''
    # --> IMR NorKyst data is stored starting at 01:00 and ending at 24:00, hence they are not
    #     in sync with the data found at norstore (gah!), meaning that we (may) need to find more
    #     than one file. Should not be a problem, but it is surely annoying...
    '''

    # Identify the files using their names (ie. not a filelist approach?)
    # ----
    year        = str(date.year)
    month       = '{:02d}'.format(date.month)
    day         = '{:02d}'.format(date.day)

    files       = [files for files in all_ncfiles if year+month+day in files]

    # I want the file that starts the same date as my date
    # ----
    for f in files:
        if year+month+day in f.split('_')[-1].split('-')[0]:
            read_file = f
            break

    return read_file


def bottom_folders(folders):
    '''
    Returns the folders on the bottom of the pyramid (hence the name)
    mandatory:
    folders   - parent folder(s) to cycle through
    '''
    # ----
    dirs = []
    for folder in folders:
        dirs.extend([x[0] for x in os.walk(folder)])

    # remove folders that are not at the top of the tree
    # ----
    leaf_branch = []
    for dr in dirs:
        if dr[-1]=='/':
            continue
        else:
            # This string is at the end of the branch, thus this is where the data is stored
            # ----
            leaf_branch.append(dr)
    return leaf_branch

# ------------------------------------------------------------------------------------------------
#                                Interpolation coefficients
# ------------------------------------------------------------------------------------------------
def nearest4(NEST, ROMS_grd):
    '''
    Create nearest four indices and weights for all of the fields
    '''
    # This could probably be coded in a more consise and good looking way... But what the heck
    # ----
    widget         = ['\n- Finding nearest 4 and weights: ', pb.Percentage(), pb.Bar()]
    bar            = pb.ProgressBar(widgets=widget, maxval=len(NEST.xn)+len(NEST.xc))
    N4             = N4ROMS(NEST, ROMS_grd)

    # The ROMS points covered by the FVCOM domain
    # --------------------------------------------------------------------------------------------
    N4.fv_rho_mask = ROMS_grd.crop_rho(xlim=[NEST.xn.min() - 5000, NEST.xn.max() + 5000],
                                       ylim=[NEST.yn.min() - 5000, NEST.yn.max() + 5000])

    N4.fv_u_mask   = ROMS_grd.crop_u(xlim=[NEST.xn.min() - 5000, NEST.xn.max() + 5000],
                                     ylim=[NEST.yn.min() - 5000, NEST.yn.max() + 5000])

    N4.fv_v_mask   = ROMS_grd.crop_v(xlim=[NEST.xn.min() - 5000, NEST.xn.max() + 5000],
                                     ylim=[NEST.yn.min() - 5000, NEST.yn.max() + 5000])

    # create a cropped version of the mask
    # --------------------------------------------------------------------------------------------
    # The indices that define the area we need to download
    rho_i, rho_j = np.where(N4.fv_rho_mask)
    N4.m_ri  = min(rho_i); N4.x_ri = max(rho_i)
    N4.m_rj  = min(rho_j); N4.x_rj = max(rho_j)

    u_i, u_j = np.where(N4.fv_u_mask)
    N4.m_ui  = min(u_i); N4.x_ui = max(u_i)
    N4.m_uj  = min(u_j); N4.x_uj = max(u_j)

    v_i, v_j = np.where(N4.fv_v_mask)
    N4.m_vi  = min(v_i); N4.x_vi = max(v_i)
    N4.m_vj  = min(v_j); N4.x_vj = max(v_j)

    # The mask (to be used later on)
    N4.cropped_rho_mask = N4.fv_rho_mask[N4.m_ri:N4.x_ri+1, N4.m_rj:N4.x_rj+1]
    N4.cropped_u_mask   = N4.fv_u_mask[N4.m_ui:N4.x_ui+1, N4.m_uj:N4.x_uj+1]
    N4.cropped_v_mask   = N4.fv_v_mask[N4.m_vi:N4.x_vi+1, N4.m_vj:N4.x_vj+1]


    # Cropping the ROMS land mask
    # --------------------------------------------------------------------------------------------
    Land_rho       = ROMS_grd.rho_mask[N4.fv_rho_mask]
    Land_u         = ROMS_grd.u_mask[N4.fv_u_mask]
    Land_v         = ROMS_grd.v_mask[N4.fv_v_mask]

    # Cropping the coordinates
    # --------------------------------------------------------------------------------------------
    x_rho          = ROMS_grd.x_rho[N4.fv_rho_mask]
    y_rho          = ROMS_grd.y_rho[N4.fv_rho_mask]

    x_u            = ROMS_grd.x_u[N4.fv_u_mask]
    y_u            = ROMS_grd.y_u[N4.fv_u_mask]

    x_v            = ROMS_grd.x_v[N4.fv_v_mask]
    y_v            = ROMS_grd.y_v[N4.fv_v_mask]

    # Getting the psi mask
    # --------------------------------------------------------------------------------------------
    umask          = np.logical_and(N4.fv_u_mask[1:,:],N4.fv_u_mask[:-1,:])
    vmask          = np.logical_and(N4.fv_v_mask[:,1:],N4.fv_v_mask[:,:-1])
    psi_mask       = np.logical_and(umask,vmask)

    # psi coordinates (Why is this not equal to the rho points?)
    # --------------------------------------------------------------------------------------------
    x_psi          = (ROMS_grd.x_u[1:,:]+ROMS_grd.x_u[:-1,:])[psi_mask]/2
    y_psi          = (ROMS_grd.y_v[:,1:]+ROMS_grd.y_v[:,:-1])[psi_mask]/2

    # Build the KDTrees, find the nearest 4 rho, u and v points
    # --------------------------------------------------------------------------------------------
    # Create position vectors
    # ----
    psi_points     = np.array([x_psi,y_psi]).transpose()
    rho_points     = np.array([x_rho,y_rho]).transpose()
    u_points       = np.array([x_u,y_u]).transpose()
    v_points       = np.array([x_v,y_v]).transpose()
    fv_nodes       = np.array([NEST.xn[:,0],NEST.yn[:,0]]).transpose()
    fv_cells       = np.array([NEST.xc[:,0],NEST.yc[:,0]]).transpose()

    # Build the trees
    # ----
    print('  - Build KDTrees')
    psi_tree       = KDTree(psi_points)
    rho_tree       = KDTree(rho_points)
    u_tree         = KDTree(u_points)
    v_tree         = KDTree(v_points)

    # Determine the search range
    # ----
    dst            = np.sqrt((x_rho-x_psi[0])**2+(y_rho-y_psi[0])**2)
    ball_radius    = 1.3*dst[dst.argsort()[0]] # 1.3 is arbitrary. Just to make sure we only
                                               # find the 4 indices we need

    # Connect FVCOM points to psi, v and u inds
    # ----
    print('  - Searching for ROMS rho, u and v cells covering FVCOM nodes and cells')
    p, fv2psi      = psi_tree.query(fv_nodes) # psi is at the centre of rho cells
    p, fv2u_centre = v_tree.query(fv_cells)   # v is a the centre of u cells
    p, fv2v_centre = u_tree.query(fv_cells)   # u is at the centre of v cells

    # Find the nearest 4 by using KDTree query balls
    # ----
    rho_inds       = rho_tree.query_ball_point(psi_points[fv2psi], r = ball_radius)

    # v points are in the centre of "u-cells"
    u_inds         = u_tree.query_ball_point(v_points[fv2u_centre], r = ball_radius)

    # u points are in the centre of "v-cells"
    v_inds         = v_tree.query_ball_point(u_points[fv2v_centre], r = ball_radius)

    # Initiate the progress bar
    # --------------------------------------------------------------------------------------------
    widget         = ['- Calculating weight coefficients: ', pb.Percentage(), pb.Bar()]
    bar            = pb.ProgressBar(widgets=widget, maxval=len(NEST.xn)+2*len(NEST.xc))

    # Nodes (rho points)
    # --------------------------------------------------------------------------------------------
    bcnt = 0;

    bar.start()
    for k, (x, y) in enumerate(zip(NEST.xn, NEST.yn)):
        bar.update(bcnt)

        nearest_indices     = np.array(rho_inds[k])

        # land check
        # ----
        if any(Land_rho[nearest_indices]) and any(NEST.cid): # the last statement turns on/off the land-test switch
            class out: pass
            out.x           = x_rho; out.y = y_rho; out.mask = Land_rho
            land_ind        = nearest_indices[Land_rho[nearest_indices]]
            land_exception(NEST, out, land_ind, [x,y], 'node')

        # store the indices
        N4.rho_index[k,:]   = nearest_indices

        # Determine the weight function
        N4.rho_coef[k,:]    = bilinear_coefficients(x_rho[nearest_indices],\
                                                    y_rho[nearest_indices], x, y)

        # Progressbar stuff
        bcnt                += 1

    # Elements (u and v)
    # ----
    # Only loop through this one if we actually need the cell data
    if any(NEST.cid):
        for k, (x, y) in enumerate(zip(NEST.xc, NEST.yc)):
            bar.update(bcnt)
            # u
            # -------------------------------------------------------------------
            nearest_indices     = np.array(u_inds[k])

            # land check
            # ----
            if any(Land_u[nearest_indices]) and any(NEST.cid):
                class out: pass
                out.x    = x_u; out.y = y_u; out.mask = Land_u
                land_ind = nearest_indices[Land_u[nearest_indices]]
                land_exception(NEST, out, land_ind, [x,y], 'cell')

            # store the indices
            N4.u_index[k,:]      = np.array(nearest_indices)

            # Determine the weight function
            N4.u_coef[k,:]       = bilinear_coefficients(x_u[nearest_indices],\
                                                         y_u[nearest_indices], x, y)
            bcnt +=1
            bar.update(bcnt)

            # v
            # -------------------------------------------------------------------
            nearest_indices      = np.array(v_inds[k])

            # land check
            # ----
            if any(Land_v[nearest_indices]) and any(NEST.cid):
                class out: pass
                out.x    = x_v; out.y = y_v; out.mask = Land_v
                land_ind = nearest_indices[Land_v[nearest_indices]]
                land_exception(NEST, out, land_ind, [x,y], 'cell')

            N4.v_coef[k,:]       = bilinear_coefficients(x_v[nearest_indices],\
                                                         y_v[nearest_indices], x, y)
            N4.v_index[k,:]      = np.array(nearest_indices)
            bcnt += 1

    bar.finish()

    # since we want to use these as indices
    # ----
    N4.rho_index     = N4.rho_index.astype(int)
    N4.u_index       = N4.u_index.astype(int)
    N4.v_index       = N4.v_index.astype(int)

    N4.NEST          = NEST
    N4.ROMS_grd      = ROMS_grd
    if any(NEST.cid):
        N4.ROMS_depth()
    else:
        N4.ROMS_node_depth()
    N4.ROMS_angle()

    return N4

def horizontal_interpolation(ROMS_out, N4):
    class HORZfield: pass
    zlen               = ROMS_out.salt.shape[-1]
    HORZfield.u        = np.sum(ROMS_out.u[N4.u_index,:]*np.repeat(N4.u_coef[:,:,np.newaxis], zlen, axis=2), axis=1)
    HORZfield.v        = np.sum(ROMS_out.v[N4.v_index,:]*np.repeat(N4.v_coef[:,:,np.newaxis], zlen, axis=2), axis=1)
    HORZfield.zeta     = np.sum(ROMS_out.zeta[N4.rho_index]*N4.rho_coef, axis=1)
    HORZfield.temp     = np.sum(ROMS_out.temp[N4.rho_index,:]*np.repeat(N4.rho_coef[:,:,np.newaxis], zlen, axis=2), axis=1)
    HORZfield.salt     = np.sum(ROMS_out.salt[N4.rho_index,:]*np.repeat(N4.rho_coef[:,:,np.newaxis], zlen, axis=2), axis=1)
    return HORZfield


def bilinear_coefficients(roms_x, roms_y, fvcom_x, fvcom_y):
    '''
    Given four points, returns interpolation coefficients
    See the formula at: http://en.wikipedia.org/wiki/Bilinear_interpolation
    '''
    # We may need to rotate the coordinate system to get a straight system
    # ----
    if len(roms_y[np.argwhere(roms_y == min(roms_y))])>1 and len(roms_x[np.argwhere(roms_x == min(roms_x))])>1:
        #We do not need to rotate, continue
        abc = 0 # because why not...

    else:
        # Get the corner to rotate around
        # ----
        first_corner       = np.where(roms_y == min(roms_y))

        # Get the angle to rotate
        # ----
        dist               = np.sqrt((roms_x-roms_x[first_corner])**2+(roms_x-roms_x[first_corner])**2)
        x_tmp1             = roms_x[dist.argsort()[1:3]]
        first_to_the_right = x_tmp1[np.where(x_tmp1 == max(x_tmp1))]
        second_corner      = np.where(roms_x == first_to_the_right)
        angle              = np.arctan2(roms_y[second_corner]-roms_y[first_corner], \
                                        roms_x[second_corner]-roms_x[first_corner])

        # Rotate around origo. (rotating the entire coordinate system should be better...)
        # ----
        x_tmp              = (roms_x-roms_x[first_corner]);  y_tmp = (roms_y-roms_y[first_corner])
        fx_t               = (fvcom_x-roms_x[first_corner]); fy_t  = (fvcom_y-roms_y[first_corner])

        # Rotate ROMS
        # ----
        roms_x             = x_tmp*np.cos(-angle) - y_tmp*np.sin(-angle)
        roms_y             = x_tmp*np.sin(-angle) + y_tmp*np.cos(-angle)

        # Rotate FVCOM
        # ----
        fvcom_x            = fx_t*np.cos(-angle) - fy_t*np.sin(-angle)
        fvcom_y            = fx_t*np.sin(-angle) + fy_t*np.cos(-angle)

    # Find the bilinear interpolation coefficients for a 4 cornered box
    #   The formula for 1d interpolation to a point x on [a,b] is f(x) = [(b-x)*f(a)+(x-a)*f(b)]/(b-a)
    #   Bilinear interpolation applies that formula once in the x direction and once in the y direction
    # ----

    # All of them are divided by the same coefficient
    bottom                 = (max(roms_x)-min(roms_x))*(max(roms_y)-min(roms_y))
    coef                   = []

    # Identify the lowest ones
    args                   = roms_y.argsort()
    lower                  = args[0:2]
    upper                  = args[2:]

    coef                   = np.zeros((4,))
    coef[lower[0]]         = abs((roms_x[lower[1]]-fvcom_x)*(roms_y[upper[0]]-fvcom_y))/bottom
    coef[lower[1]]         = abs((roms_x[lower[0]]-fvcom_x)*(roms_y[upper[0]]-fvcom_y))/bottom

    coef[upper[0]]         = abs((roms_x[upper[1]]-fvcom_x)*(roms_y[lower[0]]-fvcom_y))/bottom
    coef[upper[1]]         = abs((roms_x[upper[0]]-fvcom_x)*(roms_y[lower[0]]-fvcom_y))/bottom

    coef                   = coef/np.sum(coef)

    return coef

def get_ROMS_grid_indices(x_roms, y_roms, x, y):
    '''
    Basically just finds the grid cell you are in, and stores the four indices connected to it
    (just in a horribly inefficient way)
    '''
    # Check if KD trees combined with psi points can be used to speed up this process

    distance          = np.sqrt((x_roms - x)**2 + (y_roms - y)**2)
    indices_sorted_according_to_distance = distance.argsort()[0:6]

    # Check that you use the correct indices
    # -----------------------------------------------------------------------------------------------------
    # -> The three first are given (no more than one node from outside the gridcell can be closer than those in it)
    # ----
    three_first       = indices_sorted_according_to_distance[0:3]

    # -> The last one can be close to the grid walls, which makes it trickier to find the nearest
    # ----
    dist_last_ones    = [np.sum((x_roms[three_first]-x_roms[last])**2 + (y_roms[three_first]-y_roms[last])**2)\
                            for last in indices_sorted_according_to_distance[3:]]

    last_one          = np.where(np.array(dist_last_ones) == np.array(dist_last_ones).min())[0][0]
    indices_sorted_and_checked  = np.append(three_first, indices_sorted_according_to_distance[3:][last_one])

    return indices_sorted_and_checked

def land_exception(NEST, grid, inds, pos, kind):
    '''
    Tell the user to recreate the nestingzone, and show them where the problem is.
    '''
    plt.scatter(grid.x[grid.mask==False],grid.y[grid.mask==False], label = 'ROMS ocean points')
    plt.scatter(grid.x[inds], grid.y[inds], c='r', label = 'ROMS land at this node point')
    plt.scatter(pos[0], pos[1], c='m', label = 'This FVCOM '+kind)
    plt.triplot(NEST.xn[:,0], NEST.yn[:,0], NEST.nv, c = 'g', lw = 0.3)
    plt.axis('equal')
    plt.title('Your nestingzone cover ROMS land!')
    plt.legend()
    plt.show(block=False)
    raise LandError(f'Adjust your mesh to avoid FVCOM nesting {kind} on ROMS land')

# --------------------------------------------------------------------------------------
#                             Grid and interpolation objects
# --------------------------------------------------------------------------------------
class N4ROMS():
    '''
    Object with indices and coefficients for ROMS to FVCOM interpolation.
    All grid details needed for the nesting should be found here.
    '''
    def __init__(self, NEST, ROMS_grd):
        '''
        Initialize empty attributes
        '''
        self.rho_coef         = np.empty([len(NEST.xn),  4])
        self.rho_index        = np.empty([len(NEST.xn),  4])

        self.u_coef           = np.empty([len(NEST.xc),  4])
        self.u_index          = np.empty([len(NEST.xc),  4])

        self.v_coef           = np.empty([len(NEST.xc),  4])
        self.v_index          = np.empty([len(NEST.xc),  4])

    def ROMS_depth(self):
        '''
        Calculate the ROMS depth at the FVCOM node, u and v point
        '''
        self.fvcom_rho_dpt    = np.sum(self.ROMS_grd.h_rho[self.fv_rho_mask][self.rho_index] * self.rho_coef, axis=1)
        self.fvcom_u_dpt      = np.sum(self.ROMS_grd.h_u[self.fv_u_mask][self.u_index] *self.u_coef, axis=1)
        self.fvcom_v_dpt      = np.sum(self.ROMS_grd.h_v[self.fv_v_mask][self.v_index] *self.v_coef, axis=1)

    def ROMS_node_depth(self):
        '''
        Calculate the ROMS depth at the FVCOM nodes
        '''
        self.fvcom_rho_dpt    = np.sum(self.ROMS_grd.h_rho[self.fv_rho_mask][self.rho_index] * self.rho_coef, axis=1)

    def ROMS_angle(self):
        '''
        Calculate ROMS angle at FVCOM nodes.
        '''
        self.fvcom_angle    = np.sum(self.ROMS_grd.angle[self.fv_rho_mask][self.rho_index] * self.rho_coef, axis=1)

    def save(self, name="Nearest4"):
        '''
        Save object to file.
        '''
        pickle.dump(self, open( name + ".p", "wb" ) )

class NEST_grid():
    '''
    Object containing information about the FVCOM grid
    '''
    def __init__(self, path_to_nest, M, proj="+proj=utm +zone=33W, +north "+\
                 "+ellps=WGS84 +datum=WGS84 +units=m +no_defs"):
        """
        Reads ngrd.* file. Converts it to general format.
        """

        self.filepath = path_to_nest
        self.Proj     = Proj(proj)

        if self.filepath[-3:] == 'mat':
            self.add_grid_parameters_mat(['xn', 'yn', 'h', 'nv', 'fvangle', 'xc',
                                          'yc', 'R', 'nid', 'cid','oend1','oend2'])

            self.lonc, self.latc = self.Proj(self.xc, self.yc, inverse=True)
            self.lonn, self.latn = self.Proj(self.xn, self.yn, inverse=True)
            self.nv = self.nv

        elif self.filepath[-3:] == 'npy':
            self.add_grid_parameters_npy(['xn','yn','nv','xc','yc','lonn',
                                          'latn','lonc','latc','nid','cid','R'])

        # Add information from full fvcom grid (M), vertical coords and OBS-nodes
        # ----
        self.siglay  = M.siglay[:len(self.xn), :]
        self.siglev  = M.siglev[:len(self.xn), :]
        self.siglayz = M.siglayz[:len(self.xn), :]

        self.siglay_center = (
                              self.siglay[self.nv[:,0], :]
                            + self.siglay[self.nv[:,1], :]
                            + self.siglay[self.nv[:,2], :]
                              )/3

        self.siglev_center = (
                              self.siglev[self.nv[:,0], :]
                            + self.siglev[self.nv[:,1], :]
                            + self.siglev[self.nv[:,2], :]
                             )/3

        self.calcWeights(M)


    def add_grid_parameters_mat(self, names):
        '''
        Read grid attributes from mfile and add them to FVCOM_grid object
        '''
        grid_mfile = loadmat(self.filepath)

        if type(names) is str:
            names=[names]

        for name in names:
            setattr(self, name, grid_mfile['ngrd'][0,0][name])

        # Translate Matlab indexing to python
        self.nid = self.nid -1
        self.cid = self.cid-1
        self.nv  = self.nv-1

    def add_grid_parameters_npy(self, names):
        nest = np.load(self.filepath, allow_pickle=True)
        special_keys = ['nv','R']
        for key in names:
            if key in special_keys:
                setattr(self, key, nest.item()[key])
            else:
                setattr(self, key, nest.item()[key][:,None])

        # Parameters
        self.oend1 = nest.item()['oend1'] # Matlab legacy
        self.oend2 = nest.item()['oend2'] # Matlab legacy
        self.R     = [[nest.item()['R']]] # Matlab legacy

    def calcWeights(self, M, w1=2.5e-4, w2=2.5e-5):
        '''
        Calculates linear weights in the nesting zone from weight = w1 at the obc to
        w2 the inner end of the nesting zone. At the obc nodes, weights equals 1

        By default (matlab legacy):
        w1  = 2.5e-4
        w2  = 2.5e-5

        This routine differs from the matlab sibling since the matlab version
        didn't work well for grids with several obcs.

        The ROMS model will be weighted less near the land than elsewhere (except
        at the outermost obc-row)
        '''
        M.get_obc()

        # Find the max radius- and node distance vector
        # ----
        if self.oend1 == 1:
            for n in range(len(M.x_obc)):
                dist       = np.sqrt((M.x_obc[n]-M.x_obc[n][0])**2+(M.y_obc[n] - M.y_obc[n][0])**2)
                i          = np.where(dist>self.R[0][0])
                M.x_obc[n] = M.x_obc[n][i]
                M.y_obc[n] = M.y_obc[n][i]

        if self.oend2 == 1:
            for n in range(len(M.x_obc)):
                dist       = np.sqrt((M.x_obc[n]-M.x_obc[n][-1])**2+(M.y_obc[n]-M.y_obc[n][-1])**2)
                i          = np.where(dist>self.R[0][0])
                M.x_obc[n] = M.x_obc[n][i]
                M.y_obc[n] = M.y_obc[n][i]

        # Find the distances between the nesting zone and the obc
        # ----
        # 1. Gather the obc nodes in one vector
        xo = []; yo = []
        for n in range(len(M.x_obc)):
            xo.extend(M.x_obc[n])
            yo.extend(M.y_obc[n])

        R = []; d_node = []
        for n in range(len(self.xn)):
            d_node.append(np.min(np.sqrt((xo-self.xn[n])**2+(yo-self.yn[n])**2)))

        R = max(d_node)

        # Define the interpolation values
        # ----
        distance_range = [0,R]
        weight_range   = [w1,w2]

        # Do the same for the cell values
        # ----
        d_cell = []
        for n in range(len(self.xc)):
            d_cell.append(min(np.sqrt((xo-self.xc[n])**2+(yo-self.yc[n])**2)))

        # Estimate the weight coefficients
        # ==> Kan det hende at disse må være lik for vektor og skalar?
        # ----
        weight_node = np.interp(d_node, distance_range, weight_range)
        weight_cell = np.interp(d_cell, distance_range, weight_range)

        if np.argwhere(weight_node<0).size != 0:
            weight_node[np.where(weight_node)]=min(weight_range)

        if np.argwhere(weight_cell<0).size != 0:
            weight_cell[np.where(weight_cell)]=min(weight_range)

        # ======================================================================================
        # The weights are calculated, now we need to overwrite some of them to get a full row of
        # forced values
        # ======================================================================================
        # Force the weight at the open boundary to be 1
        # ----
        # 1. reload the full obc
        M.get_obc()
        for n in range(len(M.x_obc)):
            xo.extend(M.x_obc[n])
            yo.extend(M.y_obc[n])

        # 2. Find the nesting nodes on the boundary
        node_obc_in_nest = [];
        for x,y in zip(xo,yo):
            dst_node = np.sqrt((self.xn-x)**2+(self.yn-y)**2)
            node_obc_in_nest.append(np.where(dst_node==dst_node.min())[0][0])

        # 3. Find the cells connected to these nodes
        cell_obc_in_nest = []
        nv               = self.nv
        nest_nodes       = np.array([node_obc_in_nest[:-1],\
                                     node_obc_in_nest[1:]]).transpose()

        # If you want a qube as the outer row
        # ===========================================================================
        for i, nest_pair in enumerate(nest_nodes):
            cells = [ind for ind, corners in enumerate(nv) if nest_pair[0] in \
                     corners or nest_pair[1] in corners]
            cell_obc_in_nest.extend(cells)

        cell_obc_to_one = np.unique(cell_obc_in_nest)

        # 4. Get all of the nodes in this list (builds the nodes one row outward)
        node_obc_to_one = np.unique(nv[cell_obc_to_one].ravel())

        # 5. Finally force the weight at the outermost OBC-row
        weight_node[node_obc_to_one] = 1.0
        weight_cell[cell_obc_to_one] = 1.0

        # --> Store everything in the NEST object
        # ----
        self.weight_node = weight_node
        self.weight_cell = weight_cell
        self.obc_nodes   = node_obc_to_one
        self.obc_cells   = cell_obc_to_one

class ROMS_grid():
    '''
    Object containing grid information about ROMS coastal ocean  model grid.
    '''
    def __init__(self, pathToROMS):
        """
        Read grid coordinates from nc-files.
        """
        self.nc     = pathToROMS
        self.name   = pathToROMS.split('/')[-1].split('.')[0]

        # Open the ROMS file
        # ----
        ncdata      = Dataset(pathToROMS, 'r')

        # Get information about vertical coordinates from this file for norshelf!!!
        # --> Frank, you are messing up the scheme here...
        # ----
        ncvert_path = get_vertpath(pathToROMS)
        ncvert      = Dataset(ncvert_path, 'r')

        # temp, salt and zeta
        # ----
        self.lon_rho = ncdata.variables.get('lon_rho')[:]
        self.lat_rho = ncdata.variables.get('lat_rho')[:]
        self.h_rho   = ncdata.variables.get('h')[:]
        self.angle   = ncdata.variables.get('angle')[:]

        # Find z-level using variable sigma levels
        self.Cs_r    = ncvert.variables.get('Cs_r')[:]
        self.s_rho   = ncvert.variables.get('s_rho')[:]
        self.hc      = ncvert.variables.get('hc')[0]
        self.S       = np.tile(np.zeros((self.lon_rho.shape))[:,:,None], (1,1,len(self.Cs_r)))

        # - get the depth at each s-level
        for i, (s, c) in enumerate(zip(self.s_rho, self.Cs_r)):
            self.S[:, :, i] = (self.hc*s + self.h_rho[:, :]*c)/(self.hc+self.h_rho)

        zeta = np.mean(ncdata.variables['zeta'][:])
        # this zeta is a slight simplification (good to probably within cm accuracy)-
        # according to https://www.myroms.org/wiki/Vertical_S-coordinate#transform2,
        # we would have to use time-varying zeta, but that would require quite a bit of
        # overhaul of the dimensions, interpolation, etc.
        self.z_rho = zeta + (zeta + self.h_rho[:,:,None]) * self.S

        # u velocity
        # ----
        self.lon_u   = ncdata.variables.get('lon_u')[:]
        self.lat_u   = ncdata.variables.get('lat_u')[:]
        self.h_u     = (self.h_rho[:,1:]+self.h_rho[:,:-1])/2
        self.z_u     = (self.z_rho[:,1:,:]+self.z_rho[:,:-1,:])/2

        # v velocity
        # ----
        self.lon_v   = ncdata.variables.get('lon_v')[:]
        self.lat_v   = ncdata.variables.get('lat_v')[:]
        self.h_v     = (self.h_rho[1:,:]+self.h_rho[:-1,:])/2
        self.z_v = (self.z_rho[1:,:,:]+self.z_rho[:-1,:,:])/2

        ncvert.close()

        # ROMS fractional landmask (from pp file). Let "1" indicate ocean.
        # ----
        self.u_mask   = ((ncdata.variables.get('mask_u')[:]-1)*(-1)).astype(bool)
        self.v_mask   = ((ncdata.variables.get('mask_v')[:]-1)*(-1)).astype(bool)
        self.rho_mask = ((ncdata.variables.get('mask_rho')[:]-1)*(-1)).astype(bool)
        ncdata.close()

        # Project to UTM33 coordinates
        # ----
        UTM33W       = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
        self.x_rho, self.y_rho = UTM33W(self.lon_rho, self.lat_rho, inverse=False)
        self.x_u,   self.y_u   = UTM33W(self.lon_u,   self.lat_u,   inverse=False)
        self.x_v,   self.y_v   = UTM33W(self.lon_v,   self.lat_v,   inverse=False)

    def crop_rho(self, xlim, ylim):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_rho >= xlim[0], self.x_rho <= xlim[1])
        ind2 = np.logical_and(self.y_rho >= ylim[0], self.y_rho <= ylim[1])
        return np.logical_and(ind1, ind2)

    def crop_u(self, xlim, ylim):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_u >= xlim[0], self.x_u <= xlim[1])
        ind2 = np.logical_and(self.y_u >= ylim[0], self.y_u <= ylim[1])
        return np.logical_and(ind1, ind2)

    def crop_v(self, xlim, ylim):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_v >= xlim[0], self.x_v <= xlim[1])
        ind2 = np.logical_and(self.y_v >= ylim[0], self.y_v <= ylim[1])
        return np.logical_and(ind1, ind2)

def get_vertpath(pathToROMS):
    '''
    If the path does not necessarilly contain the stuff we need, we will crop it a bit
    '''
    if 'norshelf' in pathToROMS:
        verpath = 'https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_avg_an_20210531T00Z.nc'
    else:
        verpath = pathToROMS
    return verpath

# ---------------------------------------------------------------------------------
#                            Write the output file
# ---------------------------------------------------------------------------------

def create_nc_forcing_file(name, NEST):
    '''
    Creates empty nc file formatted to fit FVCOM open boundary ocean forcing
    '''
    nc = Dataset(name, 'w', format='NETCDF4')

    # Write global attributes
    # ----
    nc.title       = 'FVCOM Nesting File'
    nc.institution = 'Akvaplan-niva AS'
    nc.source      = 'FVCOM grid (unstructured) nesting file'
    nc.created     = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # Create dimensions
    # ----
    nc.createDimension('time', 0)
    nc.createDimension('node', len(NEST.xn))
    nc.createDimension('nele', len(NEST.xc))
    nc.createDimension('three', 3)
    nc.createDimension('siglay', len(NEST.siglay[0,:]))
    nc.createDimension('siglev', len(NEST.siglev[0,:]))

    # Create variables and variable attributes
    # ----------------------------------------------------------
    time               = nc.createVariable('time', 'single', ('time',))
    time.units         = 'days since 1858-11-17 00:00:00'
    time.format        = 'modified julian day (MJD)'
    time.time_zone     = 'UTC'

    Itime              = nc.createVariable('Itime', 'int32', ('time',))
    Itime.units        = 'days since 1858-11-17 00:00:00'
    Itime.format       = 'modified julian day (MJD)'
    Itime.time_zone    = 'UTC'

    Itime2             = nc.createVariable('Itime2', 'int32', ('time',))
    Itime2.units       = 'msec since 00:00:00'
    Itime2.time_zone   = 'UTC'

    # positions
    # ----
    # node
    lon                = nc.createVariable('lon', 'single', ('node',))
    lat                = nc.createVariable('lat', 'single', ('node',))
    x                  = nc.createVariable('x', 'single', ('node',))
    y                  = nc.createVariable('y', 'single', ('node',))
    h                  = nc.createVariable('h', 'single', ('node',))


    # center
    lonc               = nc.createVariable('lonc', 'single', ('nele',))
    latc               = nc.createVariable('latc', 'single', ('nele',))
    xc                 = nc.createVariable('xc', 'single', ('nele',))
    yc                 = nc.createVariable('yc', 'single', ('nele',))
    hc                 = nc.createVariable('h_center', 'single', ('nele',))

    # grid parameters
    # ----
    nv                 = nc.createVariable('nv', 'int32', ('three', 'nele',))

    # node
    lay                = nc.createVariable('siglay','single',('siglay','node',))
    lev                = nc.createVariable('siglev','single',('siglev','node',))

    # center
    lay_center         = nc.createVariable('siglay_center','single',('siglay','nele',))
    lev_center         = nc.createVariable('siglev_center','single',('siglev','nele',))

    # Weight coefficients (since we are nesting from ROMS)
    # ----
    wc                 = nc.createVariable('weight_cell','single',('time','nele',))
    wn                 = nc.createVariable('weight_node','single',('time','node',))

    # time dependent variables
    # ----
    zeta               = nc.createVariable('zeta', 'single', ('time', 'node',))
    ua                 = nc.createVariable('ua', 'single', ('time', 'nele',))
    va                 = nc.createVariable('va', 'single', ('time', 'nele',))
    u                  = nc.createVariable('u', 'single', ('time', 'siglay', 'nele',))
    v                  = nc.createVariable('v', 'single', ('time', 'siglay', 'nele',))
    temp               = nc.createVariable('temp', 'single', ('time', 'siglay', 'node',))
    salt               = nc.createVariable('salinity', 'single', ('time', 'siglay', 'node',))
    hyw                = nc.createVariable('hyw', 'single', ('time', 'siglev', 'node',))

    # dump the grid metrics
    # ----
    nc.variables['lat'][:]           = NEST.latn
    nc.variables['lon'][:]           = NEST.lonn
    nc.variables['latc'][:]          = NEST.latc
    nc.variables['lonc'][:]          = NEST.lonc
    nc.variables['x'][:]             = NEST.xn
    nc.variables['y'][:]             = NEST.yn
    nc.variables['xc'][:]            = NEST.xc
    nc.variables['yc'][:]            = NEST.yc
    nc.variables['h'][:]             = NEST.h
    nc.variables['h_center'][:]      = NEST.hc
    tris                             = NEST.nv+1
    nc.variables['nv'][:]            = tris.transpose()
    nc.variables['siglev'][:]        = NEST.siglev.transpose()
    nc.variables['siglay'][:]        = NEST.siglay.transpose()
    nc.variables['siglev_center'][:] = NEST.siglev_center.transpose()
    nc.variables['siglay_center'][:] = NEST.siglay_center.transpose()

    nc.close()

def roms2fvcom(outfile, time, path, index, N4, N4_local=None):
    out          = Dataset(outfile, 'r+')
    already_read = []
    counter      = 0

    angle = N4.fvcom_angle # Angle to rotate u, v
    angle = (angle[N4.NEST.nv[:, 0]] + angle[N4.NEST.nv[:, 1]] + angle[N4.NEST.nv[:, 2]]) / 3
    angle = angle

    for fvcom_time, path, index in zip(time, path, index):
        if path != already_read:
            print(' ')
            print('Reading data from: '+path)
            nc = Dataset(path,'r')
            already_read = path

        print('- '+str(netCDF4.num2date(fvcom_time, units='days since 1858-11-17 00:00:00')))

        # Get the data you need from ROMS
        # ----
        unavailable = True
        while unavailable:
            try:
                if N4_local is not None and 'norkyst_800m_his.nc4' in path: # I think that string must be unique
                    ROMS_out = get_roms_data(nc, index, N4_local)
                    break
                else:
                    ROMS_out = get_roms_data(nc, index, N4)
                    break
            except:
                print("\n --------------------\n The data is unavailable at the moment.\n"+\
                      "Let' wait a minute and try again.\n --------------------\n")
                time_mod.sleep(60)

        # Interpolate the data to FVCOM nodes and centroids horizontally and vertically
        # ----
        if N4_local is not None and 'norkyst_800m_his.nc4' in path:
            ROMS_horizontal = horizontal_interpolation(ROMS_out, N4_local)
            FVCOM_in = vertical_interpolation(ROMS_horizontal, N4_local)

        else:
            ROMS_horizontal = horizontal_interpolation(ROMS_out, N4)
            FVCOM_in = vertical_interpolation(ROMS_horizontal, N4)

        # Write the data to the output file
        out.variables['time'][counter]           = fvcom_time
        out.variables['Itime'][counter]          = np.floor(fvcom_time)
        out.variables['Itime2'][counter]         = np.round((fvcom_time - np.floor(fvcom_time)) * 60 * 60 * 1000, decimals = 0)*24
        out.variables['u'][counter, :, :]        = FVCOM_in.u*np.cos(angle) - FVCOM_in.v*np.sin(angle)
        out.variables['v'][counter, :, :]        = FVCOM_in.u*np.sin(angle) + FVCOM_in.v*np.cos(angle)
        out.variables['hyw'][counter, :, :]      = np.zeros((1, len(N4.NEST.siglev[:,0]), len(N4.NEST.siglev[0,:])))
        #out.variables['ua'][counter, :]          = FVCOM_in.ubar
        #out.variables['va'][counter, :]          = FVCOM_in.vbar
        out.variables['zeta'][counter, :]        = FVCOM_in.zeta
        out.variables['temp'][counter, :, :]     = FVCOM_in.temp
        out.variables['salinity'][counter, :, :] = FVCOM_in.salt

        # --> I guess these weights are time dependent to support nudging?
        out.variables['weight_node'][counter,:]  = N4.NEST.weight_node
        out.variables['weight_cell'][counter,:]  = N4.NEST.weight_cell

        # prepare for the next round
        counter += 1

    out.close()

def vertical_interpolation(ROMS_data, N4):
    '''Linear vertical interpolation of ROMS data to FVCOM-depths.'''
    class Data2FVCOM():
        pass

    #Data2FVCOM.ubar = ROMS_data.ubar
    #Data2FVCOM.vbar = ROMS_data.vbar
    Data2FVCOM.zeta = ROMS_data.zeta

    salt = np.flip(ROMS_data.salt, axis=1).T
    Data2FVCOM.salt = salt[N4.vi_ind1_rho, range(0, salt.shape[1])] * N4.vi_weigths1_rho + \
                      salt[N4.vi_ind2_rho, range(0, salt.shape[1])] * N4.vi_weigths2_rho


    temp = np.flip(ROMS_data.temp, axis=1).T
    Data2FVCOM.temp = temp[N4.vi_ind1_rho, range(0, temp.shape[1])] * N4.vi_weigths1_rho + \
                      temp[N4.vi_ind2_rho, range(0, temp.shape[1])] * N4.vi_weigths2_rho


    u = np.flip(ROMS_data.u, axis=1).T
    Data2FVCOM.u = u[N4.vi_ind1_u, range(0, u.shape[1])] * N4.vi_weigths1_u + \
                   u[N4.vi_ind2_u, range(0, u.shape[1])] * N4.vi_weigths2_u


    v = np.flip(ROMS_data.v, axis=1).T
    Data2FVCOM.v = v[N4.vi_ind1_v, range(0, v.shape[1])] * N4.vi_weigths1_v + \
                   v[N4.vi_ind2_v, range(0, v.shape[1])] * N4.vi_weigths2_v


    return Data2FVCOM

# ----------------------------------------------------------------------------------
# Force the FVCOM bathymetry in the nestingzone to be equal to that of ROMS, make a
# transition zone from pure BuildCase-calculated bathymetry (in the model interoior)
# to pure ROMS bathymetry in the nestingzone
# ----------------------------------------------------------------------------------
def nestingtopo(ROMS, M, NEST, R):
    '''
    Interpolate the nestingzone ROMS topography to the FVCOM grid, use that
    new depth information to overwrite
    ----------------------------------------
    - ROMS:    ROMS grid object
    - M:       FVCOM grid object
    - NEST:    NEST grid object
    - R:       Nest radius (3000 by default)

    The routine will use the depth at rho points.
    '''

    r1 = 1.1*R; r2 = 2.5*R

    # Find the indices of the OBC nodes in the mesh
    # ----
    M.get_obc()
    tmp_nodes  = M.obcnodes
    nest_nodes = []

    for i in range(len(tmp_nodes)):
        nest_nodes.extend(tmp_nodes[i])

    nest_nodes = np.array(nest_nodes)

    # Distance from the OBC
    # ----
    Rdist = []
    for xn, yn in zip(M.x, M.y):
        dst   = np.sqrt((M.x[nest_nodes]-xn)**2+(M.y[nest_nodes]-yn)**2)
        Rdist.append(min(dst))

    Rdist = np.array(Rdist)

    # Find the weight function
    # ----
    weight                = np.zeros(len(M.x))
    weight[np.where(Rdist<r1)[0]] = 1
    weight[np.where(Rdist>r1)[0]] = 0
    transition            = np.where(np.logical_and(Rdist>r1, Rdist<r2))[0]
    a                     = 1.0/(r1-r2)
    b                     = r2/(r2-r1)
    weight[transition]    = a*Rdist[transition]+b

    # Find interpolation coefficients to the nodes where h must be smoothed
    # ----
    nodes = []
    nodes.extend(np.where(Rdist<r1)[0])
    nodes.extend(transition)
    nodes = np.array(nodes)

    # Create a mesh object that can be read by nearest4
    # ----
    class Mextra: pass
    Mextra.xn             = M.x[nodes][:,None]
    Mextra.yn             = M.y[nodes][:,None]
    Mextra.xc             = M.xc[0:3][:,None] # just placeholders
    Mextra.yc             = M.yc[0:3][:,None] # just placeholders
    Mextra.cid            = []                # a hack to tell the routine to _not_ check if
                                              # FVCOM overlaps with ROMS land

    print('  - Finding interpolation coefficients for the smoothing zone')
    N4                    = nearest4(Mextra, ROMS)

    # Dump ROMS depth to FVCOM nodes
    # ----
    hroms                 = np.copy(M.h)
    interpolert           = np.sum(N4.ROMS_grd.h_rho[N4.fv_rho_mask][N4.rho_index]*N4.rho_coef, axis=1)
    hroms[nodes]          = interpolert[:]

    # Update the nodes according to their distance from the obc
    h_updated             = hroms*weight + M.h*(1-weight)

    # Dump the new data to a dummy object
    class dump: pass
    dump.lat = M.lat[:]
    dump.lon = M.lon[:]
    dump.x   = M.x[:]
    dump.y   = M.y[:]
    dump.h   = h_updated[:]

    # Write the smoothed bathymetry to a .dat file
    try:
        filename = './input/'+M.info['casename']+'_dep.dat'
        write_FVCOM_bath(dump, filename = filename)

    except:
        write_FVCOM_bath(dump)

    # Overwrite NEST and M topo fields
    M.h         = dump.h[:,None]
    M.hc        = (M.h[M.tri[:,0]] + M.h[M.tri[:,1]] + M.h[M.tri[:,2]])/3.0
    M.siglayz   = M.siglayz

    # Get M indices corresponding to the nest locations
    ind       = []
    for x, y in zip(NEST.xn[:,0], NEST.yn[:,0]):
        dst     = np.sqrt((M.x[:]-x)**2+(M.y[:]-y)**2)
        ind.append(np.where(dst==dst.min())[0][0])
    ind          = np.array(ind)

    # Store the smoothed bathymetry
    nv           = NEST.nv[:]
    NEST.h       = M.h[ind]
    NEST.hc      = (NEST.h[nv[:,0]] + NEST.h[nv[:,1]] + NEST.h[nv[:,2]])/3.0
    NEST.siglayz = NEST.h*NEST.siglay

    return M, NEST

def make_fileList_NorShelf(start_time, stop_time):
    dates = prepare_dates(start_time, stop_time)

    thredds = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_qck_fc_"
    #thredds = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_qck_an_"
    #thredds = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_his_fc_"
    #thredds = "https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_his_an_"


    time = np.empty(0)
    path = []
    index = []
    not_found = []

    for date in dates:
        Norshelf_file = thredds + "{0.year}{0.month:02}{0.day:02}".format(date) + "T00Z.nc"

        try:
            nc = Dataset(Norshelf_file, 'r')
            roms_time = nc.variables['ocean_time'][:]
            time = np.append(time, roms_time)
            path.extend([Norshelf_file]*len(roms_time))
            index.extend(list(range(0, len(roms_time))))
            nc.close()
            print(Norshelf_file.split('/')[-1])

        except OSError:
            print(Norshelf_file.split('/')[-1] + " - not found")
            not_found.append(Norshelf_file)

    # --------------------------------------------------------------------------------------------
    #     Remove overlap
    # --------------------------------------------------------------------------------------------
    time_no_overlap     = [time[-1]]
    path_no_overlap     = [path[-1]]
    index_no_overlap    = [index[-1]]

    for n in range(len(time)-1, 0, -1):
        if time[n-1] < time_no_overlap[0]:
            time_no_overlap.insert(0, time[n-1])
            path_no_overlap.insert(0, path[n-1])
            index_no_overlap.insert(0, index[n-1])


    nc = Dataset(path[0], 'r')
    roms_time_units = nc.variables['ocean_time'].units
    time_no_overlap = netCDF4.num2date(time_no_overlap, units=roms_time_units)
    time_no_overlap = netCDF4.date2num(time_no_overlap, units='days since 1858-11-17 00:00:00')
    nc.close()


    return np.array(time_no_overlap), path_no_overlap, index_no_overlap

def diagnose_time(days, msec):
    '''
    Find the missing timesteps.
    Returns full (corrected) time vector and missing indices.
    '''
    # Find the timestep in-between each index
    # ----
    dts = []
    for i in range(len(days)-1):
        dt_day  = days[i+1]-days[i]
        dt_msec = msec[i+1]-msec[i]
        dts.append(dt_day*24+dt_msec/(60*60*1000))

    # To avoid roundoff-stuff
    # ----
    dts     = np.round(dts, decimals=1)

    # Minimum timestep
    # ----
    dt      = min(dts)

    # Identify bad indices
    # ----
    bad_ind = np.where(dts>dt)[:]

    if sum(dts[bad_ind[:]])>0:
        # Give user some feedback
        # ----
        print('\n --> There are ' + str(sum(dts[bad_ind[:]])) + ' hours missing from these data')

    else:
        print('No missing dates in this forcing file')

class LandError(Exception): pass
class InputError(Exception): pass
