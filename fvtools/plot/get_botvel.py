from netCDF4 import Dataset
from datetime import datetime

import os
import netCDF4
import glob
import utm
import sys
import pandas as pd
import fvtools.grid.fvcom_grd as fvg
import numpy as np
import scipy.io as sio

from fvtools.grid.tools import num2date, date2num

def main(case, month, start_date, stop_date, out_folder = None):
    '''
    Plot bottom velocities calculated using a law-of-the-wall formula, returned as
    a 'velocities_{month}_{case}.npy' file.

    Input:
    ----
    case:       Casename for the returned .npy file
    month:      Name of month averaged over
    start_date: Start of the average (yyyy-mm-dd-hh)
    stop_date:  Stop of the average  (yyyy-mm-dd-hh)
    out_folder: Folders where the FVCOM output for this experiment is stored
    '''
    print('Finding the folders to check')
    folders     = bottom_folders(out_folder)
    all_nc      = list_ncfiles(folders)
    start, stop = prepare_dates(start_date, stop_date)

    dmin = netCDF4.Dataset(all_nc[0])
    dmax = netCDF4.Dataset(all_nc[-1])

    print('These results are available from:')
    print(f'Start: {num2date(dmin["time"][:].min())}')
    print(f'Stop:  {num2date(dmax["time"][:].min())}')

    # Create filelist
    # ----
    print('\nCompute filelist')
    Allfiles   = fileList(start, stop, all_nc)

    grd        = fvg.FVCOM_grid(Allfiles[0])
    ntime      = []
    nTimes     = []
    velsq      = np.zeros((len(Allfiles)*24, len(grd.xc)), dtype=float)

    print('\nCollect all data in one array')
    for i, f in enumerate(Allfiles[:-1]):
        print(f)
        D = Dataset(f,'r')
        velsq[i*24:(i*24)+24,:] = np.square(D.variables['u'][:,-1,:]) + np.square(D.variables['v'][:,-1,:])
        ntime.append(D.variables['time'][:])
        nTimes.append(D.variables['Times'][:,:])
        D.close()
    
    # Square root to get things back to normal
    # ----
    vel     = np.sqrt(velsq)
    velmax  = np.zeros((len(grd.xc),))
    velmean = np.zeros((len(grd.xc),))
    vel99   = np.zeros((len(grd.xc),))
    vel95   = np.zeros((len(grd.xc),))
    
    print('\nCompute statistics and save')
    for j in range(len(grd.xc)):
        velmax[j]  = vel[:,j].max()
        vel99[j]   = np.percentile(vel[:,j],99)
        vel95[j]   = np.percentile(vel[:,j],95)
        velmean[j] = np.mean(vel[:,j])

    velocities = {}
    velocities['vel']=vel
    velocities['velmax']=velmax
    velocities['vel99']=vel99
    velocities['vel95']=vel95
    velocities['velmean']=velmean
    np.save(f'velocities_{month}_{case}.npy',velocities)

def fileList(start, stop, all_nc):
    '''
    Take a timestep, couple it to a file
    ----
    - start:  Day to start
    - stop:   Day to stop
    - all_nc: nc-files to check
    '''
    List  = []

    for this in all_nc:
        d = Dataset(this)
        t = d['time'][:]
        indices = np.arange(len(t))
        if start is not None:
            if t.min()<start and t.max()<start:
                print(f' - {this} is before the date range')
                continue

        if stop is not None:
            if t.min()>stop:
                print(f' - {this} is after the date range')
                break
                
        print(f' - {this} at {num2date(t.min())}')
        List     = List + [this]


    return List

def list_ncfiles(dirs):
    '''
    returns list of all files in directories (or in one single directory)
    '''
    ncfiles = []
    for dr in dirs:
        stuff   = os.listdir(dr)
        sortert = sorted(stuff)
        tmp     = [dr+'/'+fil for fil in sortert if '.nc' in fil and 'restart' not in fil]
        ncfiles.extend(tmp)
    return ncfiles

def bottom_folders(folders):
    '''
    Returns the folders on the bottom of the pyramid (hence the name)
    mandatory: 
    folders   - parent folder(s) to cycle through
    '''
    # ----
    dirs = []

    if isinstance(folders,str):
        folders = [folders]

    for folder in folders:
        dirs.extend([x[0] for x in os.walk(folder)])
    
    # remove folders that are not at the top of the tree
    # ----
    leaf_branch = []
    if len(dirs)>1:
        for dr in dirs:
            if dr[-1]=='/':
                continue
            else:
                # This string is at the end of the branch, thus this is where the data is stored
                # ----
                leaf_branch.append(dr)
    else:
        leaf_branch = dirs
    return leaf_branch

def prepare_dates(start_time,stop_time):
    '''
    returns pandas array of dates needed
    '''
    print('\nPreparing daterange')
    start       = datetime(int(start_time.split('-')[0]), int(start_time.split('-')[1]),\
                           int(start_time.split('-')[2]))
    stop        = datetime(int(stop_time.split('-')[0]), int(stop_time.split('-')[1]),\
                           int(stop_time.split('-')[2]))

    return date2num([start])[0], date2num([stop])[0]

def parse_time_input(file_in, start, stop):
    """
    Translate time input to FVCOM time
    """
    d     = Dataset(file_in)
    units = d['time'].units

    if start is not None:
        start_num = start.split('-')
        start = date2num(datetime(int(start_num[0]), 
                                  int(start_num[1]), 
                                  int(start_num[2]), 
                                  int(start_num[3])),
                                  units = units)
    if stop is not None:
        stop_num = stop.split('-')
        stop = date2num(datetime(int(stop_num[0]), 
                                 int(stop_num[1]), 
                                 int(stop_num[2]), 
                                 int(stop_num[3])),
                                 units = units)
    return start, stop, units