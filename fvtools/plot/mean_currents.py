# ------------------------------------------------------------------
# Midle strom over tidsperiode, plot som stromlinjer
# ------------------------------------------------------------------
# Strategi:
# Midle hver dag for seg, sett sammen til slutt og lag nytt middel
# --------
import sys
import os
import utm
import pandas as pd
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cmocean as cmo
from datetime import datetime, timedelta
from netCDF4 import Dataset
from fvcom_grd import FVCOM_grid
from fvcom_plot import geoplot

#                          Main loop
# ------------------------------------------------------------------
def main(start_time, stop_time, store, folders, name, res = 200):
    '''
    returns the grid and the velocities so that you can redraw the figure
    with a different background and/or a different streamline density
    '''
    # collect the files we need to read
    paths = fileList(start_time, stop_time, store, folders, name)

    # gather data from these paths
    u,v   = get_velocities(paths)

    # read grid informatiom
    grid  = get_grid(paths[0])

    # prepare for streamlines
    vels  = setup_triangulation(grid,u,v,res)
    
    # plot the streamlines
    plot_stream(grid, vels, density)
    
    return grid, vels

# 1. Find the dates you need
# -------------------------------------------------------------------
def fileList(start_time,stop_time,store,folders,name):
    '''
    Figure out which files to load pylag given timespan and create filelist
    - start_time: first day [yyyy-mm-dd]
    - stop_time : last day  [yyyy-mm-dd]
    - store     : Path to store directory, eg: /tos-project1/NS9067K/apn_backup/FVCOM/Havard/SkjerstadNL/
    - folders   : List of folders containing data, eg: ['output_01','output_02', ...]
    - name      : name of simulation (for example NorL3)
    '''
    
    start     = datetime(int(start_time.split('-')[0]), int(start_time.split('-')[1]), int(start_time.split('-')[2]))
    stop      = datetime(int(stop_time.split('-')[0]), int(stop_time.split('-')[1]), int(stop_time.split('-')[2]))
    dates     = pd.date_range(start, stop)

    print('Looking for the files that span '+start_time + ' to ' + stop_time)
    print(' ---- ')

    # concatenate to get full path
    data_directories = [store+folders[i] for i in range(len(folders))]

    fvtime = np.empty(0)
    path   = []
    index  = []

    # Go through data directories and identify relevant data files
    for directory in data_directories:
        print()
        print(directory)
       
        files = [elem for elem in os.listdir(directory) if os.path.isfile(os.path.join(directory,elem))]
        files = [elem for elem in files if ((name in elem) and (len(elem) == len(name) + 8))]
        
        files.sort()
         
        for file in files:
            nc     = Dataset(os.path.join(directory, file), 'r')
            t      = nc.variables['time'][:]
            now    = netCDF4.num2date(t[0],nc['time'].units)
            
            if now < start:
                print(' --> '+ file+ ' is earlier than the requested daterange')
                continue
            
            if now > stop:
                print(' --> '+ file + ' is later than the requested daterange')
                break

            if len(t)<24:
                print('--> '+file+' was too short to be included.')
                continue

            print(file)
            fvtime = np.append(fvtime,t[0])
            path.extend([os.path.join(directory, file)])

    # Sort according to time
    sorted_data = sorted(zip(fvtime, path), key=lambda x: x[0])
    fvtime      = [s[0] for s in sorted_data]
    path        = [s[1] for s in sorted_data]

    # Remove overlap
    fvtime_no_overlap = [fvtime[-1]]
    path_no_overlap   = [path[-1]]

    for n in range(len(fvtime)-1,0, -1):
        if fvtime[n-1] < fvtime_no_overlap[0]:
            fvtime_no_overlap.insert(0, fvtime[n-1])
            path_no_overlap.insert(0, path[n-1])

    return path_no_overlap

# 2. collect data and metrics
# ---------------------------------------------------------------------------
def get_velocities(paths):
    '''
    gets depth averaged, long term velocities, gets their average and returns
    a depth and time averaged current
    '''
    uda   = []
    vda   = []
    first = True
    for path in paths:
        print(path)
        d  = Dataset(path)
        ua = d['ua'][:].mean(axis=0)
        va = d['va'][:].mean(axis=0)
        if first:
            uda   = ua
            vda   = va
            first = False
        else:
            uda  += ua
            vda  += va
        d.close()

    ua = uda/len(paths)
    va = vda/len(paths)

    return ua, va

def get_salinities(paths):
    '''
    gets the salinities in each layer. Averages them over the selected timeperiod.
    The user has to define which sigmalayer to plot.
    '''
    salt   = []
    first = True
    for path in paths:
        print(path)
        d    = Dataset(path)
        tmp = d['salinity'][:].mean(axis=0)
        if first:
            salt  = tmp
            first = False

        else:
            salt += tmp
        d.close()

    salt = salt/len(paths)

    return salt

def get_grid(path):
    '''
    reads grid metrics from path and returns it.
    '''
    class grid: pass
    d         = Dataset(path)
    ctri      = get_ctri(path)
    
    grid.ctri = ctri
    grid.xc   = d['xc'][:]
    grid.yc   = d['yc'][:]
    grid.nv   = d['nv'][:].transpose()-1
    grid.h    = d['h'][:]
    grid.x    = d['x'][:]
    grid.y    = d['y'][:]

    return grid

# 3. prepare data for streamlineplot
# --------------------------------------------------------------------------
def setup_triangulation(grid,ua,va,res):
    '''
    plot streamlines
    - grid object
    - u and v velocities
    - res(olution) of the structured grid to be created
    '''

    class vels: pass
    xvec   = np.arange(min(grid.xc), max(grid.xc), res)
    yvec   = np.arange(min(grid.yc), max(grid.yc), res)
    vels.xv, vels.yv = np.meshgrid(xvec, yvec)
    
    # Create triangulation
    triang  = tri.Triangulation(grid.xc, grid.yc, grid.ctri)
    tmpu    = tri.LinearTriInterpolator(triang,ua)
    tmpv    = tri.LinearTriInterpolator(triang,va)

    # interpolate to grid
    vels.u  = tmpu(vels.xv, vels.yv)
    vels.v  = tmpv(vels.xv, vels.yv)

    # store old data in the same structure
    vels.ut = ua
    vels.vt = va

    # Return (gives more flexibility for plotting)
    return vels

# 4. plot streamlines
# ---------------------------------------------------------------------------
def plot_stream(grid, vels, density, geopath = None):
    '''
    plot the streamlines with a given density on top of a georeferenced
    image (if specified)
    '''
    if geopath is not None:
        geoplot(geopath)
    lvls = np.arange(0,21)
    plt.tricontourf(grid.xc, grid.yc, grid.ctri, np.sqrt(vels.ut**2+vels.vt**2)*100,\
                        levels = lvls, cmap = cmo.cm.speed, extend='max')
    plt.colorbar(label='cm/s')
    plt.streamplot(vels.xv, vels.yv, vels.u, vels.v, density = density)

# ---------------------------------------------------------------------------
#                               etcetera
# ---------------------------------------------------------------------------
def get_ctri(gridfile):
    try:
        ctri = np.load('cell_tri.npy') # if already created
    except:
        Mobj = FVCOM_grid(gridfile)
        ctri = Mobj.cell_tri()

    return ctri

def plot_field(grid, field, geopath = None):
    '''
    plot the streamlines with a given density on top of a georeferenced
    image (if specified)
    '''
    if geopath is not None:
        geoplot(geopath)
    lvls = np.linspace(30,34,100)
    plt.tricontourf(grid.x, grid.y, grid.nv, field, levels=lvls, \
                        cmap = cmo.cm.haline, extend = 'both')
    plt.colorbar(label='psu')
