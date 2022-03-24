'''
Almost the same as JulianTimeElev in matlab, fv_tools, but with integrated writing of the forcing file.
'''
import pyTMD
import getpass
import cmocean as cmo
import os
import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
from netCDF4 import Dataset
from time import gmtime, strftime
from fvtools.plot.geoplot import geoplot
from fvtools.grid.fvcom_grd import FVCOM_grid

def main(M, 
         start_year, start_month, start_day, NumDays,
         min_int     = 20,
         model_dir   = '/data/Tides_hes/TPXO9_atlas/',
         verbose     = False,
         netcdf_name = None):
    '''
    Reads a mesh object and start/stop dates, returns tidal forcing

    Mandatory: 
    ----
    M:           Mesh object
    start_year:  Year of first reading
    start_month: Month of first reading
    start_day:   Day of first reading
    NumDays:     Number of days to go forward

    Optional:
    ----
    min_int:     Time stepsize (in minutes, 20 by default)
    model_dir:   Location of tidal model data (TPXO, AOTIM ...)
                 References the TPXO9-atlas-v5 stored on Stokes by default.
    netcdf_name: Name of julian elevation forcing file, {casename}_jul_el_obc.nc by default

    hes@akvaplan.niva.no
    '''
    # See if we support the requested file
    # ----
    if verbose: print(f'Julian time elevation forcing for {M.casename}:\n----')
    if model_dir == '/data/Tides_hes/TPXO9_atlas/':
        source = 'TPXO9-atlas-v4'
        if verbose: print(f'- Reading data from TPXO9-atlas-v5 located at: {model_dir}')
    else:
        raise InputError(f'This routine is not prepared to work with {model_dir}')

    if netcdf_name is None:
        netcdf_name = f'{M.casename}_jul_el_obc.nc'
        
    # Get time
    # ---
    if verbose: print('- Create time arrays')
    fvcom_time, tide_time = get_tide_time(NumDays, min_int, start_year, start_month, start_day)

    # Initialize the TMD reader
    # ----
    if verbose: print('- Preparing pyTMD to read results from OTIS formatted tidal model')
    model = pyTMD.model(directory = model_dir, format = 'OTIS')
    model.elevation(m = source)

    # Read model and dump data to FVCOM forcing file
    # ----
    if verbose: print('\nCompute tidally driven sea surface elevation at obc')
    amp, c = get_tide(model, M, fvcom_time, tide_time, netcdf_name, verbose)

    # Visualize amplitudes
    # ----
    visualize_amplitudes(M, amp, c, verbose = verbose)

def create_nc_forcing_file(name, M):
    '''
    Creates empty nc file formatted to fit FVCOM open boundary ocean forcing
    '''
    obc = []
    for i in range(len(M.obcnodes)):
        obc.extend(M.obcnodes[i])

    nc = Dataset(name, 'w', format = 'NETCDF4')

    # Write global attributes
    # ----
    nc.type        = 'FVCOM TIME SERIES ELEVATION FORCING FILE'
    nc.title       = 'FVCOM forcing using data from TPXO9-atlas-v5'
    nc.history     = 'File created using JulianTimeElev.py'
    nc.author      = getpass.getuser()
    nc.sourcedir   = os.getcwd()
    nc.institution = 'Akvaplan-niva AS'

    # Create dimensions
    # ----
    nc.createDimension('time', 0)
    nc.createDimension('nobc', len(obc))
    nc.createDimension('DateStrLen', 26)

    # Create variables and variable attributes
    # ----------------------------------------------------------
    time                = nc.createVariable('time', 'single', ('time',))
    time.units          = 'days since 1858-11-17 00:00:00'
    time.format         = 'modified julian day (MJD)'
    time.time_zone      = 'UTC'

    Itime               = nc.createVariable('Itime', 'int32', ('time',))
    Itime.units         = 'days since 1858-11-17 00:00:00'
    Itime.format        = 'modified julian day (MJD)'
    Itime.time_zone     = 'UTC'

    Itime2              = nc.createVariable('Itime2', 'int32', ('time',))
    Itime2.units        = 'msec since 00:00:00'
    Itime2.time_zone    = 'UTC'

    obc_nodes           = nc.createVariable('obc_nodes', 'int32', ('nobc',))
    obc_nodes.long_name = 'Open Boundary Node Number'
    obc_nodes.grid      = 'obc_grid'

    elevation           = nc.createVariable('elevation', 'single', ('time', 'nobc',))
    elevation.long_name = 'Open Boundary Elevation'
    elevation.units     = 'meters'

    # Dump a reference to the OBC nodes (in fortran notation) before continuing
    # ----
    nc.variables['obc_nodes'][:] = np.array(obc) + 1
    nc.close()

def handle_obc(M):
    '''
    Load obc, return all obcs as one continuous array
    '''
    M.get_obc()

    # Loop over all obcs to get all lat/lons, concatenate them to a cohesive list
    # ----
    lat = []; lon = []
    for i in range(M.lon_obc.shape[0]):
        lon.extend(M.lon_obc[i])
        lat.extend(M.lat_obc[i])
    lon = np.array(lon)
    lat = np.array(lat)

    return lat, lon

def get_tide(model, M, fvcom_time, tide_time, netcdf_name, verbose):
    '''
    Calculate tidally driven elevation on each OBC node for any time during the integration period
    '''
    # Interpolate data from tide model to FVCOM
    # ----
    #  - Get position of FVCOM obc nodes
    if verbose: print('- Reading obc')
    lat, lon = handle_obc(M)

    # - Write netcdf file
    if verbose: print(f'- Writing {netcdf_name} forcing file')
    create_nc_forcing_file(netcdf_name, M)

    # - Read and interpolate tide model (everything from here to the Dataset command is copied from this pyTMD example page:
    #   https://github.com/tsutterley/pyTMD/blob/main/notebooks/Plot%20Tide%20Forecasts.ipynb
    if verbose: print('- Extracting tidal constants')
    amp, ph, D, c = pyTMD.extract_tidal_constants(np.atleast_1d(lon), np.atleast_1d(lat), 
                                                  model.grid_file, model.model_file, model.projection,
                                                  TYPE = model.type, METHOD = 'spline', EXTRAPOLATE = True, GRID = model.format)

    # - Report back which constituents we use
    print(f'- Including {len(c)} constituents: {c}')

    # Phase and amplitudes as complex numbers at reference time
    # ----
    cph = -1j*ph*np.pi/180.0 
    hc  = amp*np.exp(cph)

    # Dump time to netCDF tile
    # ----
    Itime, Itime2  = get_Itime(fvcom_time)

    d = Dataset(netcdf_name, mode = 'r+')
    d['time'][:]   = fvcom_time
    d['Itime'][:]  = Itime
    d['Itime2'][:] = Itime2

    # Compute tidal range
    # ----
    print(f'- Computing tidally driven elevation at each obc node')
    widget = ['  - Progress: ', pb.Percentage(), ' ', pb.Bar()]
    bar = pb.ProgressBar(widgets=widget, maxval=hc.shape[0])
    bar.start()
    bar_count = 1
    for node in range(hc.shape[0]):
        bar.update(bar_count)
        major     = pyTMD.predict_tidal_ts(tide_time, hc[node,:][None, :], c, CORRECTIONS=model.format)
        minor     = pyTMD.infer_minor_corrections(tide_time, hc[node,:][None, :], c, CORRECTIONS=model.format)
        d['elevation'][:, node] = major + minor
        bar_count += 1
    bar.finish()
    
    # Done, tag with date and time and close file
    # ----
    d.created = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    d.close()
    return amp, c

def get_tide_time(NumDays, min_int, start_year, start_month, start_day):
    '''
    Handles time in fvcom-units and in TPXO units
    '''
    # Deal with time relative to 1.1.1992
    # ----
    num_minutes = NumDays*24*60
    minutes   = np.arange(0, num_minutes+min_int, min_int)
    tide_time = pyTMD.time.convert_calendar_dates(start_year, start_month, start_day, minute = minutes)

    # Convert tide time to FVCOM time
    # ----
    conversion = pyTMD.time.convert_calendar_dates(1992, 1, 1, epoch = (1858, 11, 17)) # from MJD to python time
    fvcom_time = tide_time + conversion

    return fvcom_time, tide_time

def get_Itime(fvcom_time):
    '''
    Returns Itime and Itime2 given time as input
    '''
    return fvcom_time.astype(int), np.rint(24*60*60*1000*(fvcom_time - fvcom_time.astype(int))).astype(int)

# Quality control:
# ----
def visualize_amplitudes(M, amp, c, wewant = ['m2','s2', 'n2', 'k1'], verbose = False):
    '''
    Georeferenced plot of amplitudes derived from the tidal inverse model
    '''
    if verbose: print('\nQC: Visualizing amplitudes of some tidal components')

    # Find extent
    xlim = [np.min(M.x), np.max(M.x)]
    ylim = [np.min(M.y), np.max(M.y)]
    dx   = np.diff(xlim)
    dy   = np.diff(ylim)

    # Define positions
    x = []; y = []
    for i in range(len(M.obcnodes)):
        x.extend(M.x_obc[i])
        y.extend(M.y_obc[i])

    # Number of nodes
    non  = 0
    for i in range(len(M.obcnodes)):
        non += len(M.obcnodes[i])

    # Bigger scope to fit the georeference
    xlim = [xlim[0]-0.15*dx, xlim[1]+0.15*dx] 
    ylim = [ylim[0]-0.15*dy, ylim[1]+0.15*dy] 

    # Get georeference
    gp   = geoplot(xlim, ylim)

    # Scatterplot
    fig, ax = plt.subplots(2, 2, figsize = [6.4, 4.8])
    ax      = ax.ravel()
    for i, const in enumerate(wewant):
        for ind, this in enumerate(c):
            if const == this:
                amplitudes = amp[:,ind]
                break
        ax[i].imshow(gp.img, extent = gp.extent) 
        cpl = ax[i].scatter(x, y, c = np.array(amplitudes), cmap = cmo.cm.matter)
        fig.colorbar(cpl, ax = ax[i], label = 'm')
        ax[i].set_xlim([np.min(M.x)-0.05*dx, np.max(M.x)+0.05*dx])
        ax[i].set_ylim([np.min(M.y)-0.05*dy, np.max(M.y)+0.05*dy])
        ax[i].set_title(f'Amplitude for {this.capitalize()}-constituent')
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    fig.suptitle('Amplitude of some constitutents, see if spatial patterns look reasonable.')
    plt.show(block = False)

def ModifyForcing(ncfile, gridfile):
    '''
    Every now and then, TPXO returns unlikely values near land, this routine lets you single out
    bad data and replace them with a nearby acceptable datapoint
    '''
    # Open forcing file in read+write mode
    d = Dataset(ncfile, 'r+')
    obcnodes = d['obc_nodes'][:]-1
    
    # Load grid
    M = FVCOM_grid(gridfile)
    x = M.x[obcnodes]
    y = M.y[obcnodes]

    # Create georeference
    # Find extent
    xlim = [np.min(x), np.max(x)]
    ylim = [np.min(y), np.max(y)]
    dx   = np.diff(xlim)
    dy   = np.diff(ylim)

    # Bigger scope to fit the georeference
    xlim = [xlim[0]-0.15*dx, xlim[1]+0.15*dx] 
    ylim = [ylim[0]-0.15*dy, ylim[1]+0.15*dy] 

    # Get georeference
    gp   = geoplot(xlim, ylim)

    # Load sea surface elevation amplitude
    amplitudes = d['elevation'][:].max(axis=0)
    
    # Find and tag nodes to overwrite
    plt.figure()
    plt.imshow(gp.img, extent = gp.extent) 
    cpl = plt.scatter(x, y, c = amplitudes, cmap = cmo.cm.matter)
    plt.colorbar(cpl, label = 'm')
    plt.xlim([np.min(M.x)-0.05*dx, np.max(M.x)+0.05*dx])
    plt.ylim([np.min(M.y)-0.05*dy, np.max(M.y)+0.05*dy])
    plt.title('Amplitude at each obc node, click at those you want to overwrite')
    plt.xticks([])
    plt.yticks([])
    plt.show(block = False)
    
    pout = np.array(plt.ginput(n = -1, timeout = -1))
    plt.scatter(pout[:,0], pout[:,1], marker = '+', c = 'r')
    plt.title('Click at the node you want to get replacement-data from')
    pin  = np.array(plt.ginput(n = -1, timeout = -1))
    plt.scatter(pin[0,0], pin[0,1], c = 'g')
    plt.title('Close the figure when you are done')
    plt.show()

    # Figure out the indexing
    outind = []
    for p in pout:
        dst = np.sqrt((x-p[0])**2+(y-p[1])**2)
        outind.append(np.where(dst == dst.min())[0][0])
    outind = np.array(outind)
    dst = np.sqrt((x-pin[0][0])**2 + (y-pin[0][1])**2)
    inind = np.where(dst == dst.min())[0][0]

    # Overwrite the actual file
    print('Overwriting the chosen nodes')
    timeseries = np.copy(d['elevation'][:, inind])[:,None]
    d['elevation'][:,outind] = timeseries
    
    # Copy the field from in-node
    print('Plot the new field')
    amplitudes = d['elevation'][:].max(axis=0)
    
    # Check if its better now?
    plt.figure()
    plt.imshow(gp.img, extent = gp.extent) 
    cpl = plt.scatter(x, y, c = amplitudes, cmap = cmo.cm.matter)
    plt.colorbar(cpl, label = 'm')
    plt.xlim([np.min(M.x)-0.05*dx, np.max(M.x)+0.05*dx])
    plt.ylim([np.min(M.y)-0.05*dy, np.max(M.y)+0.05*dy])
    plt.title('Amplitude at each obc node after overwriting.')
    plt.xticks([])
    plt.yticks([])
    plt.show(block = False)

    d.close()

class InputError(Exception): pass
