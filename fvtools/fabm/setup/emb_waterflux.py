import sys
from datetime import datetime,timedelta
from matplotlib.patches import Ellipse
import numpy as np
from pyproj import Proj
from scipy.io import loadmat
from netCDF4 import Dataset
import netCDF4
from fvtools.grid.fvcom_grd import FVCOM_grid
import fabm.setup.read_excel_positions as rep
import fabm.setup.read_excel_carbon as rec
import fabm.setup.read_txt_positions as rtp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main(savename     = None, 
         start_time   = None, 
         grd_file     = None,
         site_biomass = None, 
         cage_file    = None,
         cage_omkrets = 160.0,
         plot_flux    = False,
         z_cage       = 10.0):

    """
    Write Emamectin Benzoate (emb) flux from Aquaculture loc. to "savename".nc file. 
    Calculated from site biomass, half life in fish post-treatment. Extrapolation from 
    Day 0 - Day 7 (treatment end). Assuming 60 days iss sufficient (rest conc. ignored)
    Same princible as flux_setup (release in ellipse)
    ------

    Parameters:
    ------
    - savename:         flux file name
    - start_time:       flux file start (yyyy-mm-dd-hh)
    - grd_file:         hydrodynamic output 'Storholmen_0001.nc'
    - cage_file:        excel file with positions
    - cage_omkrets:     circumference of a cage in the farm
    - plot_flux:        Plot figures of all flux fields. default: True
    - z_cage:           Depth where the faeces is released (default = 10 meters)
    - site_biomass:     Total biomass of site (current month in question). Will be divided on no. of cages
    """

    #Define standard values:
    # ----
    halflife = 30.*24. #30 days half-life of emb in fish [h]
    lambd    = np.log(2)/halflife #[1/h]
    xt       = np.arange(0,60*24+1,1) #time series for exp decay in fish
    
    # Load grid and control volume area
    # ----
    M = FVCOM_grid(grd_file)
    
    # Prepare the file duration
    # ----
    start = parse_time(start_time)

    # Load the cages, associate a release with them.
    # ----
    cages    = read_cages(cage_file, radius = cage_omkrets/(2.0*np.pi))
    
    # Figure out which sigma layer we should release from
    # ----
    cages['sig'] = []
    for location in range(len(cages['x'])):
        sig, fz = depth2sigma(M, z_cage, cages['x'][location][0,:], cages['y'][location][0,:])
        _location_sig = int(np.mean(-sig+M.siglayz.shape[1]))
        cages['sig'].append(_location_sig)

    # Calculate emb released to water column (per second, adjusted for cages later)
    # ----
    #They medicate with 0.05 mg EMB / kg fish / day for 7 days
    TotEMB = 0.05*1000.*site_biomass*7. #Total EMB [ug = mg*1000]
    #Of which 2% goes to feed spill, 98% eaten by fish. Of the amount eaten by fish, 70% is excreted with feces
    #and 30% with mucus/urine to the water column.
    TotEMB_w = 0.98*0.3*TotEMB #Initial quantity in fish [ug]
    
    Infish_EMB = TotEMB_w*np.exp(-lambd*xt)              #Exp decay in fish starting from TotEMB_w
    Excr_EMB = np.abs(np.diff(Infish_EMB))               #Corresponding Excretion [ug per hour] h=0:60*24
    Excr_first7 = np.linspace(0,Excr_EMB[1],7*24-1)      #Assuming Linear lincrease in excretion h=0:7*24
    xtot = np.arange(0,(60+7)*24-1,1)                    #Time array ~67 days
    Excr_tot = np.concatenate((Excr_first7,Excr_EMB))    #Excretion ~67 days ug/h

    #Generate stoptime and flux-time to fit xtot which is hourly. Excr_tot as a f(flux_time)
    stop = start + timedelta(hours=len(xtot)-1)
    flux_period_seconds = (stop - start).days*(24*60*60) + (stop - start).seconds
    flux_time = [start+timedelta(hours=int(i)) for i in np.arange(len(xtot))]
    
    injections =  write_netcdf_flux(savename, start, stop, M, Excr_tot, cages, flux_time)

    # Visualize
    # ----
    print('\nDraw QC figures')
    if plot_flux:
        show_flux(M, savename)
    show_pens(M, cages)
    
    
    plt.figure(figsize=(9,9))
    plt.plot(xt,Infish_EMB)
    plt.ylabel('Total EMB in fish [ug]')
    plt.xlabel('time [h]')
    plt.title('Emamectin Benzoate decay in fish after treatment')
    
    fig=plt.figure(figsize=(9,9))
    plt.plot(flux_time,Excr_tot)
    plt.ylabel('EMB flux to water [ug/h]')
    plt.xlabel('time [h]')
    fig.autofmt_xdate()
    plt.title('Emamectin Benzoate excretion to water masses')
    
    plt.show()


def read_cages(cage_file, radius):
    """
    Read the type of file that Per-Arne tend to give us
    """
    # Initialize projection
    utm33 = Proj(proj = 'utm', zone = 33, ellps = 'WGS84', preserve_units = False)

    # Load the cages
    if cage_file[-3:] == 'txt':
        alternatives  = rtp.load_positions(cage_file)
    else:
        alternatives  = rep.load_positions(cage_file)

    # Convert to the format they expect
    cages = {}
    cages['x'] = []; cages['y'] = []
    cages['radius'] = []; cages['ell'] = []
    cages['cage_nr'] = []; cages_el = []
    for n, alt in enumerate(alternatives[0]):
        x, y = utm33(alt[0,:,1], alt[0,:,0])
        cages['x'].append(np.array([x]))
        cages['y'].append(np.array([y]))
        cages['radius'].append(np.repeat(radius, len(x)))
        cages['cage_nr'].append(len(x))

        # Attach an ellipsis (where semi minor = semi major) patch to each cage
        tmp = [Ellipse((x, y), width=2*radius, height=2*radius) 
                     for x, y, radius in zip(cages['x'][n][0], cages['y'][n][0], cages['radius'][n])]  
        cages['ell'].append(tmp)

    return cages

def parse_time(time_str):
    dtime = datetime(int(time_str.split('-')[0]),
                     int(time_str.split('-')[1]),
                     int(time_str.split('-')[2]),
                     int(time_str.split('-')[3]))

    return dtime
    

def depth2sigma(grd, z, x, y):
    '''Determine the sigma level at depth z at horizontal position x, y.'''

    ind_nearest = grd.find_nearest(x, y)
    z_fvcom = -grd.siglayz[ind_nearest,:]
    z_diff = np.abs(z_fvcom-z)
    z_ind = z_diff.argmin(axis=1)
    return z_ind, z_fvcom

def write_netcdf_flux(savename, start, stop, M, Excr_tot, cages, flux_time):
    """
    Write the netCDF forcing file
    """
    print('Dumping data to '+savename)
    outfile = Dataset(savename, 'w')
    outfile.createDimension('time', None)
    outfile.createDimension('node', len(M.x))

    times = outfile.createVariable('time', 'f4', ('time',))
    times.long_name = 'time'
    times.units = 'days since ' + str(datetime(1858, 11, 17, 0, 0, 0))
    times.format = 'modified julian day (MJD)'
    times.time_zone = 'UTC'
    times[:] = netCDF4.date2num(flux_time, units=times.units)
    #times[1] = netCDF4.date2num(stop, units=times.units)

    # Create the injections name
    # ----

    injections  = 'emb_flux_int'
    ugEMB_per_s = Excr_tot/3600.
    ugEMB_per_s_cage = ugEMB_per_s/cages['cage_nr']
    f       = outfile.createVariable(injections, 'f4', ('time', 'node'))
    f.units = 'ug s-1 m-2'
    f[:]    = np.zeros((len(flux_time), len(M.x)))

    for i, cage in enumerate(cages['ell'][0]):
        in_cage       = np.where(cage.contains_points(np.array([M.x, M.y]).T))[0]
        ugEMB_per_s_cage_node = ugEMB_per_s_cage/len(in_cage)
        for node in in_cage:
            node_area  = M.art1[node]
            f[:, node] = ugEMB_per_s_cage_node / node_area
    outfile.close()
    return injections

def show_pens(M, cages):
    """
    Scatter plots the grid and the locations
    """
    plt.figure(figsize=(9,9))
    plt.scatter(M.x, M.y, s = 1, c = 'k')
    for farm in range(len(cages['x'])):
        plt.scatter(cages['x'][farm], cages['y'][farm], s = 20, label = 'farm '+str(farm))

    plt.axis('equal')
    plt.legend()
    plt.title('Location of the farms and their pens in this flux file')
    plt.show()

def show_flux(M, fluxfile):
    """
    Plot the flux file fields to check that the routine got it right
    """
    flux = Dataset(fluxfile)
    keys = flux.variables.keys()
    keys = [key for key in keys if 'emb' in key]
    for key in keys:
        plt.figure(figsize=(9,9))
        M.plot_cvs(flux[key][:].max(axis=0))
        plt.axis('equal')
        plt.title(key)
        plt.colorbar(label='[ug/m2/s]')
