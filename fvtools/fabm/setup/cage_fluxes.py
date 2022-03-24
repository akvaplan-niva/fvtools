import sys
import fvtools.fabm.setup.read_excel_positions as rep
import fvtools.fabm.setup.read_excel_carbon as rec
import fvtools.fabm.setup.read_txt_positions as rtp
import matplotlib.pyplot as plt
import numpy as np
import netCDF4

from fvtools.grid.fvcom_grd import FVCOM_grid
from datetime import datetime
from matplotlib.patches import Ellipse
from pyproj import Proj
from scipy.io import loadmat
from netCDF4 import Dataset

def main(savename     = None, 
         start_time   = None, 
         stop_time    = None, 
         bottom_type  = 'sand',
         bio_release_file = None, 
         grd_file     = None, 
         cage_file    = None,
         cage_omkrets = 140.0,
         plot_flux    = False,
         real         = False,
         gram         = True,
         z_cage       = 10.0,
         Sc_scale     = 1.0):
    """
    Write flux for sedimentation jobs to "savename".nc file.
    ------

    Parameters:
    ------
    - savename:         flux file name
    - start_time:       flux file start (yyyy-mm-dd-hh)
    - stop_time:        flux file stop (yyyy-mm-dd-hh)
    - bottom_type:      'sand', 'mud', 'cobble', 'sand and mud', 'sand and cobble'
    - bio_release_file: excel file with tracer names and release mass
                        -> The numbers provided are assumed to be the total release
                           in the interval [start_time : stop_time]
    - grd_file:         hydrodynamic output 'Storholmen_0001.nc'
    - cage_file:        excel file with positions
    - cage_omkrets:     circumference of a cage in the farm
    - plot_flux:        Plot figures of all flux fields. default: True
    - gram:             convert input from kg to gram. default: True
    - z_cage:           Depth where the faeces is released (default = 10 meters)
    """
    # Define standard values:
    # ----
    Feces_factor = 1.59*2.94
    Feed_factor  = 1.0/0.57

    # Load grid and control volume area
    # ----
    M = FVCOM_grid(grd_file)
    
    # Prepare the file duration
    # ----
    start = parse_time(start_time)
    stop  = parse_time(stop_time)
    flux_period_seconds = (stop - start).days * (24*60*60) 

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

    # Read carbon release (per second, adjusted for cages later)
    # ----
    if gram:
        kg_to_g  = 1000.0

    carbon   = rec.read_carbon(bio_release_file)

    # Adjust acording to those fancy tunable numbers =) (Magnus Drivdal knows what they are)
    # ---
    for i, waste_type in enumerate(carbon['names']):
        if any(x in waste_type for x in ['7','8']):
            if gram:
                carbon['release'][i] *= Sc_scale*Feed_factor*kg_to_g
            else:
                carbon['release'][i] *= Sc_scale*Feed_factor
        else:
            if gram:
                carbon['release'][i] *= Sc_scale*Feces_factor*kg_to_g
            else:
                carbon['release'][i] *= Sc_scale*Feces_factor

    # Create file and write stuff which is not location specific.
    # ----
    injections =  write_netcdf_flux(savename, start, stop, M, carbon, cages, flux_period_seconds, gram)

    # Write fabm.yaml and fabm_input.nml
    # ----
    print('\nWriting fabm.yaml and fabm_input.nml')
    tracer_names = []; all_injections = []; sigma_release = []
    for injection, siglay in zip(injections, cages['sig']):
        for single_case in injection:
            tracer_name = 'tracer'+single_case.split('injection')[1].split('_')[0]
            tracer_names.append(tracer_name)
            all_injections.append(single_case)
            sigma_release.append(siglay)

    # Get the settigs we want to write to the yaml file
    # ----
    instances, injections, sources, fabm_input = define_resusp_settings(tracer_names, all_injections, sigma_release,\
                                                                        bottom_type = bottom_type, real = real, gram = gram)
    write_yaml(instances, injections, sources)
    write_namelist(fabm_input, savename, nml_file = 'fabm_input.nml')

    # Visualize
    # ----
    print('\nDraw QC figures')
    if plot_flux:
        show_flux(M, savename)
    show_pens(M, cages)
    print('\nFin.')

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

# ----------------------------------------------------------------------------------------------
#                           Routines to write forcing files
# ----------------------------------------------------------------------------------------------
def write_netcdf_flux(savename, start, stop, M, carbon, cages, flux_period_seconds, gram):
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
    times[0] = netCDF4.date2num(start, units=times.units)
    times[1] = netCDF4.date2num(stop, units=times.units)

    # remove 0-flux
    # ----
    zero_ind = np.where(carbon['release']==0)
    if any(zero_ind):
        print('- Found empty variable')
        zero_ind = zero_ind[0][0]
        carbon['release'] = np.delete(carbon['release'],zero_ind)
        print(' - Removed: '+carbon['names'][zero_ind])
        carbon['names'].pop(zero_ind)

    # Create the injections name
    # ----
    injections = []
    for farm in range(len(cages['cage_nr'])):
        if farm == 0:
            injections = [[names + '_flux_int' for names in carbon['names']]]
        else:
            tmp = [names[:-1]+str(farm)+names[-1]+ '_flux_int' for names in carbon['names']]
            injections.append(tmp)

    # Write feces flux to file
    # ----
    for farm, cagenr in enumerate(cages['cage_nr']):
        # Cycle over farms
        for inj, rel in zip(injections[farm], carbon['release']): 
            carbon_per_s      = rel/flux_period_seconds
            carbon_per_s_cage = carbon_per_s/cagenr
            f       = outfile.createVariable(inj, 'f4', ('time', 'node'))
            if gram:
                f.units = 'g s-1 m-2'
            else:
                f.units = 'kg s-1 m-2'
            f[:]    = np.zeros((2, len(M.x)))
        
            # Cycle over cages in farms
            for i, cage in enumerate(cages['ell'][farm]):
                in_cage       = np.where(cage.contains_points(np.array([M.x, M.y]).T))[0]
                carbon_per_s_cage_node = carbon_per_s_cage/len(in_cage)
                for node in in_cage:
                    node_area  = M.art1[node]
                    f[:, node] = carbon_per_s_cage_node / node_area
           
    outfile.close()
    return injections

def define_resusp_settings(tracers, 
                           injections,
                           sigma_release,
                           bottom_type = None, 
                           bed_por = 0.6, 
                           real = False,
                           gram = True):
    """
    Sets the values you need to do a sedimentation run
    - tracers:       names of tracers
    - injections:    names of injections
    - sigma_release: sigma layer to release flux
    - bottom_type:   'mud', 'sand', 'cobble', 'sand and gravel', 'sand and cobble'
    - bed_por:       bed porosity (constant = 0.6 always?)
    - real:          True: Sink velocities according to law, False: Tracer 1-4 combined to one
    """
    # Sinking velocities from Bannister:
    # The real velocities originate from Bannister, the non-real are those we have weighted to
    # use fewer tracers per simulation
    #OBS: Bug if real=True in call to this function, you get 8 tracers, but if 8 tracers are read,
    #resusp_meth is set to 0 always...? (MAD 14/4-21)
    if real:
        w = [216.0, 648.0, 1080.0, 1728.0, 3240.0, 6480.0, 7603.2, 10368.0]
    else:
        w = [None, None, None, 864.0, 3240.0, 6480.0, 7603.2, 10368.0]
        
    # Always assume the zero state initialization type
    c     = 0.0
    c_bot = 0.0

    # Values depending on bed characteristics
    # Law et al 2016: Erodibility of aquaculture waste from different bottom substrates
    if bottom_type == 'mud':
        erate = 1.3 * 10**-6

    elif bottom_type == 'sand':
        erate = 3.5 * 10**-7

    elif bottom_type == 'cobble':
        erate = 4.5 * 10**-8

    elif bottom_type == 'sand and gravel':
        erate = 6.0 * 10**-7
        
    elif bottom_type == 'sand and cobble':
        erate = 5.8 * 10**-7

    else:
        raise NameError('Invalid argument:\nBottom type: "' + bottom_type + '" is not supported.'+\
                        '\nChoose between "mud", "sand", "sand and gravel" or "sand and cobble"')

    if gram:
        erate *= 1000

    # Create a dictionary containing the information we want to pass on to the yaml file
    instances = {}; injection_out = {}; source = {}; fabm_input = {}
    for tracer, injection, sigma in zip(tracers, injections, sigma_release):
        this_ind = int(tracer[-1])-1
        if this_ind == 6 or 7:
            resusp = 0
        else:
            resusp = 1

        tracer = tracer+':'
        instances[tracer] = {}
        instances[tracer]['model:'] = 'akvaplan/tracer_sed'
        instances[tracer]['parameters:'] = {}
        instances[tracer]['parameters:']['w:'] = w[this_ind]
        instances[tracer]['parameters:']['k:'] = 0.0
        instances[tracer]['parameters:']['temperature_dependence:'] = 0
        instances[tracer]['parameters:']['density:'] = 1460.
        instances[tracer]['parameters:']['specific_volume:'] = 1
        instances[tracer]['parameters:']['do_sed:']      = '.true.'
        instances[tracer]['parameters:']['resusp_meth:'] = resusp
        instances[tracer]['parameters:']['crt_shear:']   = 0.01
        instances[tracer]['parameters:']['erate:']       = erate
        instances[tracer]['parameters:']['bed_por:']     = bed_por
        
        instances[tracer]['initialization:'] = {}
        instances[tracer]['initialization:']['c:'] = c
        instances[tracer]['initialization:']['c_bot:'] = c_bot

        inj = injection.split('_')[0]+':'
        injection_out[inj] = {}
        injection_out[inj]['model:'] = 'akvaplan/plume_hardcode'
        injection_out[inj]['parameters:'] = {}
        injection_out[inj]['parameters:']['rho:'] = 500.0
        injection_out[inj]['parameters:']['plume_layer:'] = sigma

        tracer_source = tracer.split('_')[0][:-1]+'_source:'
        source[tracer_source] = {}
        source[tracer_source]['model:']  = 'interior_source'
        source[tracer_source]['coupling:'] = {}
        source[tracer_source]['coupling:']['source:'] = inj[:-1]+'/flux'
        source[tracer_source]['coupling:']['target:'] = tracer.split('_')[0][:-1]+'/c'
        
        fabm_input[inj] = inj[:-1]+'/flux_int'
        
    return instances, injection_out, source, fabm_input

def write_yaml(instances, injections, source):
    """
    Writes a fabm.yaml file with the fields you need
    
    instances:  Dictionary containing the stuff you need for the sedimentation model
    injections: Dictionary containing settings for injection flux
    source:     Dictionary setting the coupling to the flux file
    """
    # Create the file
    # ----
    f = open('fabm.yaml', 'w')
    f.write('instances:\n')

    # Write the instances first
    # ----
    for tracer in instances.keys():
        f.write('  '+tracer+'\n')
        for setting in instances[tracer].keys():
            if setting == 'model:':
                f.write('    '+setting+' '+str(instances[tracer][setting])+'\n')
            else:
                f.write('    '+setting+'\n')
                for parameter in instances[tracer][setting].keys():
                    f.write('      '+parameter+' '+str(instances[tracer][setting][parameter])+'\n')
        
    # Specify how the variables are coupled to the forcing
    # ----
    f.write('\n')
    for injection, tracer in zip(injections, source):
        f.write('  '+injection+'\n')
        # specify injection
        for setting in injections[injection]:
            if setting == 'model:':
                f.write('    '+setting+' '+str(injections[injection][setting])+'\n')

            else:
                f.write('    '+setting+'\n')
                for parameter in injections[injection][setting]:
                    f.write('      '+parameter+' '+str(injections[injection][setting][parameter])+'\n')
                    
        # specify tracer
        f.write('  '+tracer+'\n')
        for setting in source[tracer]:
            if setting == 'model:':
                f.write('    '+setting+' '+source[tracer][setting]+'\n')
            else:
                f.write('    '+setting+'\n')
                for parameter in source[tracer][setting]:
                    f.write('      '+parameter+' '+source[tracer][setting][parameter]+'\n')
    f.close()

def write_namelist(fabm_input, outfile, nml_file = 'fabm_input.nml'):
    """
    Writes a namelist file for fabm sediment modelling
    """
    f = open(nml_file, 'w')
    for key in fabm_input.keys():
        f.write('&NML_FABM_INPUT\n')
        f.write("  VARIABLE_NAME='"+fabm_input[key]+"'\n")
        f.write("  FILE='"+outfile+"'\n")
        f.write('/\n')
    f.close()

# -----------------------------------------------------------------------------------------
#                            Show that it all worked...
# -----------------------------------------------------------------------------------------
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
    keys = [key for key in keys if 'injection' in key]
    for key in keys:
        plt.figure(figsize=(9,9))
        M.plot_cvs(flux[key][:].max(axis=0))
        plt.axis('equal')
        plt.title(key)
        plt.colorbar()
