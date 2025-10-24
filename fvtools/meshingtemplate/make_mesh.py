# --------------------------------------------------------------------
# This script runs all the commands you need when creating a mesh
# --> The upper part of it runs the commands to smooth coastlines, while
#     the lower run commands to determine the grid resolution in regions.
# --------------------------------------------------------------------

import sys
import os


import gridding.domain as domain
import gridding.coast as coast
import gridding.ocean as ocean
import gridding.process_coastline as pc
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import pandas as pd
from argparse import ArgumentParser

# --------------------------------------------------------------------------
#             PT.1:          SMOOTH THE COASTLINE
# --------------------------------------------------------------------------
def load_polygons():
    pl = raw_input('Is your polygons file called "Polylines.map"? (y/[n]): ')
    if pl=='y':
        x, y, npol = coast.grabPolylines()
    else:
        pfile      = subprocess.check_output('read -e -p "Enter the name of the polygons file: ' +\
                                             '" var ; echo $var', shell=True).rstrip()
        x, y, npol = coast.grabPolylines(mapfile = pfile)

    coast.write_polygon(x,y,npol)

def make_ocean_mesh(res):
    """
    For cases where you want to make a mesh over open ocean
    """
    mapfile = subprocess.check_output('read -e -p "Enter the name of the open-boundary file: ' +\
                                      '" var ; echo $var', shell=True).rstrip()
    ocean.write_boundary(mapfile, res = res)

def smooth_coastline():    
    # Read the input files
    # -----
    os.system('gedit scales.txt &')
    raw_input("\nFill values to the scales.txt file. Hit Enter to continue")
    while True:
        try:
            coast_file = subprocess.check_output('read -e -p "\nEnter coastline resolution file: '+\
                                                 '" var ; echo $var', shell=True).rstrip()
            print('\nRead coastfile')
            mp_coast   = pc.read_poly_data(coast_file)

            print('\nRead subdivision polygons')
            mp_sub     = pc.read_poly_data('./appdata/Polygon.txt')
            break

        except:
            print(coast_file + ' does not exist')

    # Crop the coastline for efficienty
    # ----
    try:
        print('Try to crop the coast')
        mp_coast = pc.crop_coast(mp_sub, mp_coast)
        print('Success')

    except:
        print('Failure, the coast could not be cropped\n moving on.')

    # Smooth the coastline according to provided scales
    # ----
    while True:
        mps, mpp, mpl = pc.make_coast(mp_coast,mp_sub,'scales.txt','myout.txt')
        plt.show()
        c = raw_input('Is that good enough? y/[n] \n')
        if c=='y':
            print("open the 'myout.txt' file in sms")
            break

        raw_input('update the scales.txt file, hit enter when done.')

    print(' --------------------------------------------- ')
    print('  Edit the coastline in SMS before continuing ')
    print(' --------------------------------------------- ')

# -------------------------------------------------------------------------
#             PT.2:      DETERMINE THE GRID RESOLUTION
#
# --> Edit the coastline file in SMS before continuing from here
# -------------------------------------------------------------------------
def load_coastline():
    while True:
        try:
            kyst_file  = subprocess.check_output('read -e -p "Enter filepath to the coastline '+\
                                                 ' .map: " var ; echo $var', shell=True).rstrip()
            x, y, npol = coast.read_map(kyst_file)
            break

        except:
            print(kyst_file+' does not exist, or there is something wrong with the file. try again.')

    # Split the islands from the boundary coastline, store in seperate files
    # ----
    coast.separate_polygons(x,y,npol)

    # See if the user wants to use other polygons to divide the domain resolution (you often
    # want high resolution in smaller regions than the covered in the smoothing process).
    # ----
    c = raw_input('\nDo you want to use other domain polygons when\n'+
                  'you decide which resolution SMESHING shall\n'+
                  'create in the domain? y/[n]\n \n')
    if c == 'y':
        while True:
            try:
                polygon_file = subprocess.check_output('read -e -p "Enter the name (or path if in ' +\
                                                       'another directory) of the Polylines file: ' +\
                                                       '" var ; echo $var', shell=True).rstrip()
                x, y, npol = coast.grabPolylines(mapfile=polygon_file)
                break

            except:
                print(polygon_file + ' does not exist, try again')

        coast.write_polygon(x,y,npol)

    # Estimate the resolution needed along the coast
    # ----
    print('\nRunning coastal express')
    out = os.system('sh job_distances.sh')
    if out == 32512:
        raise OSError('\njob_distances.py is not in your working directory! '+\
                      'You will find one in fvtools/mesh-template')
    

    print('\nSeparate islands from the boundary')
    coast.find_boundary()
    out = os.system('gedit PolyParameters.txt &')
    if out == 32512:
        raise OSError('\nPolyParameters.py is not in your working directory! '+\
                      'You will find one in fvtools/mesh-template')

    out = os.system('gedit set_coastres.py &')
    if out == 32512:
        raise OSError('\nset_coastres.py is not in your working directory! '+\
                      'You will find one in fvtools/mesh-template')

    raw_input('\nWrite the PolyParameters.txt file, and thereafter\n' +
              'update set_coastres.py with the indices you found above\n' +
              '---------------------------\n'+
              'Hit enter when you are done\n'+
              '---------------------------\n')

    print('\nRunning set_coastres...')
    out = os.system('python set_coastres.py')
    if out == 32512:
        raise OSError('\nset_coastres.py is not in your working directory! '+\
                      'You will find one in fvtools/mesh-template')
    
    print('\nDetermine the resolution as function from the coast')
    os.system('gedit config.yml &')

def resfield(resg         = 50,
             rx1max       = 3.0, 
             min_depth    = 5.0,
             distance_from_point= 400.0,
             drelmax      = 0.1, 
             ncount       = 3000,
             boundaryfile = 'input/boundary.txt',
             islandfile   = 'input/islands.txt'):
    '''
    Create a resolution field (areas with pre-determined resolution)
    resg  = 100 (Should be a fine mesh, but for big meshes it is next to impossible...)

    --> needs cleanup!
        - Seperate pointres from topores
        - Let the routine figure out more of the parameters itself
    '''
    # Should we "compile" with topores active?
    # ----
    topores   = raw_input('\nDo you want to use topores? y/[n]\n')
    if topores is 'y':
        topofile  = subprocess.check_output('read -e -p "Enter filepath to the bottom bathymetry file: '+\
                                            '" var ; echo $var', shell=True).rstrip()
        topores = True
    else:
        topores = False

    # Should a pointres file be included?
    # ----
    pointres  = raw_input('\nShould a pointres file be part of this? y/[n]\n') 
    if pointres is 'y':
        pointfile  = subprocess.check_output('read -e -p "Enter filepath to the pointres file: '+\
                                             '" var ; echo $var', shell=True).rstrip()
        pointres = True
    else:
        pointres = False

    # Should the topo domain be cropped?
    # ----
    crop  = raw_input('\nShould the resolution field be confined to a smaller region? y/[n]\n') 
    if crop is 'y':
        cropdomain  = subprocess.check_output('read -e -p "Enter filepath to the crop resolution .map: '+\
                                              '" var ; echo $var', shell=True).rstrip()
        crop = True
    else:
        crop = False

    # Load the polygons we use to set PolyParameters regions
    while True:
        try:
            pfile      = subprocess.check_output('read -e -p "Enter filepath to the resolution polygons .map: '+\
                                                     '" var ; echo $var', shell=True).rstrip()
            x, y, npol = coast.grabPolylines(mapfile = pfile)
            npol       = npol.astype(int)
            break
        except:
            print('- Did not find '+cropdomain+' try again.')

    # Read some numbers from the PolyParameters file
    # ----
    par       = pd.read_csv('PolyParameters.txt',sep=';')
    max_res   = par['max_res'].max()
    min_res   = par['min_res'].min()

    # See if resg need to be updated
    if min_res < resg:
        resg = min_res

    # Let the routine know how the vertical resolution of the model will be
    if topores:
        sigma     = coast.sigma_tanh(35, 0.5, 2.5)

        # Create a structured grid containing bathymetry information
        print('Run domain.topores')
        if crop:
            nold   = ncount
            ncount = 1
            

        xg, yg, rest, res = domain.topores(resg   = resg, 
                                           sigma  = sigma, 
                                           rx1max = rx1max,
                                           min_depth = min_depth, 
                                           max_res = max_res, 
                                           min_res = min_res, 
                                           drelmax = drelmax, 
                                           ncount  = ncount,
                                           topofile = topofile,
                                           boundaryfile = boundaryfile)

        # Cut the resolution to fit the model
        for p in np.unique(npol):
            # Find points within each subdomain
            print('- find points inside polygon '+str(p))
            inside  = domain.inside_polygon(xg.ravel(), yg.ravel(), x[npol==p], y[npol==p])

            print('  - force topores to bounds')
            # Convert to 1d array
            res_1d = res.ravel()

            # Crop to polygon domain
            res_c  = res_1d[inside]

            # Cut and replace
            too_big = np.where(res_c>par['max_res'][p-1])[0]
            if too_big.size>0:
                res_c[too_big] = par['max_res'][p-1]
                
            too_small = np.where(res_c<par['min_res'][p-1])[0]
            if too_small.size>0:
                res_c[too_small] = par['min_res'][p-1]

            # Return to 1d arrray
            res_1d[inside]  = res_c

            # Overwrite original resolution
            res = res_1d.reshape(res.shape)

    # Create a structured grid around the places we want "pointres" to apply
    if pointres:
        print('- Run domain.pointres')
        xp, yp, resp       = domain.pointres(resg, 
                                             pointfile, 
                                             max_res, 
                                             distance_from_point, 
                                             boundaryfile = boundaryfile)

    # Smooth the resolution gradients to make the mesh functional
    if crop:
        ncount = nold
        print('- Smooth the topores/pointres domain')
        print('Crop the resolution field to your mini-domain')
        xg, yg, res = domain.crop_resfield(xg,
                                           yg,
                                           res, 
                                           max_res,
                                           drelmax,
                                           resg,
                                           ncount, 
                                           cropdomain)
            
    elif pointres and topores:
        res  = np.minimum(res, resp)
        res  = domain.smoothres(res, resg, drelmax, ncount)

    if not topores and pointres:
        res = resp

    # Save the so-far field
    # ----
    if topores:
        np.savez('topo_resolution', xg=xg, yg=yg, res=res, ress=res)

    # Prepare the data so that it can be read by Smeshing
    # ----
    print('- Prepare the resolution data')
    x, y, resolution      = domain.prepare_resolution_data(xg, 
                                                           yg, 
                                                           res, 
                                                           boundaryfile, 
                                                           islandfile,
                                                           pout = 10, 
                                                           nout = True)

    # Remove data that is not within the model domain
    # ----
    print('- Remove redundant data')
    xt, yt, distance, obc = coast.read_boundary(boundaryfile)
    inside     = domain.inside_polygon(x, y, xt, yt)
    x          = x[inside]
    y          = y[inside]
    resolution = resolution[inside]

    # Estimate number of nodes needed (due to the resolution field, that is...)
    area = np.square(resg) #area of gridcell
    npa  = area / np.square(resolution)
    print('- Number of nodes needed: ' + str(np.sum(npa)))

    print('- Write as a textfile')
    domain.write_resolution('input/resfield.txt', x, y, resolution)

def choose_resolution(strres = None):
    while True:
        try:
            number = input('Choose the number of meters away from the coast (ld) to try first:\nld = ')
            b      = number**2
            break
        except:
            print('\n --> your input must be a number, try again\n------\n')

    print('\n---------------------------------------------------------------------------------')
    print('That was everything we needed to do here. Update config.yml with the values:\n'+\
          'num_grid_points, rmax (max resolution in the domain), dev1, dev2, rfact, dfact and ld.\n'+\
          '\nKeep num_iterations greater than 100 to get good meshes.')
    print('-----------------------------------------------------------------------------------')
    if strres is not None:
        domain.distfunc(Ld=number, strres = strres)
    else:
        domain.distfunc(Ld=number)

# -----------------------------------------------------------------------------------------------------
#                                 Tools to run the script outside of ipython
# -----------------------------------------------------------------------------------------------------
def parse_command_line():
    '''
    Parse command line arguments
    '''
    parser = ArgumentParser(description='To create a mesh you will need to smooth the coastline, \
                            divide the model domain into chunks \
                            where you want different resolutions \
                            determine some parameters defining how the resolution \
                            varies as a function of distance from nearest coast.')
    
    parser.add_argument("-full", "-f",
                        help='run the entire routine (True by default, will be set false if l or p is defined)',
                        default=True, action='store_true')
    
    parser.add_argument("-load", "-l",
                        help='load a smoothed coastline, create input files to SMESHING',
                        default=False, action='store_true')

    parser.add_argument("-ocean", "-o",
                        help="create a mesh in opean ocean",
                        default=False, action='store_true')

    parser.add_argument("-set_parameters", "-p",
                        help='determine the distance-from-coast parameters',                          
                        default=False, action='store_true')

    parser.add_argument("-smooth", "-s",
                        help='smooth the coastline',                          
                        default=False, action='store_true')

    parser.add_argument("-bath", "-b",
                        help='Use the bathymetry to set desired resolution in the mesh interoir',
                        default=False, action='store_true')

    parser.add_argument("-res", "-r",
                        help='resolution number',
                        default=None, type = float)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_command_line()
    if args.load or args.set_parameters or args.smooth or args.bath or args.ocean:
        args.full = False

    if args.full:
        load_polygons()
        smooth_coastline()
        load_coastline()
        resfield()
        if args.res is not None:
            choose_resolution(strres=args.res)
        else:
            choose_resolution()

    if args.smooth:
        smooth_coastline()

    if args.bath:
        resfield()

    if args.load:
        load_coastline()

    if args.set_parameters:
        if args.res is not None:
            choose_resolution(strres=args.res)
        else:
            choose_resolution()

    if args.ocean:
        load_polygons()
        make_ocean_mesh(args.res)
