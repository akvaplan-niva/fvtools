#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import datetime
import time
import matplotlib.mlab as mlab
from IPython.core.debugger import Tracer
import sys

from netCDF4 import Dataset
import netCDF4
import numpy as np
import matplotlib.mlab as mp
import time


import grid.fvcom_grd as fvcom_grd
import grid.tools as tools


def export2file(grd,
                variable='salinity',
                fileList="fileList.txt", 
                output_file="fvcom_out",
                interpolation_depths=[5],
                start_time=None, 
                stop_time=None,
                subset=None,
                longitude_limits=None,
                latitude_limits=None):
    '''Extract subset of FVCOM T and S data and export to netcdf file.'''
   

    print(start_time)
    print(stop_time)
    # Make interpolation matrices
    interpolation_depths = [-1*depth for depth in interpolation_depths]
    grd.make_interpolation_matrices_TS(interpolation_depths)

    # Load fileList
    fl = tools.Filelist(fileList, start_time=start_time, stop_time=stop_time)

    
    # Determine which grid points to use
    if subset is None:
        if longitude_limits is not None: 
            ind1 = mlab.find(grd.lon[:,None] >= longitude_limits[0])
            ind2 = mlab.find(grd.lon[:,None] <= longitude_limits[1])
            ind_lon = np.intersect1d(ind1, ind2) 
            
            ind3 = mlab.find(grd.lat[:,None] >= latitude_limits[0])
            ind4 = mlab.find(grd.lat[:,None] <= latitude_limits[1])
            ind_lat = np.intersect1d(ind3, ind4)

            subset = np.intersect1d(ind_lon, ind_lat)
        
        else:
            subset = np.array(range(0, len(grd.x)))
    
    else:
        subset = np.array(range(subset[0], subset[1] + 1))

    numberOfgridPoints = len(subset)

        

    # Read data from FVCOM result files, do interpolation and write to nc-file
    first_time=1
    already_read=""

    for file_name, index, fvtime, time, counter in zip(fl.path, fl.index, fl.time, fl.datetime, range(0,len(fl.time))):        
        if file_name != already_read:            
            d=Dataset(file_name,'r')
            print(time)
        
        already_read = file_name
       
        if first_time: # Initialize nc-files
            out = Dataset(output_file, 'w', format='NETCDF4')
            
            # Create dimensions
            time_dimT = out.createDimension('time', 0)
            space_dimT = out.createDimension('space', numberOfgridPoints)
            depth_dimT = out.createDimension('depth', len(interpolation_depths))



            # Create variables
            fvcom_timeT = out.createVariable('fvcom_time', 'f4',('time',))
            lonT = out.createVariable('lon', 'f4',('space',))
            latT = out.createVariable('lat', 'f4', ('space',))
            xT = out.createVariable('x', 'f4', ('space',))
            yT = out.createVariable('y', 'f4', ('space',))
            zT = out.createVariable('z', 'f4',('depth',))
            V = out.createVariable(variable, 'f4', ('space', 'depth', 'time',))
            
            
            # Write data to lon, lat, x, y and z
            lonT[:] = d.variables.get('lon')[subset]
            latT[:] = d.variables.get('lat')[subset]
            xT[:] = d.variables.get('x')[subset]
            yT[:] = d.variables.get('y')[subset]
            zT[:] = interpolation_depths

            
            first_time = 0

        # Read data from current time step
        VI = d.variables.get(variable)[index, :, :]
        VI = VI[:, subset].T
        fvtime = d.variables.get('time')[index]
        
        # Do interpolation
        V2file = np.empty([numberOfgridPoints, len(interpolation_depths)])
        for id, ind in zip(interpolation_depths, range(0, len(interpolation_depths))):
            I = getattr(grd, 'interpolation_matrix_TS_' + str(abs(int(id))) + '_m')[subset, :]
            V2file[:,ind] = np.sum(I * VI, axis = 1)

        V[:, :, counter] = V2file
        fvcom_timeT[counter] = fvtime

    out.close()
    d.close()


def parse_command_line():
    ''' Parse command line arguments'''

    parser = ArgumentParser(description = 'Extracts TS data from fvcom 3D simulation and exports to netcdf')
                         
    parser.add_argument("-grid_file", "-g", type=str,
                        help="Path to m-file with grid information",
                        default="M.mat")
 
    parser.add_argument("-variable", "-v",
                        help="Variable to extract",
                        default="salinity")
                       
    parser.add_argument("-fileList", "-f",
                         help="Name of file that that links points in time to files and indices",
                         default="fileList.txt")

    parser.add_argument("-output_file", "-o", 
                        help="Name of output nc-file (prefix)",
                        default="fvcom_out")

    parser.add_argument("-interpolation_depths", "-i",
                        help="Depth to which data are interpolated (format: 5,20)",
                        default= "5")
     
    parser.add_argument("-start_time", "-b",
                        help="First time step to include (YYYY-MM-DD-HH)")

    parser.add_argument("-stop_time", "-s",
                        help="Last time step to include (YYYY-MM-DD-HH)")

    
    parser.add_argument("-longitude_limits", "-l",
                        help="Longitude limits (minLon,maxLon)")

    parser.add_argument("-latitude_limits", "-n",
                         help="Latitude limits (minLat,maxLat)")

    args = parser.parse_args()
    
    grd = fvcom_grd.FVCOM_grid(args.grid_file)
    
    #longitude_limits = args.longitude_limits
    longitude_limits = args.longitude_limits.split(',')
    longitude_limits = [float(elem) for elem in longitude_limits] 
    
    #latitude_limits = args.latitude_limits
    latitude_limits = args.latitude_limits.split(',')
    latitude_limits = [float(elem) for elem in latitude_limits]

    interpolation_depths = args.interpolation_depths.split(',')
    interpolation_depths = [float(elem) for elem in interpolation_depths]

    return grd, args.variable, args.fileList, args.output_file, interpolation_depths, args.start_time, \
           args.stop_time, longitude_limits, latitude_limits 

def main():
    t1 = time.time()
    subset = None
    grd, variable, fileList, output_file, interpolation_depths, start_time, stop_time, longitude_limits, latitude_limits \
    = parse_command_line()

    export2file(grd, \
                variable=variable, \
                fileList=fileList, \
                output_file=output_file, \
                start_time=start_time, \
                stop_time=stop_time, \
                longitude_limits=longitude_limits,
                latitude_limits=latitude_limits,
                interpolation_depths=interpolation_depths)



    t2 = time.time()
    t = (t2 - t1) / 60.0

    print ("\nCompleted data extraction in " + str(t) + " minutes")


if __name__ == '__main__':
    
    main()






    


