#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import datetime
import time
import matplotlib.mlab as mlab
import sys
from IPython.core.debugger import Tracer


from netCDF4 import Dataset
import netCDF4
import numpy as np
import matplotlib.mlab as mp
import time


import fvcom_grd
import tools

def export2file_uv(grd,
                fileList = "fileList.txt", 
                output_file = "velocity",
                depth_layers = [1],
                start_time = None, 
                stop_time = None,
                subset = None,
                longitude_limits = None,
                latitude_limits= None):
    '''Extract subset of FVCOM u and v data from given layers
    and export to netcdf'''

    # Load fileList
    fl = tools.Filelist(fileList, start_time=start_time, stop_time=stop_time)  
    
    # Determine which grid points to use
    if subset is None:
        if longitude_limits is not None:
            ind1 = mlab.find(grd.lonc >= longitude_limits[0])
            ind2 = mlab.find(grd.lonc <= longitude_limits[1])
            ind_lon = np.intersect1d(ind1, ind2)
        
            ind3 = mlab.find(grd.latc >= latitude_limits[0])
            ind4 = mlab.find(grd.latc <= latitude_limits[1])
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

    for file_name,index,fvtime,counter in zip(fl.path, fl.index, fl.time, range(0, len(fl.time))):
        if file_name != already_read:            
            d=Dataset(file_name,'r')
            print(file_name)
        
        already_read = file_name
       
        if first_time: # Initialize nc-files
           # Tracer()()
            out_u = Dataset(output_file + '_u.nc', 'w', format='NETCDF4')
            out_v = Dataset(output_file + '_v.nc', 'w', format='NETCDF4')
            
            # Create dimensions
            time_dim_u = out_u.createDimension('time', 0)
            space_dim_u = out_u.createDimension('space', numberOfgridPoints)
            depth_dim_u = out_u.createDimension('depth_layer', len(depth_layers))

            time_dim_v = out_v.createDimension('time', 0)
            space_dim_v = out_v.createDimension('space', numberOfgridPoints)
            depth_dim_v = out_v.createDimension('depth_layer', len(depth_layers))
            

            # Create variables
            fvcom_time_u = out_u.createVariable('fvcom_time', 'f8',('time',))
            x_u = out_u.createVariable('x', 'f4', ('space',))
            y_u = out_u.createVariable('y', 'f4', ('space',))
            z_u = out_u.createVariable('z', 'f4',('depth_layer',))
            lat_u = out_u.createVariable('lat', 'f4', ('space',))
            lon_u = out_u.createVariable('lon', 'f4', ('space',))
            u = out_u.createVariable('u', 'f4', ('space', 'depth_layer', 'time',))
            
            fvcom_time_v = out_v.createVariable('fvcom_time', 'f8',('time',))
            x_v = out_v.createVariable('x', 'f4', ('space',))
            y_v = out_v.createVariable('y', 'f4', ('space',))
            z_v = out_v.createVariable('z', 'f4',('depth_layer',))
            v = out_v.createVariable('v', 'f4', ('space', 'depth_layer', 'time',))

            # Write data to lon, lat, x, y and z
            x_u[:] = d.variables.get('xc')[subset]
            y_u[:] = d.variables.get('yc')[subset]
            lon_u[:] = grd.lonc[subset]
            lat_u[:] = grd.latc[subset]
            z_u[:] = np.array(depth_layers)

            x_v[:] = d.variables.get('xc')[subset]
            y_v[:] = d.variables.get('yc')[subset]
            z_v[:] = np.array(depth_layers)

            depth_ind = [int(elem)-1 for elem in depth_layers]

            first_time = 0
        
       
        # Read data from current time step
        ui = d.variables.get('u')[index, depth_ind, :]
        ui = ui[:, subset].T
        vi = d.variables.get('v')[index, depth_ind, :]
        vi = vi[:, subset].T
        fvtime = d.variables.get('time')[index]
        
        u[:, :, counter] = ui
        fvcom_time_u[counter] = fvtime
        v[:, :, counter] = vi
        fvcom_time_v[counter] = fvtime

    out_u.close()
    out_v.close()
    d.close()


def parse_command_line():
    ''' Parse command line arguments'''

    parser = ArgumentParser(description = 'Extracts TS data from fvcom 3D simulation and exports to netcdf')

    parser.add_argument("-grid_file", "-g", type=str,
                        help="Path to m-file with grid information",
                        default="M.mat")

    parser.add_argument("-fileList", "-f",
                        help="Name of file that that links points in time to files and indices",
                        default="fileList.txt")

    parser.add_argument("-output_file", "-o",
                        help="Name of output nc-file (prefix)",
                        default="fvcom_out")

    parser.add_argument("-depth_layers", "-d",
                        help="Depth layers which are extracted (format: 1,3,10)",
                        default= "1")

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

    longitude_limits = args.longitude_limits.split(',')
    longitude_limits = [float(elem) for elem in longitude_limits]

    latitude_limits = args.latitude_limits.split(',')
    latitude_limits = [float(elem) for elem in latitude_limits]

    depth_layers = args.depth_layers.split(',')

    return grd, args.fileList, args.output_file, depth_layers, args.start_time, \
           args.stop_time, longitude_limits, latitude_limits



def main():
    t1 = time.time()
    subset = None
    grd, fileList, output_file, depth_layers, start_time, stop_time, longitude_limits, latitude_limits \
    = parse_command_line()

    export2file_uv(grd, \
                fileList = fileList, \
                output_file = output_file, \
                start_time = start_time, \
                stop_time = stop_time, \
                longitude_limits = longitude_limits,
                latitude_limits = latitude_limits,
                depth_layers = depth_layers)


    t2 = time.time()
    t = (t2 - t1) / 60.0

    print ("\nCompleted data extraction in " + str(t) + " minutes")


if __name__ == '__main__':
    main()
    
