#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import os
import numpy as np
import datetime
from netCDF4 import Dataset
import netCDF4
import matplotlib.mlab as mp
from scipy.io import savemat
from IPython.core.debugger import Tracer
from time import gmtime, strftime
import time
import sys


from grid.fvcom_grd import FVCOM_grid
from grid.tools import Filelist

def export2mat(grd, fileList, position_file, output_dir, start_time, stop_time, out_format='matlab'):
    '''Extract fvcom data (vertical profiles) from given positions and time period.'''
    # Load fileList
    fl = Filelist(fileList, start_time=start_time, stop_time=stop_time)

     # Read position file
    positionFile = open(position_file, 'r')
    station_name = []
    lon_st = np.empty(0)
    lat_st = np.empty(0)

    for line in positionFile:
        station_name.append(line.split()[0])
        lon_st = np.append(lon_st, float(line.split()[1]))
        lat_st = np.append(lat_st, float(line.split()[2]))

    positionFile.close()

    # Determine indices of TS grid points closest to each position
    pos_ind_TS = np.empty(0)
    pos_ind_uv = np.empty(0)
    for lon, lat, st in zip(lon_st, lat_st, station_name):
        distance_TS = np.square(grd.lon[:,None]  - lon) + np.square(grd.lat[:,None] - lat) 
        pos_ind_TS = np.append(pos_ind_TS, np.where(distance_TS == distance_TS.min())[0][0])

        distance_uv = np.square(grd.lonc[:,None]  - lon) + np.square(grd.latc[:,None]  - lat)
        pos_ind_uv = np.append(pos_ind_uv, np.where(distance_uv == distance_uv.min())[0][0])

    pos_ind_TS = pos_ind_TS.astype(int)
    pos_ind_uv = pos_ind_uv.astype(int)

    #Tracer()()

    if out_format == 'matlab':
        write2mat(name=position_file.split('.')[0],
                  output_dir=output_dir,
                  station_name=station_name,
                  lon_st=lon_st,
                  lat_st=lat_st,
                  grd=grd,
                  fl=fl,
                  pos_ind_TS=pos_ind_TS,
                  pos_ind_uv=pos_ind_uv)

    elif out_format == 'netcdf':
         write2nc(name=position_file.split('.')[0],
                  output_dir=output_dir,
                  station_name=station_name,
                  lon_st=lon_st,
                  lat_st=lat_st,
                  grd=grd,
                  fl=fl,
                  pos_ind_TS=pos_ind_TS,
                  pos_ind_uv=pos_ind_uv)

       


def write2mat(name, output_dir, station_name, lon_st, lat_st, grd, fl, pos_ind_TS, pos_ind_uv):
    '''Extract data from fvcom results and dum to mat-file'''

    outdata = {} # Dictionary to store data for writing
    outdata['Name'] = name
    outdata['Stations'] = station_name
    outdata['lon_measurements']=lon_st
    outdata['lat_measurements']=lat_st
    outdata['lon_TS'] = grd.lon[pos_ind_TS]
    outdata['lat_TS'] = grd.lat[pos_ind_TS]
    outdata['lonc'] = grd.lonc[pos_ind_uv]
    outdata['latc'] = grd.latc[pos_ind_uv]
    outdata['x_TS'] = grd.x[pos_ind_TS]
    outdata['y_TS'] = grd.y[pos_ind_TS]
    outdata['x_uv'] = grd.xc[pos_ind_uv]
    outdata['y_uv'] = grd.yc[pos_ind_uv]
    outdata['z_TS'] = grd.siglayz[pos_ind_TS, :]
    outdata['z_uv'] = grd.siglayz_uv[pos_ind_uv, :]
    outdata['fvcom_time']=fl.time
    outdata['T']=np.empty([grd.siglayz.shape[1], len(fl.time), len(station_name)])
    outdata['S']=np.empty([grd.siglayz.shape[1], len(fl.time), len(station_name)])
    outdata['u']=np.empty([grd.siglayz.shape[1], len(fl.time), len(station_name)])
    outdata['v']=np.empty([grd.siglayz.shape[1], len(fl.time), len(station_name)])

    already_read=""
    
    for file_name, ft, ti, counter in zip(fl.path, fl.time, fl.index, range(0, len(fl.time))):
            
        if file_name != already_read: # Only opne nc-file the first time it is used
            d=Dataset(file_name,'r')
            print(file_name)

        already_read = file_name

        TI = d.variables.get('temp')[ti, :, :]
        outdata['T'][0:grd.siglayz.shape[1], counter, 0:len(station_name)] = TI[:, pos_ind_TS]
        SI = d.variables.get('salinity')[ti, :, :]
        outdata['S'][0:grd.siglayz.shape[1], counter, 0:len(station_name)] = SI[:, pos_ind_TS]
        
        UI = d.variables.get('u')[ti, :, :]
        outdata['u'][0:grd.siglayz.shape[1], counter, 0:len(station_name)] = UI[:, pos_ind_uv]
        VI = d.variables.get('v')[ti, :, :]
        outdata['v'][0:grd.siglayz.shape[1], counter, 0:len(station_name)] = VI[:, pos_ind_uv]
     
    savemat(os.path.join(output_dir, outdata['Name'] +'.mat'), outdata)




def write2nc(name, output_dir, station_name, lon_st, lat_st, grd, fl, pos_ind_TS, pos_ind_uv):
    '''Extract data from fvcom results and dum to nc-file'''
    
    nc = Dataset(name + '.nc', 'w', format='NETCDF4')

    # Write global attributes
    nc.title = 'FVCOM results'

    station_str = '''Vertical profiles of T, S, u and v from stations:\n'''
    for s in station_name:
        station_str = station_str + s + '\n'

    nc.info = station_str  
    nc.institution = 'Akvaplan-niva AS'
    nc.created = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # Create dimensions
    nc.createDimension('stations', len(lon_st))
    nc.createDimension('depth', grd.siglayz.shape[1])
    nc.createDimension('time', 0)
    
    # Create variables
    fvcom_time = nc.createVariable('time', 'f4', ('time',))
    fvcom_time.units = 'days since 1858-11-17 00:00:00'
    fvcom_time.format = 'modified julian day (MJD)'
    fvcom_time.time_zone = 'UTC'
    fvcom_time[:] = fl.time

    lat = nc.createVariable('latitude', 'f4', ('stations',))
    lat[:] = grd.lat[pos_ind_TS]
    lon = nc.createVariable('longitude', 'f4', ('stations',))
    lon[:] = grd.lon[pos_ind_TS]

    z = nc.createVariable('depth', 'f4', ('stations', 'depth',))
    z[:] = grd.siglayz[pos_ind_TS, :]


    T = nc.createVariable('temperatuer', 'f4', ('stations', 'depth', 'time',))
    S = nc.createVariable('salinity', 'f4', ('stations', 'depth', 'time',))
    u = nc.createVariable('u', 'f4', ('stations', 'depth', 'time',))
    v = nc.createVariable('v', 'f4', ('stations', 'depth', 'time',))

 
    already_read=""
    t1 = 0 
    for file_name, ft, ti, counter in zip(fl.path, fl.time, fl.index, range(0, len(fl.time))):
        
        if file_name != already_read: # Only opne nc-file the first time it is used
            d=Dataset(file_name,'r')
            print(file_name)
            t2 = time.time()
            t = (t2 - t1)
            t1 = time.time()
            print(t)


        already_read = file_name 
      
        T[0:len(station_name), 0:grd.siglayz.shape[1], counter] = np.transpose(d.variables.get('temp')[ti, :, pos_ind_TS])
        S[0:len(station_name), 0:grd.siglayz.shape[1], counter] = np.transpose(d.variables.get('salinity')[ti, :, pos_ind_TS])
        u[0:len(station_name), 0:grd.siglayz.shape[1], counter] = np.transpose(d.variables.get('u')[ti, :, pos_ind_uv])
        v[0:len(station_name), 0:grd.siglayz.shape[1], counter] = np.transpose(d.variables.get('v')[ti, :, pos_ind_uv])

    nc.close()




def parse_command_line():
    parser = ArgumentParser(description="Read FVCOM results from \
                                        individual positions and \
                                        save as mat-files")
    
    parser.add_argument('-grid_file', '-g', 
                        help='Path to grid file (M.mat)',
                        default='M.mat')
    
    parser.add_argument('-fileList', '-f', 
                        help='path to file list (fileList.txt)',
                        default='fileList.txt')
    
    parser.add_argument('-position_file', '-p', 
                        help='path to file with positions \
                        from where data will be extracted')
    
    parser.add_argument('-output_dir', '-o', 
                        help='directory to store output file files in',
                        default=os.getcwd())
    
    parser.add_argument('-start_time', '-s', 
                        help='first time step to include (YYYY-MM-DD-HH)')
    
    parser.add_argument('-end_time', '-e', 
                        help='last time step to include (YYYY-MM-DD-HH)')
    
    parser.add_argument('-out_format', '-t', 
                        help='matlab (default) or netcdf',
                        default='matlab')
    
    args=parser.parse_args()

    grd = FVCOM_grid(args.grid_file)
   

    return grd, args.fileList, args.position_file, args.output_dir, \
           args.start_time, args.end_time, args.out_format


def main():
    grd, fileList, position_file, output_dir, start_time, end_time, out_format  \
    = parse_command_line()

    export2mat(grd,
               fileList=fileList,
               position_file=position_file,
               output_dir=output_dir,
               start_time=start_time,
               stop_time=end_time,
               out_format=out_format)


if __name__ == '__main__':

    main()



