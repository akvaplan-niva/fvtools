import os
import datetime
import sys
from IPython.core.debugger import Tracer
from netCDF4 import Dataset
import netCDF4
import numpy as np
import matplotlib.mlab as mp
import time

import fvcom_grd as fvg

def make_nestfile(fileList="fileList.txt",output_file="fvcom_nest.nc",start_time=None,stop_time=None):
    #'''Extract subset of FVCOM T and S data and export to netcdf file.'''
    # Read file list
    fid=open(fileList,'r')
    fvcom_time=np.empty(0)
    path_to_files=[]
    time_index=[]
    for line in fid:
        fvcom_time=np.append(fvcom_time,float(line.split()[0]))
        path_to_files.append(line.split()[1])
        time_index.append(int(line.split()[2]))
    fid.close()

    # If start and stop time is given, remove entries in file list outside thie time period.
    if start_time is not None:
        t = netCDF4.num2date(fvcom_time, units = 'days since 1858-11-17 00:00:00')

        year = int(start_time.split('-')[0])
        month = int(start_time.split('-')[1])
        day = int(start_time.split('-')[2])
        hour = int(start_time.split('-')[3])
        t1 = datetime.datetime(year, month, day, hour)
        ind = mp.find(t >= t1)

        fvcom_time = [fvcom_time[i] for i in ind]
        path_to_files = [path_to_files[i] for i in ind]
        time_index = [time_index[i] for i in ind]

    if stop_time is not None:
        t = netCDF4.num2date(fvcom_time, units = 'days since 1858-11-17 00:00:00')

        year = int(stop_time.split('-')[0])
        month = int(stop_time.split('-')[1])
        day = int(stop_time.split('-')[2])
        hour = int(stop_time.split('-')[3])
        t1 = datetime.datetime(year,month,day,hour)
        ind = mp.find(t<=t1)

        fvcom_time = [fvcom_time[i] for i in ind]
        path_to_files = [path_to_files[i] for i in ind]
        time_index = [time_index[i] for i in ind]

    # Get indices of grid points
    ngrd=fvg.NEST_grid('/home/olean/fvcom/run/Lindesnes/f2f/ngrd.mat')
    ngrd.nid=np.squeeze(ngrd.nid)
    ngrd.cid=np.squeeze(ngrd.cid)

    # Read data from FVCOM result files and write to nesting nc-file
    first_time=1
    already_read=""

    for file_name,index,fvtime,counter in zip(path_to_files,time_index,fvcom_time,range(0,len(fvcom_time))):
        if file_name != already_read:
            d=Dataset(file_name,'r')
            print(file_name)

        already_read = file_name

        if first_time: # Initialize nc-files
            out_nest = Dataset(output_file, 'w', format='NETCDF4')
            kb=ngrd.get_kb()

            # Create dimensions
            timedim = out_nest.createDimension('time', 0)
            nodedim = out_nest.createDimension('node', len(ngrd.nid))
            celldim = out_nest.createDimension('nele', len(ngrd.cid))
            threedim = out_nest.createDimension('three', 3)
            levdim = out_nest.createDimension('siglev',kb)
            laydim = out_nest.createDimension('siglay',kb-1)

            # Create variables
            time = out_nest.createVariable('time', 'f4',('time',))
            time.units = "days since 1858-11-17 00:00:00"
            time.format = "modified julian day (MJD)"
            time.time_zone = "UTC"
            Itime = out_nest.createVariable('Itime', 'f4',('time',))
            Itime.units = "days since 1858-11-17 00:00:00"
            Itime.format = "modified julian day (MJD)"
            Itime.time_zone = "UTC"
            Itime2 = out_nest.createVariable('Itime2', 'f4',('time',))
            Itime2.units = "msec since 00:00:00"
            Itime2.time_zone = "UTC"
            lon = out_nest.createVariable('lon', 'f4',('node',))
            lat = out_nest.createVariable('lat', 'f4', ('node',))
            lonc = out_nest.createVariable('lonc', 'f4',('nele',))
            latc = out_nest.createVariable('latc', 'f4', ('nele',))
            x = out_nest.createVariable('x', 'f4', ('node',))
            y = out_nest.createVariable('y', 'f4', ('node',))
            xc = out_nest.createVariable('xc', 'f4', ('nele',))
            yc = out_nest.createVariable('yc', 'f4', ('nele',))
            zeta = out_nest.createVariable('zeta', 'f4', ('node', 'time',))
            ua = out_nest.createVariable('ua', 'f4', ('nele', 'time',))
            va = out_nest.createVariable('va', 'f4', ('nele', 'time',))
            u = out_nest.createVariable('u', 'f4', ('nele', 'siglay', 'time',))
            v = out_nest.createVariable('v', 'f4', ('nele', 'siglay', 'time',))
            temp = out_nest.createVariable('temp', 'f4', ('node', 'siglay', 'time',))
            salt = out_nest.createVariable('salinity', 'f4', ('node', 'siglay', 'time',))
            hyw = out_nest.createVariable('hyw', 'f4', ('node', 'siglev', 'time',))
            tracer = out_nest.createVariable('tracer_c', 'f4', ('node', 'siglay', 'time'))

            # Write data to lon, lat, x, y and z
            lon[:] = d.variables.get('lon')[:][ngrd.nid-1]
            lat[:] = d.variables.get('lat')[:][ngrd.nid-1]
            x[:] = d.variables.get('x')[:][ngrd.nid-1]
            y[:] = d.variables.get('y')[:][ngrd.nid-1]
            lonc[:] = d.variables.get('lonc')[:][ngrd.cid-1]
            latc[:] = d.variables.get('latc')[:][ngrd.cid-1]
            xc[:] = d.variables.get('xc')[:][ngrd.cid-1]
            yc[:] = d.variables.get('yc')[:][ngrd.cid-1]

            first_time = 0

      #  # Read data from current time step
        time[counter] = d.variables.get('time')[:][index]
        Itime[counter] = d.variables.get('Itime')[:][index]
        Itime2[counter] = d.variables.get('Itime2')[:][index]
        zeta[:,counter] = d.variables.get('zeta')[:][index,ngrd.nid-1]
        ua[:,counter] = d.variables.get('ua')[:][index, ngrd.cid-1]
        va[:,counter] = d.variables.get('va')[:][index, ngrd.cid-1]
        u[:,:,counter] = d.variables.get('u')[:][index, :, ngrd.cid-1]
        v[:,:,counter] = d.variables.get('v')[:][index, :, ngrd.cid-1]
        temp[:,:,counter] = d.variables.get('temp')[:][index, :, ngrd.nid-1]
        salt[:,:,counter] = d.variables.get('salinity')[:][index, :, ngrd.nid-1]
        tracer[:,:,counter] = d.variables.get('tracer_c')[:][index, :, ngrd.nid-1]

    out_nest.close()
    d.close()
