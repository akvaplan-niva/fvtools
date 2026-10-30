#import os
from datetime import date,datetime,timedelta
import sys

#from IPython.core.debugger import Tracer
from netCDF4 import Dataset
#import netCDF4
import numpy as np
#import matplotlib.mlab as mp
#import time
#import math
import utm
import grid.fvcom_grd as fvg

def initialize_singlenode(rfile='restart.nc',varname='tracer_c',inival=1,latitude=58.02173,longitude=7.116):
    '''Initialize tracer as inval in single node and zero elsewhere'''
    posxy=utm.from_latlon(latitude,longitude,33)
    x=posxy[0]
    y=posxy[1]
    #Loading FVCOM grid
    grd=fvg.FVCOM_grid('M.mat')
    kb=grd.get_kb()
    #Finding nearest point
    ds2=np.square(grd.x-x)+np.square(grd.y-y)
    index_min=np.where(ds2==ds2.min())[0].astype(int)
    tracer=np.zeros([ds2.shape[0],kb-1])
    tracer[index_min,:]=inival
    #Load tracer data into restart file.
    restart=Dataset(rfile,'r+')
    restart.variables[varname][0,:,:]=np.transpose(tracer)
    restart.close()

def flux_singlenode(source=1e-4,latitude=67.96,longitude=15.45,starttime=datetime(2013,1,1,0,0,0),endtime=datetime(2014,1,1,0,0,0)):
    '''Make a flux file with a constant flux source in a single node'''
    posxy=utm.from_latlon(latitude,longitude,33)#OBS zone hardcoded, check if right
    x=posxy[0]
    y=posxy[1]
    #Loading FVCOM grid
    grd=fvg.FVCOM_grid('M.mat')
    #Finding nearest point
    ds2=np.square(grd.x-x)+np.square(grd.y-y)
    index_min=np.where(ds2==ds2.min())[0].astype(int)
    flux0=source
    time = [starttime,endtime]
    date0 = datetime(1858,11,17,0,0,0)
    time = [time[x].toordinal()-date0.toordinal() for x in range(len(time))]
    flux=np.zeros([ds2.shape[0],2])
    flux[index_min,:]=flux0

    #Write to netcdf-file input/flux.nc
    fout = Dataset('input/flux.nc','w',format='NETCDF4')
    timeN = fout.createDimension('time',None)
    nondeN = fout.createDimension('node',ds2.shape[0])
    inj = fout.createVariable('injection_flux_int','f4',('time','node'))
    inj.units='m3/s'
    times = fout.createVariable('time','f4',('time',))
    times.long_name = 'time'
    times.units = 'days since '+str(date0)
    times.format = 'modified julian day (MJD)'
    times.time_zone = 'UTC'
    #write to netcdf created
    times[:] = time[:]
    inj[:] = flux.transpose()
    fout.close()
    
def flux_singlenode_5(source=1e-4,latitude=58.02,longitude=7.12,starttime=datetime(2013,1,1,0,0,0),endtime=datetime(2014,1,1,0,0,0)):
    '''Make a flux file dividing the total flux input into normal distributed flux over 5 tracers'''
    posxy=utm.from_latlon(latitude,longitude,33)#OBS zone hardcoded, check if right
    x=posxy[0]
    y=posxy[1]
    #Loading FVCOM grid
    grd=fvg.FVCOM_grid('M.mat')
    #Finding nearest point
    ds2=np.square(grd.x-x)+np.square(grd.y-y)
    index_min=np.where(ds2==ds2.min())[0].astype(int)
    flux0=source #[kg s^-1 m^-2] assuming source is already divided by control volume area
    time = [starttime,endtime]
    date0 = datetime(1858,11,17,0,0,0)
    time = [time[x].toordinal()-date0.toordinal() for x in range(len(time))]

    flux01=np.zeros([ds2.shape[0],2])
    flux02=np.zeros([ds2.shape[0],2])
    flux03=np.zeros([ds2.shape[0],2])
    flux04=np.zeros([ds2.shape[0],2])
    flux05=np.zeros([ds2.shape[0],2])

    flux01[index_min,:]=0.075*flux0
    flux02[index_min,:]=0.25*flux0
    flux03[index_min,:]=0.35*flux0
    flux04[index_min,:]=0.25*flux0
    flux05[index_min,:]=0.075*flux0

    #Write to netcdf-file input/flux.nc
    fout = Dataset('input/flux.nc','w',format='NETCDF4')
    timeN = fout.createDimension('time',None)
    nondeN = fout.createDimension('node',grd.x.shape[0])
    inj1 = fout.createVariable('injection1_flux_int','f4',('time','node'))
    inj1.units='kg s-1 m-2'
    inj2 = fout.createVariable('injection2_flux_int','f4',('time','node'))
    inj2.units='kg s-1 m-2'
    inj3 = fout.createVariable('injection3_flux_int','f4',('time','node'))
    inj3.units='kg s-1 m-2'
    inj4 = fout.createVariable('injection4_flux_int','f4',('time','node'))
    inj4.units='kg s-1 m-2'
    inj5 = fout.createVariable('injection5_flux_int','f4',('time','node'))
    inj5.units='kg s-1 m-2'

    times = fout.createVariable('time','f4',('time',))
    times.long_name = 'time'
    times.units = 'days since '+str(date0)
    times.format = 'modified julian day (MJD)'
    times.time_zone = 'UTC'
    #write to netcdf created
    times[:] = time[:]
    inj1[:] = flux01.transpose()
    inj2[:] = flux02.transpose()
    inj3[:] = flux03.transpose()
    inj4[:] = flux04.transpose()
    inj5[:] = flux05.transpose()

    fout.close()
    

def flux_singlenode_5T(source=1e-4,latitude=58.02,longitude=7.12,starttime=datetime(2013,1,1,0,0,0),endtime=datetime(2014,1,1,0,0,0)):
    '''Same as flux_singlenode_5 but time dependent release with 10 1h-releases evenly spread during 14 first days'''

    posxy=utm.from_latlon(latitude,longitude,33)#OBS zone hardcoded, check if right
    x=posxy[0]
    y=posxy[1]
    #Loading FVCOM grid
    grd=fvg.FVCOM_grid('M.mat')
    #Finding nearest point
    ds2=np.square(grd.x-x)+np.square(grd.y-y)
    index_min=np.where(ds2==ds2.min())[0].astype(int)
    flux0=source #[kg s^-1 m^-2] assuming source is already divided by control volume area
    
    tdelta=endtime-starttime
    time = [starttime + timedelta(n)/24 for n in range(0,tdelta.days*24)]
    #start and stop time for hourly release
    timestart = [time[x] for x in np.arange(1,14*24,34)]
    timestop = [time[x] for x in np.arange(1,14*24,34)+1]
    #fluxtime for 1min after start and one min before start
    fluxtime1 = [timestart[x] + timedelta(0,60) for x in range(len(timestart))]
    fluxtime2 = [timestop[x] - timedelta(0,60) for x in range(len(timestop))]
    #no flux time one min before start and after stop for model to ramp up/down flux
    nofluxtime1 = [timestart[x] - timedelta(0,60) for x in range(len(timestart))]
    nofluxtime2 = [timestop[x] + timedelta(0,60) for x in range(len(timestop))]

    timeF = sorted([starttime] + nofluxtime1 + fluxtime1 + fluxtime2 + nofluxtime2 + [endtime])
    date0 = datetime(1858,11,17,0,0,0)
    #timeFn = [timeF[x].toordinal()-date0.toordinal() for x in range(len(timeF))] #Obs only days
    timeFn = [(timeF[x]-date0).days + (timeF[x]-date0).seconds/86400.0 for x in range(len(timeF))]
    #Find indices with flux in timeF
    indexT = dict((value,idx) for idx,value in enumerate(timeF))
    index_flux = [indexT[x] for x in sorted(fluxtime1+fluxtime2)]

    flux01=np.zeros([ds2.shape[0],len(timeF)])
    flux02=np.zeros([ds2.shape[0],len(timeF)])
    flux03=np.zeros([ds2.shape[0],len(timeF)])
    flux04=np.zeros([ds2.shape[0],len(timeF)])
    flux05=np.zeros([ds2.shape[0],len(timeF)])

    flux01[index_min,index_flux]=0.075*flux0
    flux02[index_min,index_flux]=0.25*flux0
    flux03[index_min,index_flux]=0.35*flux0
    flux04[index_min,index_flux]=0.25*flux0
    flux05[index_min,index_flux]=0.075*flux0

    #Write to netcdf-file input/flux.nc
    fout = Dataset('input/flux.nc','w',format='NETCDF4')
    timeN = fout.createDimension('time',None)
    nondeN = fout.createDimension('node',grd.x.shape[0])
    inj1 = fout.createVariable('injection1_flux_int','f4',('time','node'))
    inj1.units='kg s-1 m-2'
    inj2 = fout.createVariable('injection2_flux_int','f4',('time','node'))
    inj2.units='kg s-1 m-2'
    inj3 = fout.createVariable('injection3_flux_int','f4',('time','node'))
    inj3.units='kg s-1 m-2'
    inj4 = fout.createVariable('injection4_flux_int','f4',('time','node'))
    inj4.units='kg s-1 m-2'
    inj5 = fout.createVariable('injection5_flux_int','f4',('time','node'))
    inj5.units='kg s-1 m-2'

    times = fout.createVariable('time','f4',('time',))
    times.long_name = 'time'
    times.units = 'days since '+str(date0)
    times.format = 'modified julian day (MJD)'
    times.time_zone = 'UTC'
    #write to netcdf created
    times[:] = timeFn[:]
    inj1[:] = flux01.transpose()
    inj2[:] = flux02.transpose()
    inj3[:] = flux03.transpose()
    inj4[:] = flux04.transpose()
    inj5[:] = flux05.transpose()

    fout.close()
       

    
