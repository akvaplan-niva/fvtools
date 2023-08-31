from netCDF4 import Dataset
from datetime import datetime

import os
import netCDF4
import glob
import utm
import sys
import pandas as pd
sys.path.append('/work/hdj002/fvcom_pytools/fvtools')
sys.path.append('/work/hdj002/fvcom_pytools/fvtools/fvtools/grid')
import fvtools.grid.fvcom_grd as fvg
import numpy as np
import scipy.io as sio

from numpy import matlib
import time

def main(case, month, start_date, stop_date, out_folder = None, depth = None):
    #case       = 'Sarnes'
    #month      = 'sep-nov'

    #start_date = '2013-09-01'
    #stop_date  = '2013-11-01'

    #folders = np.loadtxt(out_folder)
    #folders    = ['/lustre/work/hes/FVCOM_results/grieg.Sarnes/output',\
    #              '/lustre/work/hes/FVCOM_results/grieg.Sarnes/output_aut',\
    #              '/lustre/work/hes/FVCOM_results/grieg.Sarnes/output_aut2',\
    #              '/lustre/work/hes/FVCOM_results/grieg.Sarnes/output_aut3']

    #out_folder = ['/lustre/work/hes/FVCOM_results/grieg.Sarnes/']
    #depth = 1 - in meters above bottom, default is 1 m   

    if depth is None:
        depth = 1

    print('Finding the folders to check')
    folders    = bottom_folders(out_folder)

    # Create filelist
    # ----
    Allfiles   = make_fileList(start_date, stop_date, folders)

    print(Allfiles[0])
    grd        = fvg.FVCOM_grid(Allfiles[0])
    ntime      = []
    nTimes     = []
    velsq      = np.zeros((len(Allfiles)*24,grd.xc.shape[0]),dtype=float)
    velsq_u      = np.zeros((len(Allfiles)*24,grd.xc.shape[0]),dtype=float)
    velsq_v      = np.zeros((len(Allfiles)*24,grd.xc.shape[0]),dtype=float)
    velsq_2      = np.zeros((len(Allfiles)*24,grd.xc.shape[0]),dtype=float)


    try:
        print('try to load weight_ind_btm.npy from the working directory')
        bc        = np.load('weight_ind_botm.npy')
        height = bc[1,-1]
        
        print('check if depth is correct')
        if height == depth:
            print('sucess\n')

    except:
        print('\nFailure.')
        print('We need to which depth cell to use')
        bc = get_dc('weight_ind_botm', depth, Allfiles)

    in1 = bc[:,0]
    in1 = in1.astype(int)
    w1 = bc[:,2]
    in2 = bc[:,1]
    in2 = in2.astype(int)
    w2 = bc[:,3]
    uin = np.zeros((24,grd.siglay.shape[1],grd.xc.shape[0]),dtype=float)
    vin = np.zeros((24,grd.siglay.shape[1],grd.xc.shape[0]),dtype=float)
    ui1 = np.unique(in1)
    ui2 = np.unique(in2)

    start = time.time()
    for i in range(len(Allfiles)-1):
        print(Allfiles[i])
        D = Dataset(Allfiles[i],'r')
        v = np.array(D['v'][:,ui1,:])
        u = np.array(D['u'][:,ui1,:])
        for n in range(len(ui1)):
            print(n)
            result = np.where(in1 == ui1[n])
            ind = np.concatenate(np.array(result))
            v1 = v[:,n,ind]*w1[ind]
            u1 = u[:,n,ind]*w1[ind]
            vin[:,ui1[n],ind] = v1
            uin[:,ui1[n],ind] = u1         

        v = np.array(D['v'][:,ui2,:])
        u = np.array(D['u'][:,ui2,:])
        for n in range(len(ui2)):
            print(n)
            result = np.where(in2 == ui2[n])
            ind = np.concatenate(np.array(result))
            v2 = v[:,n,ind]*w2[ind]
            u2 = u[:,n,ind]*w2[ind]
            vin[:,ui2[n],ind] = v2
            uin[:,ui2[n],ind] = u2   

        vuse =  np.sum(vin,axis=1)
        uuse =  np.sum(uin,axis=1)
        velsq_2[i*24:(i*24)+24,:] = np.square(v[:,-1,:]) + np.square(u[:,-1,:])
        velsq[i*24:(i*24)+24,:] = np.square(vuse) + np.square(uuse)
        velsq_u[i*24:(i*24)+24,:] = uuse#np.square(uuse)
        velsq_v[i*24:(i*24)+24,:] = vuse#np.square(vuse) 
        ntime.append(D.variables['time'][:])
        nTimes.append(D.variables['Times'][:,:])
        D.close()
  
    end = time.time()
    print(end - start)
    
    # Square root to get things back to normal
    # ----
    vel     = np.sqrt(velsq)
    vel_u   = velsq_u #np.sqrt(velsq_u)
    vel_v   = velsq_v #np.sqrt(velsq_v)
    vel2     = np.sqrt(velsq_2)
    velmax  = np.zeros((grd.xc.shape[0],))
    velmax_2  = np.zeros((grd.xc.shape[0],))
    velmean = np.zeros((grd.xc.shape[0],))
    velmean_u = np.zeros((grd.xc.shape[0],))
    velmean_v = np.zeros((grd.xc.shape[0],))
    velmean_2 = np.zeros((grd.xc.shape[0],))
    vel99   = np.zeros((grd.xc.shape[0],))
    vel95   = np.zeros((grd.xc.shape[0],))
    vel99_2   = np.zeros((grd.xc.shape[0],))
    vel95_2   = np.zeros((grd.xc.shape[0],))

    for j in range(len(velmax)):
        velmax[j]  = vel[:,j].max()
        vel99[j]   = np.percentile(vel[:,j],99)
        vel95[j]   = np.percentile(vel[:,j],95)
        velmean[j] = np.mean(vel[:,j])
        velmean_u[j] = np.mean(vel_u[:,j])
        velmean_v[j] = np.mean(vel_v[:,j])
        velmean_2[j] = np.mean(vel2[:,j])
        velmax_2[j]  = vel2[:,j].max()
        vel99_2[j]   = np.percentile(vel2[:,j],99)
        vel95_2[j]   = np.percentile(vel2[:,j],95)

    velocities = {}
#    velocities['vel']=vel
    velocities['velmean_u']=velmean_u
    velocities['velmean_v']=velmean_v
    velocities['velmax']=velmax
    velocities['velmax_2']=velmax_2
    velocities['vel99']=vel99
    velocities['vel99_2']=vel99_2
    velocities['vel95']=vel95
    velocities['vel95_2']=vel95_2
    velocities['velmean']=velmean
    velocities['velmean_2']=velmean_2
    np.save('velocities_'+month+'_'+case+'.npy',velocities)

def make_fileList(start_time, stop_time, folders):
    '''
    Go through the met offices thredds server and link points in time to files.
    format: yyyy-mm-dd-hh
    '''
    dates   = prepare_dates(start_time, stop_time)
    time    = np.empty(0)
    path    = []
    index   = []

    all_nc  = list_ncfiles(folders) 

    for date in dates:
        # See where the files are available
        # ----
        for ncfile in all_nc:
            d       = Dataset(ncfile)
            t_fvcom = d.variables.get('time')[:]
            ti = np.linspace(0, len(t_fvcom)-1, num=len(t_fvcom))
            for tt in range(len(t_fvcom)-1):
                tt = int(tt)
                time = float(d['Itime'][tt]) + float(d['Itime2'][tt]/(24*60*60*1000))
                if netCDF4.num2date(time, units = d['time'].units)==date:
                    file     = ncfile
                    print('found ' + str(netCDF4.num2date(time, units = d['time'].units))+' in '+file)
                    all_nc.remove(ncfile)
                    break # exit this loop

            #elif netCDF4.num2date(t_fvcom[0], units = d['time'].units)<date:
               # all_nc.remove(ncfile)

        # Store
        path     = path + [file]

    return path

def list_ncfiles(dirs):
    '''
    returns list of all files in directories (or in one single directory)
    '''
    ncfiles = []
    for dr in dirs:
        stuff   = os.listdir(dr)
        sortert = sorted(stuff)
        tmp     = [dr+'/'+fil for fil in sortert if '.nc' in fil and 'restart' not in fil]
        ncfiles.extend(tmp)
        #ncfiles.extend([dr+'/'+fil for fil in stuff if '.nc' in fil and 'restart' not in fil])
    return ncfiles

def bottom_folders(folders):
    '''
    Returns the folders on the bottom of the pyramid (hence the name)
    mandatory: 
    folders   - parent folder(s) to cycle through
    '''
    # ----
    dirs = []

    if isinstance(folders,str):
        folders = [folders]

    for folder in folders:
        dirs.extend([x[0] for x in os.walk(folder)])
    
    # remove folders that are not at the top of the tree
    # ----
    leaf_branch = []
    if len(dirs)>1:
        for dr in dirs:
            if dr[-1]=='/':
                continue
            else:
                # This string is at the end of the branch, thus this is where the data is stored
                # ----
                leaf_branch.append(dr)
    else:
        leaf_branch = dirs
    return leaf_branch

def prepare_dates(start_time,stop_time):
    '''
    returns pandas array of dates needed
    '''
    print('\npreparing dates')
    start       = datetime(int(start_time.split('-')[0]), int(start_time.split('-')[1]),\
                           int(start_time.split('-')[2]),int(start_time.split('-')[3]))
    stop        = datetime(int(stop_time.split('-')[0]), int(stop_time.split('-')[1]),\
                           int(stop_time.split('-')[2]),int(start_time.split('-')[3]))

    return pd.date_range(start,stop)

def get_dc(outfile,depth,Allfiles):

    D = Dataset(Allfiles[0],'r')
    sig = np.array(D['siglay_center'])
    h = np.array(D['h_center'])
    hh = np.matlib.repmat(h,D['siglay_center'].shape[0],1)
    sigz = sig*hh

    sig_db = hh - abs(sigz)

    min_values1 = []
    index_values1 = []
    min_values2 = []
    index_values2 = []
    index_weigth1 = []
    index_weigth2 = []
    dpt = []

    for i in range(len(h)):
        print(i)
        #Find closest depth cell to your intended depth
        values = abs(sig_db[:,i] - depth)
        min_value1 = min(values)
        min_values1.append(min_value1)
        index_min1 = np.argmin(values)
        index_values1.append(index_min1)

        #Find second closest depth cell to your intended depth
        values = abs(sig_db[:,i] - depth)
        i1 = list(range(0,index_min1))
        i2 = list(range(index_min1+1,len(sig[:,1])))
        ind1 = np.array(i1)
        ind2 = np.array(i2)
        if index_min1 == len(sig[:,1])-1:
            ind = ind1
        else:
            ind = np.concatenate((ind1, ind2))

        min_value2 = min(values[ind.astype(int)])
        min_values2.append(min_value2)
        index_min2 = np.argmin(values[ind.astype(int)])
        index_values2.append(ind[index_min2])

        w1 = abs(min_value1)
        w2 = abs(min_value2)
        wgth_1 = 1 - w1/(w1+w2)
        wgth_2 = 1 - w2/(w1+w2)
        if index_min1 == len(sig[:,1])-1:
            if sig_db[index_min1,i] > depth:
                wgth_1 = 1
                wgth_2 = 0

        index_weigth1.append(wgth_1);
        index_weigth2.append(wgth_2);


        dpt.append(depth)


    out = pd.DataFrame({'index1': index_values1,
                       'index2': index_values2,
                       'weight1': index_weigth1,
                       'weight2': index_weigth2,
                       'minval1': min_values1,
                       'minval2': min_values2,
                       'depth': dpt})
    np.save(outfile,out)
    bc = out.to_numpy()
    return bc
