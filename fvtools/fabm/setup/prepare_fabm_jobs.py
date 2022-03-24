import os
import sys
import numpy as np
import xlrd #Only for make_flux_file_real
from shutil import copyfile
from configparser import SafeConfigParser
from datetime import datetime, timedelta
from netCDF4 import Dataset
import utm
from scipy.io import loadmat
import calendar
import pwd
usr=pwd.getpwuid(os.getuid())[0]

from grid.tools import Filelist
import grid.fvcom_grd as fvg
import matplotlib.pyplot as plt


def read_cfg_file(conf_file):
    '''Read information about experiment setup from configuration file.'''

    conf = SafeConfigParser()
    conf.read(conf_file)
    
    return {'source_case_name': conf.get('hydro', 'case_name'), \
            'case_name': conf.get('experiments', 'case_name'), \
            'restart_fileList': conf.get('hydro', 'restart_fileList'), \
            'template_directory': conf.get('template', 'template_directory'), \
            'start_times': conf.get('experiments', 'start_times').split('\n'), \
            'simulation_length': int(conf.get('experiments', 'simulation_length')), \
            'run_directory': conf.get('experiments', 'run_directory'), \
            'result_directory': conf.get('experiments', 'result_directory'), \
            'source_directory': conf.get('hydro', 'source_directory'), \
            'wall_time': conf.get('experiments', 'wall_time'), \
            'lnodes': conf.get('experiments', 'lnodes'), \
            'ppn': conf.get('experiments', 'ppn'), \
            'atm_file': conf.get('experiments', 'atm_file'), \
            'nest_file': conf.get('experiments', 'nest_file'), \
            'grid_file': conf.get('hydro', 'grid_file'), \
            'area_file': conf.get('hydro', 'area_file'), \
            'kb': conf.get('hydro', 'kb'), \
            'center_points': conf.get('experiments', 'center_points') 
            }


def make_run_directories(conf):
    '''Make directory for each experiment copy dat-files, nml-file, run_fvcom.sh and 
       restart-file to run directory.'''

   
    for start_time, num  in zip(conf['start_times'], range(1, len(conf['start_times']) + 1)):
        
        print('Setting up run period starting ' + start_time.__str__()) 
        run_dir = os.path.join(conf['run_directory'], conf['case_name'] + "_" + '{0:02d}'.format(num))
        input_dir = os.path.join(run_dir, "input")
        os.mkdir(run_dir)
        os.mkdir(input_dir)
    
        # Copy dat-files to input directory
        copyfile(os.path.join(conf['source_directory'], 'input', conf['source_case_name'] + '_grd.dat'), 
                 os.path.join(input_dir, conf['case_name'] + '_' + '{0:02}'.format(num) + '_grd.dat'))
        
        copyfile(os.path.join(conf['source_directory'], 'input', conf['source_case_name'] + '_obc.dat'), 
                 os.path.join(input_dir, conf['case_name']  + '_' + '{0:02}'.format(num) + '_obc.dat'))

        copyfile(os.path.join(conf['source_directory'], 'input', conf['source_case_name'] + '_dep.dat'), 
                 os.path.join(input_dir, conf['case_name'] + '_' + '{0:02}'.format(num) + '_dep.dat'))
        
        copyfile(os.path.join(conf['source_directory'], 'input', conf['source_case_name'] + '_sigma.dat'), 
                 os.path.join(input_dir, conf['case_name'] + '_' + '{0:02}'.format(num) + '_sigma.dat'))

        copyfile(os.path.join(conf['source_directory'], 'input', conf['source_case_name'] + '_cor.dat'), 
                 os.path.join(input_dir, conf['case_name'] + '_' + '{0:02}'.format(num) + '_cor.dat'))
        
        copyfile(os.path.join(conf['source_directory'], 'input', conf['source_case_name'] + '_spg.dat'), 
                 os.path.join(input_dir, conf['case_name'] + '_' + '{0:02}'.format(num) + '_spg.dat'))



        # copy run_fvcom.sh
        changes = {'WALLTIME': conf['wall_time'], \
                   'NODES': conf['lnodes'], \
                   'PPN': conf['ppn']}

        copy_template(os.path.join(conf['template_directory'], 'run_fvcom.sh'),
                      os.path.join(conf['run_directory'], conf['case_name'] + "_" + '{0:02d}'.format(num), 'run_fvcom.sh'),
                      replacements=changes)


        
        # Copy nml-file
        start_time = datetime(int(start_time.split('-')[0]),
                              int(start_time.split('-')[1]),
                              int(start_time.split('-')[2]),
                              int(start_time.split('-')[3]))

        stop_time = start_time + timedelta(days=conf['simulation_length'])

        changes = {'STARTTIME': str(start_time), \
                   'STOPTIME': str(stop_time), \
                   'ATMFILE': 'atm_forcing.nc', \
                   'NESTFILE': 'nesting_file.nc', \
                   'CASENAME': conf['case_name'], \
                   'INPUT_DIRECTORY': os.path.join(conf['run_directory'], conf['case_name'] +  "_" + '{0:02d}'.format(num), 'input'), \
                   'OUTPUT_DIRECTORY': os.path.join(conf['result_directory'], conf['case_name'] +  "_" + '{0:02d}'.format(num), 'output')}
 
                             
        copy_template(os.path.join(conf['template_directory'], 'template.nml'),
                      os.path.join(conf['run_directory'], conf['case_name'] +  "_" + '{0:02d}'.format(num), 
                                   conf['case_name'] + "_" + '{0:02d}'.format(num) + '_run.nml'),
                      replacements=changes)

        # Copy river files
        copyfile(os.path.join(conf['source_directory'], 'input', 'riverdata.nc'),
                              os.path.join(input_dir, 'riverdata.nc'))
        copyfile(os.path.join(conf['source_directory'], 'input', 'RiverNamelist.nml'),
                              os.path.join(input_dir, 'RiverNamelist.nml'))
 
 
        # Add link to atm-forcing file in input directory
        os.symlink(conf['atm_file'], os.path.join(input_dir, 'atm_forcing.nc'))        

        # Identify correct restart-file and copy
        fl = Filelist(conf['restart_fileList'])
        nearest = fl.find_nearest(start_time.year, start_time.month, start_time.day)
          
        copyfile(fl.path[nearest], os.path.join(input_dir, 'restart_0001.nc'))


        # Add tracer variables to restart file
        add_tracers_to_restart_file(os.path.join(input_dir, 'restart_0001.nc'))

        
        # Copy nesting-file to input directory and add tracer variables to nesting file
        copyfile(conf['nest_file'], os.path.join(input_dir, 'nesting_file.nc'))          
        add_tracers_to_nesting_file(os.path.join(input_dir, 'nesting_file.nc'))


        # Make fabm-flux file
        make_flux_file(conf, input_dir, start_time, stop_time+timedelta(days=1))

        # Copy other fabm files
        copyfile(os.path.join(conf['template_directory'], 'fabm.yaml'),
                 os.path.join(conf['run_directory'], conf['case_name'] +  "_" + '{0:02d}'.format(num), 'fabm.yaml'))
        copyfile(os.path.join(conf['template_directory'], 'fabm_input.nml'),
                 os.path.join(conf['run_directory'], conf['case_name'] +  "_" + '{0:02d}'.format(num), 'fabm_input.nml'))



def make_result_directories(conf):
    '''Make result-folder for each experiment'''
    
    for start_time, num  in zip(conf['start_times'], range(1, len(conf['start_times']) + 1)):
        os.mkdir(os.path.join(conf['result_directory'], conf['case_name'] +  "_" + '{0:02d}'.format(num)))
        os.mkdir(os.path.join(conf['result_directory'], conf['case_name'] +  "_" + '{0:02d}'.format(num), 'output'))

def make_jobsumission_script(conf):
    '''Write script for submitting all jobs at once'''
    pass


def add_tracers_to_restart_file(restart_file, ini_val=0.0):
    '''Add tracer variables to exixsting restart nc-file'''
    
    restart = Dataset(restart_file,'r+')

    time = restart.variables['time'][:]
    temp = restart.variables['temp'][:,:,:]

    inival=0.0

    tr1 = restart.createVariable('tracer1_c','f4',('time','siglay','node'))
    tr1.units='quantity m-3'
    tr1[:,:,:] = inival

    tr2 = restart.createVariable('tracer2_c','f4',('time','siglay','node'))
    tr2.units='quantity m-3'
    tr2[:,:,:] = inival

    tr3 = restart.createVariable('tracer3_c','f4',('time','siglay','node'))
    tr3.units='quantity m-3'
    tr3[:,:,:] = inival

    tr4 = restart.createVariable('tracer4_c','f4',('time','siglay','node'))
    tr4.units='quantity m-3'
    tr4[:,:,:] = inival

    tr5 = restart.createVariable('tracer5_c','f4',('time','siglay','node'))
    tr5.units='quantity m-3'
    tr5[:,:,:] = inival

    tr6 = restart.createVariable('tracer6_c','f4',('time','siglay','node'))
    tr6.units='quantity m-3'
    tr6[:,:,:] = inival

    tr7 = restart.createVariable('tracer7_c','f4',('time','siglay','node'))
    tr7.units='quantity m-3'
    tr7[:,:,:] = inival

    tr8 = restart.createVariable('tracer8_c','f4',('time','siglay','node'))
    tr8.units='quantity m-3'
    tr8[:,:,:] = inival

    tr1b = restart.createVariable('tracer1_c_bot','f4',('time','node'))
    tr1b.units='quantity m-2'
    tr1b[:,:] = inival

    tr2b = restart.createVariable('tracer2_c_bot','f4',('time','node'))
    tr2b.units='quantity m-2'
    tr2b[:,:] = inival

    tr3b = restart.createVariable('tracer3_c_bot','f4',('time','node'))
    tr3b.units='quantity m-2'
    tr3b[:,:] = inival

    tr4b = restart.createVariable('tracer4_c_bot','f4',('time','node'))
    tr4b.units='quantity m-2'
    tr4b[:,:] = inival

    tr5b = restart.createVariable('tracer5_c_bot','f4',('time','node'))
    tr5b.units='quantity m-2'
    tr5b[:,:] = inival

    tr6b = restart.createVariable('tracer6_c_bot','f4',('time','node'))
    tr6b.units='quantity m-2'
    tr6b[:,:] = inival

    tr7b = restart.createVariable('tracer7_c_bot','f4',('time','node'))
    tr7b.units='quantity m-2'
    tr7b[:,:] = inival

    tr8b = restart.createVariable('tracer8_c_bot','f4',('time','node'))
    tr8b.units='quantity m-2'
    tr8b[:,:] = inival

    restart.close()





def add_tracers_to_nesting_file(restart_file, ini_val=0.0):
     '''Add tracer variables to exixsting nesting nc-file'''

     nest = Dataset(restart_file, 'r+')
     inival = 0.0

     tr1 = nest.createVariable('tracer1_c','f4',('time','siglay','node'))
     tr1.units='quantity m-3'
     tr1[:,:,:] = inival

     tr2 = nest.createVariable('tracer2_c','f4',('time','siglay','node'))
     tr2.units='quantity m-3'
     tr2[:,:,:] = inival

     tr3 = nest.createVariable('tracer3_c','f4',('time','siglay','node'))
     tr3.units='quantity m-3'
     tr3[:,:,:] = inival

     tr4 = nest.createVariable('tracer4_c','f4',('time','siglay','node'))
     tr4.units='quantity m-3'
     tr4[:,:,:] = inival

     tr5 = nest.createVariable('tracer5_c','f4',('time','siglay','node'))
     tr5.units='quantity m-3'
     tr5[:,:,:] = inival

     tr6 = nest.createVariable('tracer6_c','f4',('time','siglay','node'))
     tr6.units='quantity m-3'
     tr6[:,:,:] = inival

     tr7 = nest.createVariable('tracer7_c','f4',('time','siglay','node'))
     tr7.units='quantity m-3'
     tr7[:,:,:] = inival

     tr8 = nest.createVariable('tracer8_c','f4',('time','siglay','node'))
     tr8.units='quantity m-3'
     tr8[:,:,:] = inival

     tr1b = nest.createVariable('tracer1_c_bot','f4',('time','node'))
     tr1b.units='quantity m-2'
     tr1b[:,:] = inival

     tr2b = nest.createVariable('tracer2_c_bot','f4',('time','node'))
     tr2b.units='quantity m-2'
     tr2b[:,:] = inival

     tr3b = nest.createVariable('tracer3_c_bot','f4',('time','node'))
     tr3b.units='quantity m-2'
     tr3b[:,:] = inival

     tr4b = nest.createVariable('tracer4_c_bot','f4',('time','node'))
     tr4b.units='quantity m-2'
     tr4b[:,:] = inival

     tr5b = nest.createVariable('tracer5_c_bot','f4',('time','node'))
     tr5b.units='quantity m-2'
     tr5b[:,:] = inival

     tr6b = nest.createVariable('tracer6_c_bot','f4',('time','node'))
     tr6b.units='quantity m-2'
     tr6b[:,:] = inival

     tr7b = nest.createVariable('tracer7_c_bot','f4',('time','node'))
     tr7b.units='quantity m-2'
     tr7b[:,:] = inival

     tr8b = nest.createVariable('tracer8_c_bot','f4',('time','node'))
     tr8b.units='quantity m-2'
     tr8b[:,:] = inival

     nest.close()



def make_flux_file(conf, input_dir, starttime, endtime):
    '''Make flux input file'''

    grd=fvg.FVCOM_grid(conf['grid_file'])
    kb = int(conf['kb'])
    
    longitude, latitude = np.loadtxt(conf['center_points'], delimiter='\t', skiprows=1, unpack=True)
    posxy=[utm.from_latlon(latitude[x],longitude[x],33) for x in range(len(latitude))]
    xx=[posxy[x][0] for x in range(len(latitude))]
    yy=[posxy[x][1] for x in range(len(longitude))]

    #Find nearest grid point to each center point
    ds2=[np.square(grd.x-xx[i])+np.square(grd.y-yy[i]) for i in range(len(latitude))]
    index_min=[np.where(ds2[x][:]==ds2[x][:].min())[0].astype(int) for x in range(len(ds2))]


    #------------------------------
    #total flux 38 m^3/h ; 50 mg/l -> 50000 mg/m^3 => 527.78 mg/s
    #tot_flux = 600 #mg/s
    #Lese inn control volume area:
    MA = loadmat(conf['area_file'], struct_as_record=True, squeeze_me=False)
    ctr_vol_area = MA['art1'][index_min,0]

    #Distribusjon av synkehast. Dele paa kontrollvolum -> mg/s/m^2
    flux0_1=100.0/ctr_vol_area
    flux0_2=100.0/ctr_vol_area
    flux0_3=100.0/ctr_vol_area
    flux0_4=100.0/ctr_vol_area
    flux0_5=100.0/ctr_vol_area
    flux0_6=100.0/ctr_vol_area
    flux0_7=100.0/ctr_vol_area
    flux0_8=100.0/ctr_vol_area



    #Adjust release times to only release during days
    tdelta=endtime-starttime
    #hourly time
    time = [starttime + timedelta(n)/24 for n in range(0,tdelta.days*24)]

    #identify index of releasetimes:
    fi=[]
    flux_stop = endtime - timedelta(days=5)
    for i in range(len(time)):
        if(time[i].hour >=6 and time[i].hour<18 and time[i] <= flux_stop):
            fi.append(i)

    date0 = datetime(1858,11,17,0,0,0)
    timeFn = [(time[x]-date0).days + (time[x]-date0).seconds/86400.0 for x in range(len(time))]

    #Set flux to nonzero at 06:00-18:00
    flux1=np.zeros([grd.x.shape[0],len(time)])
    flux1[index_min,np.array(fi)] = flux0_1
    flux2=np.zeros([grd.x.shape[0],len(time)])
    flux2[index_min,np.array(fi)]=flux0_2
    flux3=np.zeros([grd.x.shape[0],len(time)])
    flux3[index_min,np.array(fi)]=flux0_3
    flux4=np.zeros([grd.x.shape[0],len(time)])
    flux4[index_min,np.array(fi)]=flux0_4
    flux5=np.zeros([grd.x.shape[0],len(time)])
    flux5[index_min,np.array(fi)]=flux0_5
    flux6=np.zeros([grd.x.shape[0],len(time)])
    flux6[index_min,np.array(fi)]=flux0_6
    flux7=np.zeros([grd.x.shape[0],len(time)])
    flux7[index_min,np.array(fi)]=flux0_7
    flux8=np.zeros([grd.x.shape[0],len(time)])
    flux8[index_min,np.array(fi)]=flux0_8



    #Write to netcdf-file input/flux.nc
    fout = Dataset(os.path.join(input_dir, 'flux.nc'),'w',format='NETCDF4')
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
    inj6 = fout.createVariable('injection6_flux_int','f4',('time','node'))
    inj6.units='kg s-1 m-2'
    inj7 = fout.createVariable('injection7_flux_int','f4',('time','node'))
    inj7.units='kg s-1 m-2'
    inj8 = fout.createVariable('injection8_flux_int','f4',('time','node'))
    inj8.units='kg s-1 m-2'

    times = fout.createVariable('time','f4',('time',))
    times.long_name = 'time'
    times.units = 'days since '+str(date0)
    times.format = 'modified julian day (MJD)'
    times.time_zone = 'UTC'

    #write to netcdf created
    times[:] = timeFn[:]
    inj1[:] = flux1.transpose()
    inj2[:] = flux2.transpose()
    inj3[:] = flux3.transpose()
    inj4[:] = flux4.transpose()
    inj5[:] = flux5.transpose()
    inj6[:] = flux6.transpose()
    inj7[:] = flux7.transpose()
    inj8[:] = flux8.transpose()

    fout.close()

def make_flux_file_real(conf, input_dir, starttime, endtime, Carbon_xlFile, Sc_scale=1.0, Feces_factor=1.59*2.94, Feed_factor=2.0):
    '''Make flux input file based on real production at site'''

    grd=fvg.FVCOM_grid(conf['grid_file'])
    kb = int(conf['kb'])
    
    longitude, latitude = np.loadtxt(conf['center_points'], delimiter='\t', skiprows=1, unpack=True)
    posxy=[utm.from_latlon(latitude[x],longitude[x],33) for x in range(len(latitude))]
    xx=[posxy[x][0] for x in range(len(latitude))]
    yy=[posxy[x][1] for x in range(len(longitude))]

    #Find nearest grid point to each center point
    ds2=[np.square(grd.x-xx[i])+np.square(grd.y-yy[i]) for i in range(len(latitude))]
    index_min=[np.where(ds2[x][:]==ds2[x][:].min())[0].astype(int) for x in range(len(ds2))]

    #Check by plot that positions are correct:
    plt.figure()
    plt.scatter(grd.x,grd.y)
    plt.plot(grd.x[index_min[:],0],grd.y[index_min[:],0],'r.')
    plt.show()
    #plt.close()

    #Read Carbon per month from excel file and convert to released tracers:-----------------------------
    #OBS: make separate "read excel file" def eventually -> output: T1kg, T2kg .... (monthly mass of tracers)
    #Right now calculating from carbon to tracer, eventually excel file with tracer directly
    startm= 1 
    endm = 12 #OBS hardcode for one year
    xl_scen = xlrd.open_workbook(Carbon_xlFile)
    xl_sheet = xl_scen.sheet_by_index(0)

    header1 = xl_sheet.row(0)
    header2 = xl_sheet.row(1)

    months =  [int(xl_sheet.col(1)[x].value) for x in np.arange(2,14)]
    #Obs: oppsett for en case hvor et anlegg med 8 tracere med manedsfordeling: T1kg per mnd 1:12
    #Obs Anta at xl-skjema er i karbon og skalere tilbake til kg tracer (Sc_factor er for a justere exp)
    cind=2
    T1kg = [xl_sheet.col_values(cind)[x]*Sc_scale*Feces_factor for x in np.arange(2,14)]
    T2kg = [xl_sheet.col_values(cind+1)[x]*Sc_scale*Feces_factor for x in np.arange(2,14)]
    T3kg = [xl_sheet.col_values(cind+2)[x]*Sc_scale*Feces_factor for x in np.arange(2,14)]
    T4kg = [xl_sheet.col_values(cind+3)[x]*Sc_scale*Feces_factor for x in np.arange(2,14)]
    T5kg = [xl_sheet.col_values(cind+4)[x]*Sc_scale*Feces_factor for x in np.arange(2,14)]
    T6kg = [xl_sheet.col_values(cind+5)[x]*Sc_scale*Feces_factor for x in np.arange(2,14)]
    T7kg = [xl_sheet.col_values(cind+6)[x]*Sc_scale*Feed_factor for x in np.arange(2,14)]
    T8kg = [xl_sheet.col_values(cind+7)[x]*Sc_scale*Feed_factor for x in np.arange(2,14)]

    #--------------------------------------------------------------------------------------------------------

    #------------------------------
    #total flux 38 m^3/h ; 50 mg/l -> 50000 mg/m^3 => 527.78 mg/s
    #tot_flux = 600 #mg/s
    #Lese inn control volume area:
    MA = loadmat(conf['area_file'], struct_as_record=True, squeeze_me=False)
    ctr_vol_area = MA['art1'][index_min,0]
    #dim = float(calendar.monthrange(starttime.year,starttime.month)[1]) #days in month
    monthday = [31,28,31,30,31,30,31,31,30,31,30,31]

    #Spre likt antall kilo paa hver merde (med ulikt kontroll volum) og regne om til fluks kg/s -> list of months
    #(anta utslipp 12 av 24 t)
    T1kgs_perpoint = [T1kg[x]/max(ctr_vol_area.shape)/(monthday[x]*86400/2.0) for x in np.arange(len(T1kg))]
    T2kgs_perpoint = [T2kg[x]/max(ctr_vol_area.shape)/(monthday[x]*86400/2.0) for x in np.arange(len(T2kg))]
    T3kgs_perpoint = [T3kg[x]/max(ctr_vol_area.shape)/(monthday[x]*86400/2.0) for x in np.arange(len(T3kg))]
    T4kgs_perpoint = [T4kg[x]/max(ctr_vol_area.shape)/(monthday[x]*86400/2.0) for x in np.arange(len(T4kg))]
    T5kgs_perpoint = [T5kg[x]/max(ctr_vol_area.shape)/(monthday[x]*86400/2.0) for x in np.arange(len(T5kg))]
    T6kgs_perpoint = [T6kg[x]/max(ctr_vol_area.shape)/(monthday[x]*86400/2.0) for x in np.arange(len(T6kg))]
    T7kgs_perpoint = [T7kg[x]/max(ctr_vol_area.shape)/(monthday[x]*86400/2.0) for x in np.arange(len(T7kg))]
    T8kgs_perpoint = [T8kg[x]/max(ctr_vol_area.shape)/(monthday[x]*86400/2.0) for x in np.arange(len(T8kg))]
    #Now units: [kg/s] in each of the cages - to adjust for control vol. area of each point this has to be divided later

    #First split into realease per hour of the half day we release:
    tdelta=endtime-starttime
    #hourly time
    time = [starttime + timedelta(n)/24 for n in range(0,tdelta.days*24)]

    #identify index of releasetimes:
    fi=[]
    flux_stop = endtime - timedelta(days=1)
    
    for i in range(len(time)):
        if(time[i].hour >=6 and time[i].hour<18 and time[i] <= flux_stop):
            fi.append(i)

    #Flux in kg/s from each cage as a f(time)
    flux0_1=np.zeros(len(time))
    flux0_2=np.zeros(len(time))
    flux0_3=np.zeros(len(time))
    flux0_4=np.zeros(len(time))
    flux0_5=np.zeros(len(time))
    flux0_6=np.zeros(len(time))
    flux0_7=np.zeros(len(time))
    flux0_8=np.zeros(len(time))

    count=0
    for n in range(len(monthday)):
        flux0_1[fi[count:int(count+monthday[n]*24/2)]] = T1kgs_perpoint[n]
        flux0_2[fi[count:int(count+monthday[n]*24/2)]] = T2kgs_perpoint[n]
        flux0_3[fi[count:int(count+monthday[n]*24/2)]] = T3kgs_perpoint[n]
        flux0_4[fi[count:int(count+monthday[n]*24/2)]] = T4kgs_perpoint[n]
        flux0_5[fi[count:int(count+monthday[n]*24/2)]] = T5kgs_perpoint[n]
        flux0_6[fi[count:int(count+monthday[n]*24/2)]] = T6kgs_perpoint[n]
        flux0_7[fi[count:int(count+monthday[n]*24/2)]] = T7kgs_perpoint[n]
        flux0_8[fi[count:int(count+monthday[n]*24/2)]] = T8kgs_perpoint[n]
        count = int(count + monthday[n]*24/2)
    
    #Divide by control volume area of each cage point to get flux in [kg/s/m^2]
    cflux1 = np.zeros([grd.x.shape[0],len(time)])
    cflux2 = np.zeros([grd.x.shape[0],len(time)])
    cflux3 = np.zeros([grd.x.shape[0],len(time)])
    cflux4 = np.zeros([grd.x.shape[0],len(time)])
    cflux5 = np.zeros([grd.x.shape[0],len(time)])
    cflux6 = np.zeros([grd.x.shape[0],len(time)])
    cflux7 = np.zeros([grd.x.shape[0],len(time)])
    cflux8 = np.zeros([grd.x.shape[0],len(time)])
    for m in range(len(ctr_vol_area)):
      cflux1[index_min[m],:] = flux0_1/ctr_vol_area[m]
      cflux2[index_min[m],:] = flux0_2/ctr_vol_area[m]
      cflux3[index_min[m],:] = flux0_3/ctr_vol_area[m]
      cflux4[index_min[m],:] = flux0_4/ctr_vol_area[m]
      cflux5[index_min[m],:] = flux0_5/ctr_vol_area[m]
      cflux6[index_min[m],:] = flux0_6/ctr_vol_area[m]
      cflux7[index_min[m],:] = flux0_7/ctr_vol_area[m]
      cflux8[index_min[m],:] = flux0_8/ctr_vol_area[m]


    date0 = datetime(1858,11,17,0,0,0)
    timeFn = [(time[x]-date0).days + (time[x]-date0).seconds/86400.0 for x in range(len(time))]

    #Write to netcdf-file input/flux.nc
    fout = Dataset(os.path.join(input_dir, 'flux.nc'),'w',format='NETCDF4')
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
    inj6 = fout.createVariable('injection6_flux_int','f4',('time','node'))
    inj6.units='kg s-1 m-2'
    inj7 = fout.createVariable('injection7_flux_int','f4',('time','node'))
    inj7.units='kg s-1 m-2'
    inj8 = fout.createVariable('injection8_flux_int','f4',('time','node'))
    inj8.units='kg s-1 m-2'

    times = fout.createVariable('time','f4',('time',))
    times.long_name = 'time'
    times.units = 'days since '+str(date0)
    times.format = 'modified julian day (MJD)'
    times.time_zone = 'UTC'
    
    #Control plot of flux tracer 6:
    plt.figure()
    plt.plot(time,cflux6.transpose()[:,index_min[0]])
    plt.show()
    
    #write to netcdf created
    times[:] = timeFn[:]
    inj1[:] = cflux1.transpose()
    inj2[:] = cflux2.transpose()
    inj3[:] = cflux3.transpose()
    inj4[:] = cflux4.transpose()
    inj5[:] = cflux5.transpose()
    inj6[:] = cflux6.transpose()
    inj7[:] = cflux7.transpose()
    inj8[:] = cflux8.transpose()

    fout.close()



def copy_template(template_file, out_file, replacements='None'):
    '''Copy file and replace expressions if input 'replacements' is given.'''
    
    template = open(template_file, 'r')
    out = open(out_file, 'w')
    for line in template:
        if replacements:
            for key, value in replacements.iteritems():
                if key in line:
                    line = line.replace(key, value)

        out.write(line)

    out.close()
    template.close()
 
                    


