#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

#import cPickle as pickle
from time import gmtime, strftime

from IPython.core.debugger import Tracer
from netCDF4 import Dataset
import numpy as np
import netCDF4

from grid.tools import Filelist
from grid.tools import load

def wrf2fvcom(Nearest4,
              fileList="/home/hdj002/python_script/fvcom_pytools/atm/wrf_fileList.txt", 
              outfile="atm_forcing.nc", 
              start_time=None, 
              stop_time=None):
    '''Read WRF-data, interpolate to FVCOM-grid and export to nc-file.'''
    
    if isinstance(Nearest4, str):
        Nearest4 = load(Nearest4)

    if isinstance(fileList, str):
        Fl = Filelist(fileList, start_time=start_time, stop_time=stop_time, time_format='WRF')
         
    out = Dataset(outfile, 'r+')    

    
    # Loop through file list and add data to nc-file
    first_time=1
    already_read=""
    counter=0
    for time, dtime, path, index in zip(Fl.time, Fl.datetime, Fl.path, Fl.index):
     
        if path != already_read:
            nc = Dataset(path,'r')
            print('')
            print('''Reading from file:''')
            print(path)


        already_read = path

        if first_time:
            out.variables['lat'][:] = Nearest4.FVCOM_grd.y
            out.variables['lon'][:] = Nearest4.FVCOM_grd.x
            out.variables['latc'][:] = Nearest4.FVCOM_grd.yc
            out.variables['lonc'][:] = Nearest4.FVCOM_grd.xc
            out.variables['nv'][:] = Nearest4.FVCOM_grd.tri.T + 1

            first_time = 0
        
        #Tracer()()
        fvcom_time = netCDF4.date2num(dtime, units = 'days since 1858-11-17 00:00:00') 
        out.variables['time'][counter] = fvcom_time
        out.variables['Itime'][counter] = fvcom_time
        out.variables['Itime2'][counter] = (fvcom_time - np.floor(fvcom_time)) * 24 * 60 * 60 * 1000

        Uwind = nc.variables.get('Uwind')[index, :, :][Nearest4.fv_domain_mask] 
        #Uwind = Uwind[Nearest4.fv_domain_mask]
        Uwind = np.sum(Uwind[Nearest4.cindex] * Nearest4.ccoef, axis=1)
        out.variables['U10'][counter, :] = Uwind
        out.variables['uwind_speed'][counter, :] = Uwind

        Vwind = nc.variables.get('Vwind')[index, :, :][Nearest4.fv_domain_mask]
        Vwind = np.sum(Vwind[Nearest4.cindex] * Nearest4.ccoef, axis=1)
        out.variables['V10'][counter, :] = Vwind
        out.variables['vwind_speed'][counter, :] = Vwind

        Tair = nc.variables.get('Tair')[index, :, :] [Nearest4.fv_domain_mask]
        Tair = np.sum(Tair[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        out.variables['SAT'][counter, :] = Tair

        Pair = nc.variables.get('Pair')[index, :, :] [Nearest4.fv_domain_mask]
        Pair = np.sum(Pair[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        out.variables['air_pressure'][counter, :] = 100 * Pair
        
        Qair = nc.variables.get('Qair')[index, :, :] [Nearest4.fv_domain_mask]
        Qair = np.sum(Qair[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        
        relative_humidity = rhumid_calc(Tair, Qair, Pair)
        out.variables['relative_humidity'][counter, :] = 100 * relative_humidity
         
        counter = counter + 1
        
        if np.mod(counter, 8) == 0:
            print(dtime)

    out.close()



def wrf_prec2fvcom(Nearest4,
              fileList="/home/hdj002/python_script/fvcom_pytools/atm/wrf_prec_fileList.txt", 
              outfile="atm_forcing.nc", 
              start_time=None, 
              stop_time=None):
    '''Read WRF-data, interpolate to FVCOM-grid and export to nc-file.'''
    
    if isinstance(Nearest4, str):
        Nearest4 = load(Nearest4)
     
    if isinstance(fileList, str):
        Fl = Filelist(fileList, start_time=start_time, stop_time=stop_time, time_format='WRF')

    out = Dataset(outfile, 'r+')    

    # Loop through file list and add data to nc-file
    already_read=""
    counter=0
    for time, dtime, path, index in zip(Fl.time, Fl.datetime, Fl.path, Fl.index):
        
        if path != already_read:
            nc = Dataset(path,'r')
            print('')
            print('''Reading from file:''')
            print(path)


        already_read = path

       
        #Tracer()()       
        rain = nc.variables.get('rain')[index, :, :] [Nearest4.fv_domain_mask]
        rain = np.sum(rain[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        out.variables['precip'][counter, :] = rain / 1000
        
        out.variables['evap'][counter, :] = np.zeros(rain.shape)
         
        counter = counter + 1
        if np.mod(counter, 8) == 0:
            print(dtime)

   
    out.close()





def wrf_fluxes2fvcom(Nearest4,
              fileList="/home/hdj002/python_script/fvcom_pytools/atm/wrf_fluxes_fileList.txt", 
              outfile="atm_forcing.nc", 
              start_time=None, 
              stop_time=None):
    '''Read WRF-data, interpolate to FVCOM-grid and export to nc-file.'''
    
    if isinstance(Nearest4, str):
        Nearest4 = load(Nearest4)

    if isinstance(fileList, str):
        Fl = Filelist(fileList, start_time=start_time, stop_time=stop_time, time_format='WRF')
         
    out = Dataset(outfile, 'r+')    

    
    # Loop through file list and add data to nc-file
    already_read=""
    counter=0
    for time, dtime, path, index in zip(Fl.time, Fl.datetime, Fl.path, Fl.index):
       
        if path != already_read:
            nc = Dataset(path,'r')
            print('')
            print('''Reading from file:''')
            print(path)


        already_read = path

       
        #Tracer()()        
        swrad = nc.variables.get('swrad')[index, :, :] [Nearest4.fv_domain_mask]
        swrad = np.sum(swrad[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        out.variables['short_wave'][counter, :] = swrad
 
        lwrad_down = nc.variables.get('lwrad_down')[index, :, :] [Nearest4.fv_domain_mask]
        lwrad_down = np.sum(lwrad_down[Nearest4.nindex] * Nearest4.ncoef, axis=1)
        out.variables['long_wave'][counter, :] = lwrad_down
 
        counter = counter + 1
        if np.mod(counter, 8) == 0:
            print(dtime)

    out.close()



def create_nc_forcing_file(name, FVCOM_grd):
    '''Creates empty nc file formatted to fit fvcom atm forcing'''
    nc = Dataset(name, 'w', format='NETCDF4')
    
    # Write global attributes
    nc.title = 'FVCOM Forcing File'
    nc.institution = 'Akvaplan-niva AS'
    nc.source = 'FVCOM grid (unstructured) surface forcing'
    nc.created = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # Create dimensions
    nc.createDimension('time', 0)
    nc.createDimension('node', len(FVCOM_grd.x))
    nc.createDimension('nele', len(FVCOM_grd.xc))
    nc.createDimension('three', 3)

    # Create variables and variable attributes
    Itime = nc.createVariable('Itime', 'int32', ('time',))
    Itime.units = 'days since 1858-11-17 00:00:00'
    Itime.format = 'modified julian day (MJD)'
    Itime.time_zone = 'UTC'

    Itime2 = nc.createVariable('Itime2', 'int32', ('time',))
    Itime2.units = 'msec since 00:00:00'
    Itime2.time_zone = 'UTC'
    
    lat = nc.createVariable('lat', 'single', ('node',))
    lat.long_name = 'nodal latitude'
    lat.units = 'degrees'
    
    latc = nc.createVariable('latc', 'single', ('nele',))
    latc.long_name = 'elemental latitude'
    latc.units = 'degrees'

    lon = nc.createVariable('lon', 'single', ('node',))
    lon.long_name = 'nodal longitude'
    lon.units = 'degrees'
    
    lonc = nc.createVariable('lonc', 'single', ('nele',))
    lonc.long_name = 'elemental longitude'
    lonc.units = 'degrees'

    nv = nc.createVariable('nv', 'int32', ('three', 'nele',))
    nv.long_name = 'nodes surrounding elements'    
    
    precip = nc.createVariable('precip', 'single', ('time', 'node',))
    precip.long_name = 'Precipitation'
    precip.description = 'Precipitation, ocean lose water is negative'
    precip.units = 'm s-1'
    precip.grid = 'fvcom_grid'
    precip.coordinates = ''
    precip.type = 'data'

    time = nc.createVariable('time', 'single', ('time',))
    time.units = 'days since 1858-11-17 00:00:00'
    time.format = 'modified julian day (MJD)'
    time.time_zone = 'UTC'
    
    evap = nc.createVariable('evap', 'single', ('time', 'node',))
    evap.long_name = 'Evaporation'
    evap.description = 'Evaporation, ocean lose water is negative'
    evap.units = 'm s-1'
    evap.grid = 'fvcom_grid'
    evap.coordinates = ''
    evap.type = 'data'
    
    relative_humidity = nc.createVariable('relative_humidity', 'single', ('time', 'node',))
    relative_humidity.long_name = 'Relative Humidity'
    relative_humidity.units = 'kg/kg'
    relative_humidity.grid = 'fvcom_grid'
    relative_humidity.coordinates = ''
    relative_humidity.type = 'data'

    long_wave = nc.createVariable('long_wave', 'single', ('time', 'node',))
    long_wave.long_name = 'Long Wave Radiation'
    long_wave.units = 'W m-2'
    long_wave.grid = 'fvcom_grid'
    long_wave.coordinates = ''
    long_wave.type = 'data'

    short_wave = nc.createVariable('short_wave', 'single', ('time', 'node',))
    short_wave.long_name = 'Short Wave Radiation'
    short_wave.units = 'W m-2'
    short_wave.grid = 'fvcom_grid'
    short_wave.coordinates = ''
    short_wave.type = 'data'

    air_pressure = nc.createVariable('air_pressure', 'single', ('time', 'node',))
    air_pressure.long_name = 'Surface Air Pressure'
    air_pressure.units = 'Pa'
    air_pressure.grid = 'fvcom_grid'
    air_pressure.coordinates = ''
    air_pressure.type = 'data'

    SAT = nc.createVariable('SAT', 'single', ('time', 'node',))
    SAT.long_name = 'Sea surface air temperature'
    SAT.units = 'Degree (C)'
    SAT.grid = 'fvcom_grid'
    SAT.coordinates = ''
    SAT.type = 'data'

    U10 = nc.createVariable('U10', 'single', ('time', 'nele',))
    U10.long_name = 'Eastward Wind Speed'
    U10.units = 'm/s'
    U10.grid = 'fvcom_grid'
    U10.coordinates = ''
    U10.type = 'data'
    
    uwind_speed = nc.createVariable('uwind_speed', 'single', ('time', 'nele',))
    uwind_speed.long_name = 'Eastward Wind Speed'
    uwind_speed.units = 'm/s'
    uwind_speed.grid = 'fvcom_grid'
    uwind_speed.coordinates = ''
    uwind_speed.type = 'data'

    V10 = nc.createVariable('V10', 'single', ('time', 'nele',))
    V10.long_name = 'Northward Wind Speed'
    V10.units = 'm/s'
    V10.grid = 'fvcom_grid'
    V10.coordinates = ''
    V10.type = 'data'
    
    vwind_speed = nc.createVariable('vwind_speed', 'single', ('time', 'nele',))
    vwind_speed.long_name = 'Northward Wind Speed'
    vwind_speed.units = 'm/s'
    vwind_speed.grid = 'fvcom_grid'
    vwind_speed.coordinates = ''
    vwind_speed.type = 'data'

    nc.close()




def nearest4(FVCOM_grd, WRF_grd):
    '''Create nearest four indices and weights'''
   
    N4 = N4WRF(FVCOM_grd, WRF_grd)

    # Find mask for extracing WRF-points limited by FVCOM grid domain
    N4.fv_domain_mask = WRF_grd.crop_grid(xlim=[FVCOM_grd.x.min() - 800, FVCOM_grd.x.max() + 800],
                                          ylim=[FVCOM_grd.y.min() - 800, FVCOM_grd.y.max() + 800]
                                         )
                                            
    x_wrf = WRF_grd.x[N4.fv_domain_mask]
    y_wrf = WRF_grd.y[N4.fv_domain_mask]

    k = 0 
    for x, y in zip(FVCOM_grd.x, FVCOM_grd.y):
        distance = np.sqrt((x_wrf - x)**2 + (y_wrf - y)**2)       
        indices_sorted_according_to_distance = distance.argsort()[0:4]
        nearest_distances = distance[indices_sorted_according_to_distance]
        N4.ndistance[k,:] = nearest_distances
        nearest_distances = 1 / nearest_distances
        sum_of_distances = np.sum(nearest_distances)
        N4.nindex[k,:] = indices_sorted_according_to_distance
        N4.ncoef[k,:] = nearest_distances/sum_of_distances
        k = k + 1
        
    
    k = 0
    for x, y in zip(FVCOM_grd.xc, FVCOM_grd.yc):
        distance = np.sqrt((x_wrf - x)**2 + (y_wrf - y)**2)       
        indices_sorted_according_to_distance = distance.argsort()[0:4]
        nearest_distances = distance[indices_sorted_according_to_distance]
        nearest_distances = 1 / nearest_distances
        sum_of_distances = nearest_distances.sum()
        N4.cindex[k,:] = indices_sorted_according_to_distance
        N4.ccoef[k,:] = nearest_distances/sum_of_distances
        k = k + 1
        
    
    N4.nindex = N4.nindex.astype(int)
    N4.cindex = N4.cindex.astype(int)
    N4.FVCOM_grd = FVCOM_grd
    N4.WRF_grd = WRF_grd

    return N4



class N4WRF():
    '''Object with indices and coefficients for WRF to FVCOM interpolation'''

    def __init__(self, FVCOM_grd, WRF_grd):
        '''Initialize empty attributes'''
        self.ncoef = np.empty([len(FVCOM_grd.x), 4])
        self.nindex = np.empty([len(FVCOM_grd.x), 4])
        self.ccoef = np.empty([len(FVCOM_grd.xc), 4])
        self.cindex = np.empty([len(FVCOM_grd.xc), 4])
        self.ndistance = np.empty([len(FVCOM_grd.x), 4])
        self.cdistance = np.empty([len(FVCOM_grd.xc), 4])
         
    def save(self, name="Nearest4"):
        '''Save object to file.'''
        pickle.dump(self, open( name + ".p", "wb" ) )

       

def rhumid_calc(air_temp, specific_humidity, sea_level_pressure):
    '''Calculate relative humidity from specific humidity, temperature and sea level pressure'''

    # Calculation saturation pressure
    saturation_pressure = 611.2 *np.exp((17.67 * air_temp) / (243.5 + air_temp));

    relative_humidity = 100 * sea_level_pressure * specific_humidity / (0.622 * saturation_pressure);

    return relative_humidity







    


