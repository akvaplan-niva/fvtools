"""Post-process downloaded ERA5 data for FVCOM surface forcing."""
import os
import xarray as xr
import numpy as np
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units

def process_era5(folder, era5_file = 'ERA5.nc'):
    '''
    Process the data downloaded from ERA5 so that it can be used to create FVCOM forcing files
    
    :param folder:    str, folder the era5 data is stored with
    :param era5_file: str, name of the ERA5 file you will pass to the interpolation routine
    '''
    # Combine all files to one
    files = [folder + f for f in os.listdir(folder) if len(f) > 26 and '.nc' in f]
    with xr.open_mfdataset(files) as df:
        # Rename netCDF attributes to the format expected by interpolate_regular
        df = df.rename_dims({'valid_time': 'time', 'longitude': 'lon', 'latitude': 'lat'})
        df = df.rename_vars({'valid_time': 'time', 'longitude': 'lon', 'latitude': 'lat'})
        
        # Calculate the relative humidity and dump to the dataset
        time = df['time'].compute().data
        temp   = df.t2m.compute().data * units.K
        relhum = relative_humidity_from_dewpoint(temp, df.d2m.compute().data * units.K).to('percent')
        df['relative_humidity'] = (('time', 'lat', 'lon'), relhum)
        
        # Return the accumulated data as a speed
        tp = _de_accumulate(df['tp'].compute().data, time)
        e  = _de_accumulate(df['e'].compute().data, time)
        
        # Dump the data to the netCDF file
        df.tp.data = tp
        df.e.data  = e
        
        # Convert temperatures to celsius
        df.t2m.data = temp.to('celsius')
        
        # Update the attributes to better 
        df['tp'] = df.tp.assign_attrs(units = 'm/s')
        df['tp'] = df.tp.assign_attrs(long_name = 'precipitation rate')
        df['e']  = df.e.assign_attrs(units = 'm/s')
        df['e']  = df.e.assign_attrs(long_name = 'evaporation rate')
        df['t2m']  = df.t2m.assign_attrs(units = 'Degree (C)')

        # This file can be read by add_surface_forcing, and thus interpolated to the FVCOM grid
        df.to_netcdf(era5_file)

def _de_accumulate(acc, time):
    '''
    For hourly accumulated data
    '''
    # time difference in seconds
    dt = np.diff(time) / np.timedelta64(1, 's')
    
    # Assume the first timestep was as long as the next one, and that the first timestep was the first value
    dt = np.insert(dt, 0, dt[0], axis = 0)

    return acc / dt[:, None, None]

def fvcom_era5_translation():
    '''
    Translate the variable names in the ERA5 file to FVCOM name conventions
    '''
    fvcom_era5 = {
        'vwind_speed': 'v10',
        'uwind_speed': 'u10',
        'short_wave': 'avg_snswrf', 
        'long_wave': 'avg_sdlwrf',
        'evap': 'e', 
        'precip': 'tp',
        'cloud_cover': 'tcc',
        'air_pressure': 'sp',
        'air_temperature': 't2m',
        'relative_humidity': 'relative_humidity'
    }
    # Tag what type of grid point we will interpolate stuff to
    elemdata = ['U10', 'V10', 'uwind_speed', 'vwind_speed']
    modes = {}
    for var in fvcom_era5:
        if var in elemdata:
            modes[var] = 'elements'
        else:
            modes[var] = 'nodes'

    return fvcom_era5, modes
