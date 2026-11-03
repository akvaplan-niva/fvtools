import cdsapi
import os
import xarray as xr
import numpy as np
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units
from datetime import datetime, timedelta

from fvtools.grid.fvcom_grd import FVCOM_grid

def main(model: FVCOM_grid, start: datetime, stop: datetime) -> None:
    '''
    Download ERA5 data using the cdsapi
    
    model: pyfvcom.preproc.Model instance
    '''
    # Define what data you will download
    years  = np.array([start.year, stop.year])
    months = np.array([start.month, stop.month])
    if np.diff(years) == 0:
        years = [f'{years[0]}']
        months = [f'{m:2d}' for m in range(months[0], months[1]+1)]
    else:
        years = [f'{y}' for y in range(years[0], years[1]+1)]
        months = [f'{m:2d}' for m in range(1,13)]

    variables = [
        #"10m_u_component_of_wind",
        #"10m_v_component_of_wind",
        #"2m_temperature",
        #"surface_pressure",
        #"total_precipitation",
        #"mean_surface_downward_long_wave_radiation_flux",
        #"mean_surface_net_short_wave_radiation_flux",
        #"total_cloud_cover",
        #"evaporation",
        "2m_dewpoint_temperature",
        "land_sea_mask"
    ]

    dataset = "reanalysis-era5-single-levels"
    for variable in variables:
        request = {
            "product_type": ["reanalysis"],
            "variable": [variable],
            "year": years,
            "month": months,
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [model.lat.max(), model.lon.min(), model.lat.min(), model.lon.max()]
        }
        
        client = cdsapi.Client()
        client.retrieve(dataset, request).download()

def process_era5(folder, era5_file = 'ERA5.nc'):
    '''
    Process the data downloaded from ERA5 so that it can be used to create FVCOM forcing files
    
    :param folder:    str, folder the era5 data is stored with
    :param era5_file: str, name of the ERA5 file you will pass to the interpolation routine
    '''
    # Combine all files to one
    files = [f for f in os.listdir(folder) if len(f) > 26 and '.nc' in f]
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
