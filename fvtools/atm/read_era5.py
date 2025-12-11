from fvtools.interpolators.horizontal_interpolator import N4Coefficients
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.era5_grid import ERA5
from fvtools.grid.tools import date2num

from time import strftime, gmtime
from datetime import datetime
from netCDF4 import Dataset

import xarray as xr
import progressbar as pb
import numpy as np

def main(grd_file, outfile, era5file, start_time, stop_time):
    '''
    Create AROME atmospheric forcing file for FVCOM 

    Parameters:
    ----
    grd_file:   'M.npy' or equivalent
    outfile:    name of netcdf output
    era5file:   name of ERA5 file we're interpolating from
    start_time: 'yyyy-mm-dd-hh'
    stop_time:  'yyyy-mm-dd-hh'
    nearest4:   'nearest4arome.npy' file if you have already made an _atm file for this mesh (=None by default)
    latlon:     Set true if you are making a latlon model (will otherwise rotate to adjust for UTM north/east distortion)
    '''
    print(f'\nCreate {outfile} - unstuctured grid atmospheric forcing file\n---')
    print(f'- Load {grd_file} and {era5file}')

    # Load the FVCOM grid object
    M = FVCOM_grid(grd_file)

    # Load the AROME grid, project lat lon coordinates to the same x,y
    E = ERA5(era5file, Proj = M.Proj)

    # Compute nearest 4 coefficients from ERA5 grid points to the FVCOM grid
    print(f'\n- Compute nearest 4')
    N4 = N4Coefficients(M, E)
    N4.compute_nearest4()

    # Create a forcing file for the model
    create_surface_forcing_file(M, outfile)

    # Interpolate from ERA5 to FVCOM
    # - loop over all timesteps
    df = xr.open_dataset(era5file).sel(time = slice(start_time, stop_time))

    # Aliases
    fvcom_era5 = fvcom_era5_translation()
    element_variables = ['vwind_speed', 'uwind_speed']

    # Interpolate to the forcing file
    print('\nInterpolating to the forcing file')
    widget = [f'- Interpolating ERA5 imesteps to the FVCOM grid: ', pb.Percentage(), pb.BouncingBar(), pb.ETA()]
    bar = pb.ProgressBar(widgets = widget, maxval = df.time.shape[0])
    bar.start()
    with Dataset(outfile, 'r+') as d:
        for i, (t, g) in enumerate(df.groupby('time')):
            bar.update(i)
            # Dump the time variables
            time = date2num([datetime.fromisoformat(str(t)+'Z')])[0]
            Itime = np.floor(time)
            d['time'][i]   = time
            d['Itime'][i]  = Itime
            d['Itime2'][i] = (time - Itime) * 24 * 60 * 60 * 1000

            # Interpolate the variables to the grid and dump to the forcing file
            for this_fvcom, era_variable in fvcom_era5.items():
                if this_fvcom in element_variables:
                    if era_variable[1] == 'all':
                        d[this_fvcom][i,:] = np.sum(g[era_variable[0]].isel(time = 0).data[N4.fv_domain_mask][N4.cindex] * N4.ccoef, axis = 1)
                    elif era_variable[1] == 'ocean':
                        d[this_fvcom][i,:] = np.sum(g[era_variable[0]].isel(time = 0).data[N4.fv_domain_mask][N4.cindex_ocean] * N4.ccoef_ocean, axis = 1)
                else:
                    if era_variable[1] == 'all':
                        d[this_fvcom][i,:] = np.sum(g[era_variable[0]].isel(time = 0).data[N4.fv_domain_mask][N4.nindex] * N4.ncoef, axis = 1)
                    elif era_variable[1] == 'ocean':
                        d[this_fvcom][i,:] = np.sum(g[era_variable[0]].isel(time = 0).data[N4.fv_domain_mask][N4.nindex_ocean] * N4.ncoef_ocean, axis = 1)
    bar.finish()
    print(f'Finished! ERA5 data interpolated to {outfile}')

def create_surface_forcing_file(M, ncfile, format = 'NETCDF4', precision = 'f4', **kwargs):
    '''
    Write a forcing file for atmospheric data
    
    :param ncfile:    name of the netCDF file
    :param format:    netCDF format, defaults to netCDF4, optional
    :param precision: 'f4' by default,
    
    Any other keyword arguments will be passed on to netCDF4.Dataset
    '''
    print(f'\n- Creating {ncfile}')
    if not '.nc' in ncfile[-3:]:
        raise ValueError(f'the file {ncfile} is not a .nc file')
    
    ncopts = {}
    if 'ncopts' in kwargs:
        ncopts = kwargs['ncopts']
        kwargs.pop('ncopts')

    # Define the global attributes
    globals = {
        f'title': 'FVCOM forcing file',
        'source': 'FVCOM grid (unstructured) surface forcing',
        'history': f'File created by read_era5 in fvtools',
        'created': f'Created {strftime("%Y-%m-%d %H:%M:%S", gmtime())}',
        'Conventions': 'CF-1.0'
        }

    dims = {
        'time': None, 
        'nele': len(M.xc), 
        'node': len(M.x),
        'three': 3
        }
    
    def add_variable(atmfile, name, data, dimensions, attributes = None, format = 'f4', ncopts = {}):
        var = atmfile.createVariable(name, format, dimensions, **ncopts)
        if attributes:
            for attribute in attributes:
                setattr(var, attribute, attributes[attribute])

        var[:] = data
    
    with Dataset(ncfile, 'w', clobber=True, format=format) as atmfile:            
        # Set attributes and define dimensions
        for attribute in globals:
            setattr(atmfile, attribute, globals[attribute])

        for dimension in dims:
            atmfile.createDimension(dimension, dims[dimension])

        # Add time to the file
        time               = atmfile.createVariable('time', 'single', ('time',))
        time.units         = 'days since 1858-11-17 00:00:00'
        time.format        = 'modified julian day (MJD)'
        time.time_zone     = 'UTC'

        Itime              = atmfile.createVariable('Itime', 'int32', ('time',))
        Itime.units        = 'days since 1858-11-17 00:00:00'
        Itime.format       = 'modified julian day (MJD)'
        Itime.time_zone    = 'UTC'

        Itime2             = atmfile.createVariable('Itime2', 'int32', ('time',))
        Itime2.units       = 'msec since 00:00:00'
        Itime2.time_zone   = 'UTC'

        lons = {'units': 'degrees_north', 'standard_name': 'latitude'}
        lats = {'units': 'degrees_east', 'standard_name': 'longitude'}
        
        print('Adding grid variables to netCDF')
        for g in ['x', 'y']:
            atts = {'units': 'meters', 'long_name': f'nodal {g}-coordinate'}
            add_variable(atmfile, g, M.x, ['node'], attributes=atts, ncopts=ncopts)

        atts = {'long_name': 'nodal longitude'} | lons
        add_variable(atmfile, 'lon', M.lon, ['node'], attributes=atts, ncopts=ncopts)

        atts = {'long_name': 'nodal latitude'} | lats
        add_variable(atmfile, 'lat', M.lat, ['node'], attributes=atts, ncopts=ncopts)

        for g in ['xc', 'yc']:
            atts = {'units': 'meters', 'long_name': f'zonal {g.split("c")[0]}-coordinate'}
            add_variable(atmfile, g, getattr(M, g), ['nele'], attributes=atts, ncopts=ncopts)

        atts = {'long_name': 'zonal longitude'} | lons
        add_variable(atmfile, 'lonc', M.lonc, ['nele'], attributes=atts, ncopts=ncopts)

        atts = {'long_name': 'zonal latitude'} | lats
        add_variable(atmfile, 'latc', M.latc, ['nele'], attributes=atts, ncopts=ncopts)

        atts = {'long_name': 'nodes surrounding element'}
        add_variable(atmfile, 'nv', M.tri.T+1, ['three', 'nele'], format='i4', attributes=atts, ncopts=ncopts)

        print('Define which surface forcing we will dump to the netCDF')
        atts = {
            'precip': {
                'long_name': 'Precipitation',
                'description': 'Precipitation, ocean lose water if negative',
                'units': 'm s-1', 
                'positive': 'up'
                },
            'evap': {
                'long_name': 'Evaporation',
                'description': 'Evaporation, ocean lose water is negative',
                'units': 'm s-1', 
                'positive': 'up'
                },
            'relative_humidity': {'long_name': 'Relative Humidity', 'units': '%'},
            'long_wave': {'long_name': 'Long Wave Radiation', 'units': 'W m-2'},
            'short_wave': {'long_name': 'Short Wave Radiation', 'units': 'W m-2'},
            'cloud_cover': {'long_name': 'Cloud Area Fraction', 'units': 'cloud covered fraction of sky [0,1]'},
            'air_pressure': {'long_name': 'Surface Air Pressure', 'units': 'Pa'},
            'air_temperature': {'long_name': 'Sea Surface Air Temperature', 'units': 'Degree (C)'},
            'uwind_speed': {'long_name': 'Eastward Wind Speed', 'units': 'm/s'},
            'vwind_speed': {'long_name': 'Eastward Wind Speed', 'units': 'm/s'},
        }

        # Add grid identifier metadata to the netCDF file
        elemdata = ['uwind_speed', 'vwind_speed']
        template = {
            'grid': 'fvcom_grd',
            'coordinates': '',
            'type': 'data'
        }
        for attr in atts:
            if attr in elemdata:
                atts[attr] = atts[attr] | template | {'location': 'nele'}
            else:
                atts[attr] = atts[attr] | template | {'location': 'node'}

        # Loop over all attributes and add variable to the netCDF file
        for attr in atts:
            print(f'  - Adding {attr} to the surface forcing file')
            var = atmfile.createVariable(attr, precision, ['time', atts[attr]['location']], **ncopts)
            for attribute in atts[attr]:
                setattr(var, attribute, atts[attr][attribute])

def fvcom_era5_translation():
    '''
    - Link the names of era5 variables to FVCOM forcing names
    - Tell the interpolator wether to use land points in the ERA5 forcing, or exclusively 
      use ocean points
    '''
    return {
        'vwind_speed': ['v10', 'all'],
        'uwind_speed': ['u10', 'all'],
        'short_wave': ['avg_snswrf', 'ocean'], 
        'long_wave': ['avg_sdlwrf', 'ocean'],
        'evap': ['e', 'ocean'], 
        'precip': ['tp', 'all'],
        'cloud_cover': ['tcc', 'all'],
        'air_pressure': ['sp', 'ocean'],
        'air_temperature': ['t2m', 'all'],
        'relative_humidity': ['relative_humidity', 'all']
    }