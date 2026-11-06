import numpy as np
import xarray as xr
import netCDF4
import pandas as pd

epoch = pd.Timestamp('1858-11-17 00:00:00')


def create_dataset(flux_varname, flux, nodes, time):
    ds = xr.Dataset(
        data_vars={
            flux_varname: (('time', 'node'), flux),
            'Itime': ('time', time.astype(np.float32)),
            'Itime2': ('time', [0, 0]),
        },
        coords={
            'time': ('time', time.astype(np.float32)),
        }
    )
    _add_metadata(ds, flux_varname)
    return ds


def _add_metadata(ds, flux_varname):
    """
    Adds metadata to flux Dataset in-place
    """
    for t in ['time', 'Itime']:
        ds[t].attrs['units'] = 'days since 1858-11-17 00:00:00'
        ds[t].attrs['format'] = 'modified julian day (MJD)'
        ds[t].attrs['time_zone'] = 'UTC'

    t = 'Itime2'
    ds[t].attrs['units'] = 'msec since 00:00:00'
    ds[t].attrs['time_zone'] = 'UTC'

    ds[flux_varname].attrs['units'] = 'mmol s^-1 m^-2'
    ds.attrs['type'] = 'FABM FORCING FILE'
    ds.attrs['institution'] = 'Akvaplan-niva'
    ds.attrs['history'] = f'Created with xarray {xr.__version__} on {pd.Timestamp.now()}'


def create_zero_data(nodes, start_time, stop_time):
    """
    Creates zero flux.

    Nodes: Node indices
    """
    time = np.array([
        (start_time - epoch).days-1,
        (stop_time - epoch).days+1
    ]).astype(np.float32)

    flux = np.zeros((len(time), len(nodes))).astype(np.float32)

    return flux, nodes, time
