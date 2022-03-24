import numpy as np
import xarray as xr
import pandas as pd


from ..util.time import epoch_pandas as epoch

def create_zero_dataset(flux_varname, nodes, times):
    flux, nodes, time = _create_zero_data(nodes, times)

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
    ds.attrs['history'] = f'Created with xarray on {pd.Timestamp.now()}'


def _create_zero_data(index, time):
    """
    Creates zero data on a grid given by index and time stamps.
    """
    time_in_days = np.array([(t-epoch).days-1 for t in time]).astype(np.float32)

    data = np.zeros((len(time), len(index))).astype(np.float32)

    return data, index, time
