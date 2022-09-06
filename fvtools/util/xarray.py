import numpy as np
import xarray as xr
import pandas as pd
import netCDF4

def load_fvcom_file_as_xarray(fname, **kwarg):
    """
    Works around xarray limitation to load FVCOM grid data
    """
    with netCDF4.Dataset(fname) as f:
        ds = xr.load_dataset(fname, decode_times=False, drop_variables=['siglay', 'siglev'], **kwarg)
        if 'siglay' in ds.dims:
            ds = _rename_siglay(ds, f)
        if 'siglev' in ds.dims:
            ds = _rename_siglev(ds, f)

    return ds

def open_fvcom_file_as_xarray(fname, **kwarg):
    """
    Works around xarray limitation to load FVCOM grid data
    """
    with netCDF4.Dataset(fname) as f:
        ds = xr.open_dataset(fname, decode_times=False, drop_variables=['siglay', 'siglev'], **kwarg)
        if 'siglay' in ds.dims:
            ds = _rename_siglay(ds, f)
        if 'siglev' in ds.dims:
            ds = _rename_siglev(ds, f)

    return ds

def _rename_siglay(ds, f):
    ds = ds.rename_dims({'siglay': 'siglay_dim'})
    ds = ds.assign(variables={
        'siglay': (('siglay_dim', 'node'), np.copy(f['siglay'])),
    })
    return ds


def _rename_siglev(ds, f):
    ds = ds.rename_dims({'siglev': 'siglev_dim'})
    ds = ds.assign(variables={
        'siglev': (('siglev_dim', 'node'), np.copy(f['siglev'])),
    })
    return ds
