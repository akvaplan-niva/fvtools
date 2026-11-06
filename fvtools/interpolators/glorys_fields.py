"""GLORYS/CMEMS field I/O and interpolation shared by restart and nesting.

Input field arrays retain native (depth, latitude, longitude) order.
Horizontal results are (point, depth); vertical results (siglay, point).
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from fvtools.interpolators.roms_interpolators import _horizontal_interpolate_blockwise

VARIABLES = {'temp': 'thetao', 'salinity': 'so', 'u': 'uo', 'v': 'vo', 'zeta': 'zos'}

def _fill_below_bottom(profile):
    """Copy deepest valid value downward; reject gaps inside a wet column.

    Parameters
    ----------
    profile : ndarray, shape (z, point), shallow first
    """
    a = np.ma.filled(np.ma.asarray(profile, dtype=np.float64), np.nan).copy()
    if a.ndim != 2:
        raise ValueError('Expected (depth, points)')

    good = np.isfinite(a)
    if not good[0].all():
        raise ValueError('CMEMS surface land/missing data in a selected interpolation corner')

    if np.any((~good[:-1]) & good[1:]):
        raise ValueError('CMEMS profile has an interior missing level; cannot fill it as seabed')

    for k in range(1, len(a)):
        bad = ~good[k]
        a[k, bad] = a[k - 1, bad]
    return a


def _horizontal(data, mask, ids, weights):
    """
    ROMS-style four-corner horizontal interpolation, in small query blocks.

    ``data``: (depth, latitude, longitude) or (latitude, longitude).
    Returns (point, depth) or (point,).
    """
    values = np.ma.filled(np.ma.asarray(data), np.nan)
    if values.ndim == 3:
        cloud = values[:, mask].T  # (wet point, depth)
    elif values.ndim == 2:
        cloud = values[mask]
    else:
        raise ValueError('Expected 2D or 3D CMEMS field')
    if not np.isfinite(cloud[ids]).all():
        if values.ndim == 3:
            cloud = _fill_below_bottom(cloud.T).T
        else:
            raise ValueError('CMEMS surface field has missing values in interpolation corners')
    return _horizontal_interpolate_blockwise(cloud, ids, weights)


def _vertical(values, matrices):
    """
    Apply existing FVCOM vertical coefficient matrices to (point, z) data.
    """
    i1, i2, w1, w2 = matrices
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError('Expected (point, z)')

    points = np.arange(values.shape[0])[None, :]
    z = values.T
    return z[i1, points] * w1 + z[i2, points] * w2


def _open_cmems(path):
    ds = xr.open_dataset(path, decode_times=True)
    missing = set(VARIABLES.values()) - set(ds.variables)
    missing_dims = {'time', 'depth', 'latitude', 'longitude'} - set(ds.coords)
    if missing or missing_dims:
        ds.close()
        raise ValueError(f'Expected CMEMS data: missing fields {sorted(missing)}, coordinates {sorted(missing_dims)}')

    for name, cmems_name in VARIABLES.items():
        expected = (('time', 'latitude', 'longitude') if name == 'zeta'
                    else ('time', 'depth', 'latitude', 'longitude'))

        if set(ds[cmems_name].dims) != set(expected):
            ds.close()
            raise ValueError(f'{cmems_name}: expected dimensions {expected}, got {ds[cmems_name].dims}')
    depth = np.asarray(ds.depth.values, dtype=float)

    if len(depth) < 2 or not np.isfinite(depth).all() or (np.diff(depth) <= 0).any() or depth[0] < 0:
        ds.close()
        raise ValueError('CMEMS depths must be positive, finite, shallow-to-deep')

    if ds.longitude.ndim != 1 or ds.latitude.ndim != 1:
        ds.close()
        raise ValueError('This CMEMS adapter expects rectilinear 1D latitude and longitude')

    ds = ds.sortby('latitude').sortby('longitude')
    return ds
