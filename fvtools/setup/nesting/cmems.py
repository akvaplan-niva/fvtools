"""CMEMS (GLORYS) z-level fields to FVCOM nesting NetCDF.

Offline/local NetCDF first.  The same structured four-corner bilinear and
``calc_interp_matrices`` kernels used in ROMS nesting are reused. No CMEMS
network access takes place in this module.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4
import os
import numpy as np
import xarray as xr
from netCDF4 import Dataset, date2num


from fvtools.interpolators.structured_bilinear import structured_bilinear_weights
from fvtools.interpolators.vertical_interpolation import calc_interp_matrices
from fvtools.interpolators.roms_interpolators import _horizontal_interpolate_blockwise

from fvtools.grid.fvcom_grd import FVCOM_grid, NEST_grid
from fvtools.grid.glorys_grid import GLORYS

from fvtools.setup.nesting.roms import create_nc_forcing_file
from fvtools.setup.nesting.matchtopo import MatchTopo

VARIABLES = {'temp': 'thetao', 'salinity': 'so', 'u': 'uo', 'v': 'vo', 'zeta': 'zos'}


def _fill_below_bottom(profile):
    """
    Copy deepest valid value downward; reject gaps inside a wet column.

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
    
    # A wet value below a dry level is not a simple bottom mask.
    if np.any((~good[:-1]) & good[1:]):
        raise ValueError('CMEMS profile has an interior missing level; cannot fill it as seabed')
    
    for k in range(1, len(a)):
        bad = ~good[k]
        a[k, bad] = a[k - 1, bad]
    return a


def _horizontal(data, mask, ids, weights):
    """
    ROMS-style four-corner interpolation, bounded by small query blocks.

    ``data``: (depth, latitude, longitude), or (latitude, longitude).
    Returns (depth, target) or (target,).
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

def _native_time_indices(source_times, *, start_time=None, stop_time=None):
    """Select original CMEMS samples without interpolation/resampling.

    Returns indices into ``ds.time``, preserving the original timestamps.
    """
    times = np.asarray(source_times, dtype='datetime64[ns]')
    if times.ndim != 1 or not len(times) or np.isnat(times).any():
        raise ValueError('CMEMS time must be a nonempty, valid 1D coordinate')
    if np.any(times[1:] <= times[:-1]):
        raise ValueError('CMEMS times must be strictly increasing')

    first = times[0] if start_time is None else np.datetime64(start_time, 'ns')
    last = times[-1] if stop_time is None else np.datetime64(stop_time, 'ns')
    if np.isnat(first) or np.isnat(last) or first > last:
        raise ValueError('Invalid CMEMS time interval')
    if first < times[0] or last > times[-1]:
        raise ValueError('Requested period must be within CMEMS time coverage')
    return np.flatnonzero((times >= first) & (times <= last))





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
   
    # Sorting here is cheap (index-only) and ensures spatial array orientation.
    ds = ds.sortby('latitude').sortby('longitude')
    return ds


def _prepare_geometry(M, NEST, CMEMS):
    """Reuse the GLORYS grid geometry and four-corner ROMS-style weights.

    Source mask, coordinates, and depth levels come from the same GLORYS
    instance as MatchTopo, with exactly the same ordering as the open dataset.
    """
    wet = (~np.asarray(CMEMS.land_mask, dtype=bool) &
           ~np.asarray(CMEMS.land_3d[0], dtype=bool))
    if wet.shape != CMEMS.lon.shape or not wet.any():
        raise ValueError('GLORYS grid has no valid surface ocean cells')
    xx, yy = M.Proj(CMEMS.lon, CMEMS.lat)
    xy_source = np.column_stack((xx[wet], yy[wet]))
    nodes = np.column_stack((NEST.x, NEST.y))
    cells = np.column_stack((NEST.xc, NEST.yc))
    ni, nw = structured_bilinear_weights(xy_source, wet, nodes)
    ci, cw = structured_bilinear_weights(xy_source, wet, cells)

    depths = np.asarray(CMEMS.depth, dtype=np.float64)
    def matrices(sig, h):
        target = np.abs(np.asarray(sig).T) * np.asarray(h)[None, :]
        source = np.broadcast_to(depths[:, None], (len(depths), target.shape[1]))
        return calc_interp_matrices(source, target)

    hn = np.asarray(NEST.h)
    hc = np.mean(hn[NEST.tri], axis=1)
    return wet, (ni, nw), (ci, cw), matrices(NEST.siglay, hn), matrices(NEST.siglay_center, hc)


def _read_and_interpolate(ds, i, mask, nm, cm, nv, cv, NEST, *, latlon):
    """
    Read each variable ONCE for the requested native CMEMS time step.
    """
    def field(key):
        name = VARIABLES[key]
        selection = ds[name].isel(time=i)
        return np.asarray(selection.transpose(*(
            ('latitude', 'longitude') if key == 'zeta'
            else ('depth', 'latitude', 'longitude'))).values)

    zeta = _horizontal(field('zeta'), mask, *nm)
    temp = _vertical(_horizontal(field('temp'), mask, *nm), nv)
    salt = _vertical(_horizontal(field('salinity'), mask, *nm), nv)
    u = _vertical(_horizontal(field('u'), mask, *cm), cv)
    v = _vertical(_horizontal(field('v'), mask, *cm), cv)

    if not latlon:
        angle = NEST.cell_utm_angle
        u, v = NEST.rotate_vector_to_utm(u, v, angle[None, :])

    # FVCOM sigma layers are surface-first. Average over actual sigma thickness.
    thickness = np.abs(np.diff(NEST.siglev_center, axis=1)).T
    thickness /= thickness.sum(axis=0, keepdims=True)
    ua = (u * thickness).sum(axis=0)
    va = (v * thickness).sum(axis=0)
    result = dict(zeta=zeta, temp=temp, salinity=salt, u=u, v=v, ua=ua, va=va)
    for key, arr in result.items():
        if not np.isfinite(arr).all():
            raise ValueError(f'Non-finite {key} after interpolation at CMEMS time index {i}')
    return result


def main(fvcom_grd, nest_grd, cmems_file, outfile, *,
         start_time=None, stop_time=None, latlon=True,
         weights=(2.5e-4, 2.5e-5)):
    """Write FVCOM nesting from CMEMS native time samples, without resampling.

    The FVCOM grid is matched to GLORYS bathymetry (and sigma recomputed)
    before interpolation weights are prepared. FVCOM handles time interpolation.
    Source NetCDF fields are accessed one time record at a time.

    Notes
    -----
    MatchTopo also updates M.npy and the FVCOM input bathymetry file, as in
    ROMS nesting. Use an original, not previously ROMS-matched, FVCOM grid.
    """
    from fvtools.grid.fvcom_grd import FVCOM_grid, NEST_grid
    from fvtools.grid.glorys_grid import GLORYS
    from fvtools.setup.nesting.matchtopo import MatchTopo
    from fvtools.setup.nesting.roms import create_nc_forcing_file
    from netCDF4 import Dataset, date2num

    ds = _open_cmems(cmems_file)
    destination = Path(outfile)
    temporary = destination.with_name(
        f'.{destination.name}.{os.getpid()}.{uuid4().hex}.partial')
    try:
        # Construct GLORYS *from the sorted open dataset*, so its source-mask
        # indexing always matches the (latitude,longitude) NetCDF arrays.
        CMEMS = GLORYS(ds)
        M = FVCOM_grid(str(fvcom_grd))
        NEST = NEST_grid(str(nest_grd), M, proj=M.reference)
        NEST.calcWeights(M, w1=max(weights), w2=min(weights))

        print('\nMatch GLORYS bathymetry and FVCOM sigma in the nesting zone\n---')
        match = MatchTopo(CMEMS, M, NEST, latlon)
        M, NEST = match.add_GLORYS_bathymetry_to_FVCOM_and_NEST()

        times = np.asarray(ds.time.values, dtype='datetime64[ns]')
        chosen = _native_time_indices(times, start_time=start_time,
                                      stop_time=stop_time)
        if not len(chosen):
            raise ValueError('No native CMEMS timestamps within the requested period')
        output_times = times[chosen]
        wet, nm, cm, nv, cv = _prepare_geometry(M, NEST, CMEMS)

        # Preserve the original CMEMS timestamps in FVCOM's MJD encoding.
        # Conversion via datetime64[us] retains sub-second precision.
        python_times = output_times.astype('datetime64[us]').astype(object).tolist()
        mjd = date2num(python_times,
                       units='days since 1858-11-17 00:00:00',
                       calendar='standard')
        create_nc_forcing_file(str(temporary), NEST, 'CMEMS', mjd,
                               latlon, M.reference)

        with Dataset(str(temporary), mode='r+') as out:
            out.source_dataset = str(cmems_file)
            out.vertical_interpolation = (
                'fvtools.interpolators.vertical_interpolation.calc_interp_matrices')
            out.temporal_interpolation = 'none - FVCOM interpolates in time'
            for dst_i, src_i in enumerate(chosen):
                result = _read_and_interpolate(ds, int(src_i), wet,
                                               nm, cm, nv, cv, NEST,
                                               latlon=latlon)
                for key, value in result.items():
                    out.variables[key][dst_i, ...] = value
                out.variables['weight_node'][dst_i, :] = NEST.weight_node
                out.variables['weight_cell'][dst_i, :] = NEST.weight_cell
                out.variables['hyw'][dst_i, :, :] = 0.0

        os.replace(temporary, destination)
        return str(destination)
    finally:
        ds.close()
        temporary.unlink(missing_ok=True)
