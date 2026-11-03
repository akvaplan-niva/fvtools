"""Interpolate one local GLORYS timestep into an FVCOM restart file.

Uses the same horizontal and vertical numerical kernels as CMEMS nesting.
An existing restart is copied and replaced atomically after *all* requested
fields have been validated; unrelated restart variables are not changed.
"""

from __future__ import annotations

from contextlib import chdir
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil

import numpy as np
from netCDF4 import Dataset, num2date
from scipy.spatial import cKDTree

from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.glorys_grid import GLORYS
from fvtools.interpolators.structured_bilinear import structured_bilinear_weights
from fvtools.interpolators.vertical_interpolation import calc_interp_matrices
from fvtools.interpolators.glorys_barotropic import (
    depth_average_glorys, correct_fvcom_barotropic, fvcom_target_depths,
)
from ._helpers.empty_initial import make_initial_file
from fvtools.setup.nesting.cmems import (
    VARIABLES, _horizontal, _open_cmems, _vertical,
)


def main(restartfile=None, restart_date=None, fvcom_grid='M.npy',
         glorys_file='glorys.nc', uv=False, latlon=False, *,
         time_method='exact', max_time_difference_hours=12,
         coastal_max_distance_m=50_000, chunk_size=20_000,
         overwrite=False):
    """
    Initialize an FVCOM restart from a local, combined GLORYS NetCDF.

    Parameters
    ----------
    restartfile : path, optional
        Existing restart to update. If ``restart_date`` is specified, output
        path of a *new* restart (defaults to ``<casename>_initial.nc``).
    restart_date : str, optional
        ``YYYY-MM-DD-HH``; create the restart with ``make_initial_file``.
        Otherwise read the existing restart's exact time.
    glorys_file : path
        Combined daily CMEMS NetCDF with thetao, so, uo, vo, zos, deptho.
    uv : bool
        Include u/v/ua/va (cells). If false only node-based zeta/temp/salinity.
    latlon : bool
        If false rotate east/north velocities into FVCOM projected x/y.
    time_method : {'exact', 'nearest'}
        Default exact. For daily-mean GLORYS, explicitly choose ``nearest``
        with a suitable max_time_difference_hours (default 12).
    coastal_max_distance_m : float
        Maximum nearest-wet fallback distance where all source corners are dry.
    overwrite : bool
        Permit replacing an existing output when *creating* a new restart.
        Updating an explicitly supplied existing restart is always permitted.

    Returns the output restart path. A failed interpolation never alters it.
    """
    print('\nInterpolate CMEMS/Glorys oceanographic properties to the FVCOM grid')
    if restart_date is None and restartfile is None:
        raise ValueError('Specify restartfile or restart_date')
    
    grid = FVCOM_grid(str(fvcom_grid))
    creating = restart_date is not None
    if creating:
        datetime.strptime(restart_date, '%Y-%m-%d-%H')
    destination = Path(restartfile or f'{grid.casename}_initial.nc').resolve()

    # Make sure that the routine has been given sufficient information to make a new hotstart for FVCOM
    # ----
    if not creating and not destination.is_file():
        raise FileNotFoundError(destination)
    
    if creating and destination.exists() and not overwrite:
        raise FileExistsError(f'Restart already exists: {destination}; use overwrite=True')
    
    if not destination.parent.is_dir():
        raise FileNotFoundError(f'Output directory does not exist: {destination.parent}')

    with _open_cmems(glorys_file) as ds:
        source = GLORYS(ds)
        if source.bathymetry_source != 'deptho':
            raise ValueError('GLORYS restart requires static deptho in combined file')
        with TemporaryDirectory(prefix='.glorys_restart_', dir=destination.parent) as temp:
            workdir = Path(temp)
            if creating:
                # Isolated cwd prevents make_initial_file from overwriting any pre-existing original or leaving stray files on failure.
                with chdir(workdir):
                    initial = make_initial_file(grid, restart_date, obc_type=3)
                staged = workdir / Path(initial).name
            else:
                staged = workdir / 'staged_restart.nc'
                shutil.copy2(destination, staged)

            with Dataset(staged, 'r+') as nc:
                requested_time = _restart_time(nc)
                selected = _source_time_index(
                    ds.time.values, requested_time, method=time_method,
                    max_hours=max_time_difference_hours)
                _write_restart(nc, grid, source, ds, selected, uv=uv,
                               latlon=latlon, chunk_size=chunk_size,
                               coastal_max_distance_m=coastal_max_distance_m)
                nc.glorys_source_file = str(glorys_file)
                nc.glorys_source_time = np.datetime_as_string(
                    np.datetime64(ds.time.values[selected]), unit='s')
                nc.glorys_restart_time = np.datetime_as_string(requested_time, unit='s')
                nc.glorys_velocity_coordinates = 'geographic east/north' if latlon else 'FVCOM projected x/y'
                nc.glorys_interpolation = 'structured bilinear (wet-corner correction), sigma-to-z via calc_interp_matrices'
                if uv:
                    nc.glorys_barotropic_method = (
                        'GLORYS z-level piecewise-linear depth integral, '
                        'using deptho and zos before horizontal interpolation; '
                        'FVCOM sigma u/v corrected to match ua/va without changing shear'
                    )

            # Atomic replacement on the same filesystem: never publish a
            # partially interpolated or unvalidated restart.
            os.replace(staged, destination)

def _restart_time(nc):
    """Read FVCOM time with millisecond accuracy (the ``time`` field is f32)."""
    if ('Itime' in nc.variables and 'Itime2' in nc.variables
            and len(nc['Itime']) and len(nc['Itime2'])):
        days = int(nc['Itime'][0])
        msecs = int(nc['Itime2'][0])
        return (np.datetime64('1858-11-17T00:00:00', 'ms')
                + np.timedelta64(days, 'D')
                + np.timedelta64(msecs, 'ms'))
    if 'time' not in nc.variables or not len(nc['time']):
        raise ValueError('FVCOM restart has no valid time coordinate')
    value = num2date(nc['time'][0], units=nc['time'].units,
                     only_use_cftime_datetimes=False)
    return np.datetime64(value.strftime('%Y-%m-%dT%H:%M:%S'), 'ms')


def _source_time_index(times, target, *, method='exact', max_hours=12):
    """Return a GLORYS index without silently interpolating daily means in time."""
    times = np.asarray(times, dtype='datetime64[ms]')
    target = np.datetime64(target, 'ms')
    if times.ndim != 1 or not len(times) or np.isnat(times).any():
        raise ValueError('GLORYS has an empty or invalid time coordinate')
    if np.any(times[1:] <= times[:-1]):
        raise ValueError('GLORYS time must be strictly increasing')
    if method not in ('exact', 'nearest'):
        raise ValueError("time_method must be 'exact' or 'nearest'")
    if not np.isfinite(max_hours) or max_hours < 0:
        raise ValueError('max_time_difference_hours must be nonnegative and finite')

    distances = np.abs(times - target)
    i = int(np.argmin(distances))
    if method == 'exact' and distances[i] != np.timedelta64(0, 'ms'):
        raise ValueError(f'No exact GLORYS sample for restart {target}; '
                         'use time_method="nearest" explicitly for daily mean data')
    if method == 'nearest' and distances[i] > np.timedelta64(
            int(round(max_hours * 3_600_000)), 'ms'):
        raise ValueError(f'Nearest GLORYS record {times[i]} is too far from restart {target}')
    return i


def _coastal_weights(glorys, model, targets, *, max_distance_m):
    """Projected structured bilinear weights, correcting GLORYS land corners.

    Like the ROMS restart code, renormalize available wet corners and use a
    nearest wet point only when all four corners are dry. Refuse any such
    extrapolation beyond ``max_distance_m``. Targets outside the downloaded
    GLORYS rectangle are *never* extrapolated.
    """
    if not np.isfinite(max_distance_m) or max_distance_m < 0:
        raise ValueError('coastal_max_distance_m must be finite and nonnegative')
    wet = (~np.asarray(glorys.land_mask, dtype=bool)
           & ~np.asarray(glorys.land_3d[0], dtype=bool))
    if wet.shape != glorys.lon.shape or not wet.any():
        raise ValueError('No GLORYS wet cells for restart interpolation')
    x, y = model.Proj(glorys.lon, glorys.lat)
    source = np.column_stack((np.asarray(x).ravel(), np.asarray(y).ravel()))
    targets = np.asarray(targets, dtype=float)
    full_ids, weights = structured_bilinear_weights(
        source, np.ones(wet.shape, dtype=bool), targets)

    corner_wet = wet.ravel()[full_ids]
    weights = np.where(corner_wet, weights, 0.)
    sums = weights.sum(axis=1)
    covered = sums > 1e-14
    weights[covered] /= sums[covered, None]

    lookup = np.full(wet.size, -1, dtype=np.int64)
    lookup[np.flatnonzero(wet.ravel())] = np.arange(wet.sum())
    ids = lookup[full_ids]
    ids[ids < 0] = 0  # dry source corners carry zero weight

    missing = ~covered
    if np.any(missing):
        tree = cKDTree(source[wet.ravel()])
        distances, nearest = tree.query(targets[missing], workers=1)
        if np.any(distances > max_distance_m):
            raise ValueError(
                f'{np.count_nonzero(distances > max_distance_m)} FVCOM targets '
                f'are farther than {max_distance_m:g} m from GLORYS wet points')
        ids[missing] = nearest[:, None]
        weights[missing] = 0.25
    return wet, ids, weights


def _target_depths(sigma, depth, zeta):
    """Compatibility alias for the shared FVCOM-to-GLORYS depth formula."""
    return fvcom_target_depths(sigma, depth, zeta)


def _interpolate_profile(field, mask, ids, weights, depths, target_depth):
    value = _horizontal(field, mask, ids, weights)
    source_depth = np.broadcast_to(np.asarray(depths)[:, None],
                                   (len(depths), value.shape[0]))
    matrices = calc_interp_matrices(source_depth, target_depth)
    out = _vertical(value, matrices)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite GLORYS restart profile after interpolation')
    return out


def _validate_restart(nc, grid, uv):
    nnode, ncell = len(grid.x), len(grid.xc)
    nsigma = grid.siglay.shape[1]
    required = {'zeta': ('time', 'node'),
                'temp': ('time', 'siglay', 'node'),
                'salinity': ('time', 'siglay', 'node')}
    if uv:
        required.update({
            'u': ('time', 'siglay', 'nele'),
            'v': ('time', 'siglay', 'nele'),
            'ua': ('time', 'nele'), 'va': ('time', 'nele'),
        })
    for name, dims in required.items():
        if name not in nc.variables or nc[name].dimensions != dims:
            raise ValueError(f'FVCOM restart missing {name} or dimensions {dims}')
    for dim, length in [('node', nnode), ('nele', ncell), ('siglay', nsigma)]:
        if dim not in nc.dimensions or len(nc.dimensions[dim]) != length:
            raise ValueError(f'FVCOM restart {dim} dimension does not match the grid')
    if 'h' in nc.variables:
        bathy = np.ma.filled(np.ma.asarray(nc['h'][:]), np.nan)
        if not np.allclose(bathy, grid.h, rtol=0, atol=1e-3):
            raise ValueError('Restart bathymetry differs from FVCOM_grid; use the matching grid')
    return tuple(required)


def _read_field(ds, name, index):
    dims = ('latitude', 'longitude') if name == 'zeta' else (
        'depth', 'latitude', 'longitude')
    return np.ma.filled(np.ma.asarray(
        ds[VARIABLES[name]].isel(time=index).transpose(*dims).values,
        dtype=np.float64), np.nan)


def _write_restart(nc, grid, source, ds, time_index, *, uv, latlon,
                   chunk_size, coastal_max_distance_m):
    written = _validate_restart(nc, grid, uv)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')

    node_targets = np.column_stack((grid.x, grid.y))
    node_mask, node_ids, node_w = _coastal_weights(
        source, grid, node_targets, max_distance_m=coastal_max_distance_m)
    cell_data = None
    if uv:
        cell_targets = np.column_stack((grid.xc, grid.yc))
        cell_mask, cell_ids, cell_w = _coastal_weights(
            source, grid, cell_targets, max_distance_m=coastal_max_distance_m)
        cell_data = (cell_mask, cell_ids, cell_w)

    # Read each field just once from local GLORYS NetCDF.
    values = {name: _read_field(ds, name, time_index) for name in
              (('zeta', 'temp', 'salinity', 'u', 'v') if uv else
               ('zeta', 'temp', 'salinity'))}

    zeta = np.empty(len(grid.x), dtype=np.float64)
    for first in range(0, len(zeta), chunk_size):
        last = min(first + chunk_size, len(zeta))
        zeta[first:last] = _horizontal(
            values['zeta'], node_mask, node_ids[first:last], node_w[first:last])
    if not np.isfinite(zeta).all():
        raise ValueError('Nonfinite GLORYS zeta in restart')
    nc['zeta'][0, :] = zeta

    for first in range(0, len(zeta), chunk_size):
        last = min(first + chunk_size, len(zeta))
        target = _target_depths(grid.siglay[first:last],
                                np.asarray(grid.h)[first:last], zeta[first:last])
        for name in ('temp', 'salinity'):
            nc[name][0, :, first:last] = _interpolate_profile(
                values[name], node_mask, node_ids[first:last],
                node_w[first:last], source.depth, target)

    if uv:
        mask, ids, weights = cell_data
        tri = np.asarray(grid.tri, dtype=int)
        h_cell = np.mean(np.asarray(grid.h)[tri], axis=1)
        z_cell = np.mean(zeta[tri], axis=1)
        # Compute barotropic velocity directly from the GLORYS z profiles,
        # then interpolate horizontally. Never average FVCOM sigma profiles.
        source_ua, source_va = depth_average_glorys(
            values['u'], values['v'], source.depth, source.h_rho,
            values['zeta'], wet_mask=mask,
        )
        angles = np.asarray(grid.cell_utm_angle) if not latlon else None
        for first in range(0, len(grid.xc), chunk_size):
            last = min(first + chunk_size, len(grid.xc))
            target = _target_depths(
                grid.siglay_center[first:last], h_cell[first:last], z_cell[first:last])
            u = _interpolate_profile(values['u'], mask, ids[first:last],
                                     weights[first:last], source.depth, target)
            v = _interpolate_profile(values['v'], mask, ids[first:last],
                                     weights[first:last], source.depth, target)
            if not latlon:
                u, v = grid.rotate_vector_to_utm(u, v, angles[None, first:last])
            if not np.isfinite(u).all() or not np.isfinite(v).all():
                raise ValueError('Nonfinite rotated GLORYS currents')
            ua = _horizontal(source_ua, mask, ids[first:last], weights[first:last])
            va = _horizontal(source_va, mask, ids[first:last], weights[first:last])
            if not latlon:
                ua, va = grid.rotate_vector_to_utm(ua, va, angles[first:last])
            if not np.isfinite(ua).all() or not np.isfinite(va).all():
                raise ValueError('Nonfinite GLORYS depth-averaged currents')
            # Do this after rotation, in the same coordinate system as ua/va.
            # Sigma interfaces provide the actual relative layer thicknesses.
            u, v = correct_fvcom_barotropic(
                u, v, ua, va, grid.siglev_center[first:last],
            )
            for name, data in (('u', u), ('v', v), ('ua', ua), ('va', va)):
                nc[name][0, ..., first:last] = data

    # Read written data again from the temporary restart (including f32 cast).
    for name in written:
        var = nc[name]
        for first in range(0, var.shape[-1], chunk_size):
            last = min(first + chunk_size, var.shape[-1])
            data = np.ma.filled(np.ma.asarray(var[0, ..., first:last]), np.nan)
            if not np.isfinite(data).all():
                raise ValueError(f'Nonfinite {name} in staged FVCOM restart')