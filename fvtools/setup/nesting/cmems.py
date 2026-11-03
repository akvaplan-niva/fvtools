"""CMEMS (GLORYS) z-level fields to FVCOM nesting NetCDF.

Offline/local NetCDF first. The same structured four-corner bilinear and
``calc_interp_matrices`` kernels used in ROMS nesting are reused. No CMEMS
network access takes place in this module.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4
import os
import numpy as np
from netCDF4 import Dataset, date2num

from fvtools.interpolators.glorys_fields import (
    VARIABLES, _fill_below_bottom, _horizontal, _vertical, _open_cmems,
)
from fvtools.interpolators.structured_bilinear import structured_bilinear_weights
from fvtools.interpolators.vertical_interpolation import calc_interp_matrices
from fvtools.interpolators.glorys_barotropic import (
    depth_average_glorys, correct_fvcom_barotropic, fvcom_target_depths,
)

from fvtools.grid.fvcom_grd import FVCOM_grid, NEST_grid
from fvtools.grid.glorys_grid import GLORYS

from fvtools.setup.nesting.roms import create_nc_forcing_file
from fvtools.setup.nesting.matchtopo import MatchTopo

def main(
        fvcom_grd, nest_grd, cmems_file, outfile, *,
        start_time=None, stop_time=None, latlon=True,
        weights=(2.5e-4, 2.5e-5)
        ):
    """
    Write FVCOM nesting from CMEMS native time samples.

    The FVCOM grid is matched to GLORYS bathymetry (and sigma recomputed)
    before interpolation weights are prepared.

    Notes
    -----
    MatchTopo also updates M.npy and the FVCOM input bathymetry file the first time.

    start_time, stop_time are optional - the script will by default interpolate all timesteps
    to the FVCOM nesting file.
    """
    print('Interpolating CMEMS/GLORYS data to the open boundary')
    ds = _open_cmems(cmems_file)
    destination = Path(outfile)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.{uuid4().hex}.partial')
    try:
        CMEMS = GLORYS(ds)
        if CMEMS.bathymetry_source != 'deptho':
            raise ValueError('GLORYS nesting requires static deptho in the source NetCDF')
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
        wet, node_horizontal, cell_horizontal, node_vertical, cell_vertical = _prepare_geometry(M, NEST, CMEMS)

        python_times = output_times.astype('datetime64[us]').astype(object).tolist()
        mjd = date2num(python_times, units='days since 1858-11-17 00:00:00', calendar='standard')
        create_nc_forcing_file(str(temporary), NEST, 'CMEMS', mjd, latlon, M.reference)

        with Dataset(str(temporary), mode='r+') as out:
            out.source_dataset = str(cmems_file)
            for dst_i, src_i in enumerate(chosen):
                result = _read_and_interpolate(
                    ds, int(src_i), wet, 
                    node_horizontal, cell_horizontal,
                    node_vertical, cell_vertical, 
                    NEST, latlon=latlon
                    )
                
                for key, value in result.items():
                    out.variables[key][dst_i, ...] = value
                out.variables['weight_node'][dst_i, :] = NEST.weight_node
                out.variables['weight_cell'][dst_i, :] = NEST.weight_cell
                out.variables['hyw'][dst_i, :, :] = 0.0

        os.replace(temporary, destination)
    finally:
        ds.close()
        temporary.unlink(missing_ok=True)

def _native_time_indices(source_times, *, start_time=None, stop_time=None):
    """
    Select original CMEMS samples without interpolation/resampling.
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


def _prepare_geometry(M, NEST, CMEMS):
    """
    Reuse the GLORYS grid geometry and four-corner ROMS-style weights.
    """
    wet = (~np.asarray(CMEMS.land_mask, dtype=bool) & ~np.asarray(CMEMS.land_3d[0], dtype=bool))
    if wet.shape != CMEMS.lon.shape or not wet.any():
        raise ValueError('GLORYS grid has no valid surface ocean cells')
    xx, yy = M.Proj(CMEMS.lon, CMEMS.lat)
    xy_source = np.column_stack((xx[wet], yy[wet]))
    nodes = np.column_stack((NEST.x, NEST.y))
    cells = np.column_stack((NEST.xc, NEST.yc))
    node_indices, node_weights = structured_bilinear_weights(xy_source, wet, nodes)
    cell_indices, cell_weights = structured_bilinear_weights(xy_source, wet, cells)

    depths = np.asarray(CMEMS.depth, dtype=np.float64)

    def matrices_for_zeta(sigma, h):
        """Cache time-independent source depths; rebuild vertical weights per time."""
        h = np.asarray(h, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        source = np.broadcast_to(depths[:, None], (len(depths), len(h)))

        def at_zeta(zeta):
            # GLORYS has fixed z-levels, but FVCOM sigma midpoints move with
            # the interpolated free surface even if the bathymetry matches.
            target = fvcom_target_depths(sigma, h, zeta)
            return calc_interp_matrices(source, target)

        return at_zeta

    hn = np.asarray(NEST.h)
    hc = np.mean(hn[NEST.tri], axis=1)
    return (wet, (node_indices, node_weights), (cell_indices, cell_weights),
            matrices_for_zeta(NEST.siglay, hn),
            matrices_for_zeta(NEST.siglay_center, hc))


def _read_and_interpolate(ds, i, mask, nm, cm, nv, cv, NEST, *, latlon):
    """
    Read one GLORYS timestep and interpolate all seven FVCOM fields.

    Barotropic ua/va are calculated by integrating GLORYS uo/vo over the
    GLORYS water column before the horizontal interpolation to FVCOM cells.
    The interpolated FVCOM 3-D currents are subsequently barotropically
    corrected to match ua/va, preserving their vertical shear. Vertical
    interpolation matrices are computed using this timestep's interpolated
    FVCOM sea surface, for both nodes and cells.
    """
    def field(key):
        name = VARIABLES[key]
        selection = ds[name].isel(time=i)
        return np.ma.filled(np.ma.asarray(selection.transpose(*(
            ('latitude', 'longitude') if key == 'zeta'
            else ('depth', 'latitude', 'longitude'))).values), np.nan)

    zeta_source = field('zeta')
    u_source = field('u')
    v_source = field('v')
    if 'deptho' not in ds:
        raise ValueError('GLORYS barotropic velocity requires static deptho in the source NetCDF')
    deptho = np.ma.filled(np.ma.asarray(
        ds['deptho'].transpose('latitude', 'longitude').values), np.nan)
    bar_u, bar_v = depth_average_glorys(
        u_source, v_source, ds.depth.values, deptho, zeta_source,
        wet_mask=mask,
    )

    zeta = _horizontal(zeta_source, mask, *nm)
    # Interpolate surface elevation only once. At element centers FVCOM uses
    # the mean of its three node values. Recompute vertical weights for this
    # timestep, while retaining all horizontal interpolation coefficients.
    node_matrices = nv(zeta)
    zeta_cell = np.mean(zeta[np.asarray(NEST.tri, dtype=int)], axis=1)
    cell_matrices = cv(zeta_cell)
    temp = _vertical(_horizontal(field('temp'), mask, *nm), node_matrices)
    salt = _vertical(_horizontal(field('salinity'), mask, *nm), node_matrices)
    u = _vertical(_horizontal(u_source, mask, *cm), cell_matrices)
    v = _vertical(_horizontal(v_source, mask, *cm), cell_matrices)
    ua = _horizontal(bar_u, mask, *cm)
    va = _horizontal(bar_v, mask, *cm)

    if not latlon:
        angle = NEST.cell_utm_angle
        u, v = NEST.rotate_vector_to_utm(u, v, angle[None, :])
        ua, va = NEST.rotate_vector_to_utm(ua, va, angle)

    # Correct only the barotropic mode. The source-integrated ua/va remain
    # unchanged; vertically averaged FVCOM u/v now match them exactly.
    # Use local sigma layer thicknesses, never an unweighted layer mean.
    u, v = correct_fvcom_barotropic(u, v, ua, va, NEST.siglev_center)

    result = dict(zeta=zeta, temp=temp, salinity=salt, u=u, v=v, ua=ua, va=va)
    for key, arr in result.items():
        if not np.isfinite(arr).all():
            raise ValueError(f'Non-finite {key} after interpolation at CMEMS time index {i}')
    return result