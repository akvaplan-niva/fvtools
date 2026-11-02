"""Add TPXO barotropic elevation and currents to an FVCOM restart.

Run *after* GLORYS restart interpolation.  Reuses the same harmonic
interpolator, UTide prediction, duplicate-application guard, and atomic
transaction as ``fvtools.setup.nesting.tpxo``.

TPXO U/V files contain transports (m2/s), converted to m/s using the
matching TPXO hu/hv, not FVCOM bathymetry. Tidal velocities are depth
uniform and therefore do not change the vertical shear of GLORYS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

from fvtools.setup.nesting import tpxo

DEFAULT_CONSTITUENTS = ('M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1')


def _restart_metadata(nc, currents):
    """Get a single, millisecond-accurate restart time and target locations."""
    if ('Itime' not in nc.variables or 'Itime2' not in nc.variables
            or len(nc['Itime']) != 1 or len(nc['Itime2']) != 1):
        raise ValueError('TPXO restart requires one Itime/Itime2 timestamp')
    day = int(nc['Itime'][0])
    msec = int(nc['Itime2'][0])
    if not 0 <= msec < 86_400_000:
        raise ValueError('Invalid restart Itime2; expected milliseconds in one day')
    t = np.array([day + msec / 86_400_000.0], dtype=np.float64)
    if 'zeta' not in nc.variables or nc['zeta'].dimensions != ('time', 'node'):
        raise ValueError('Restart must contain zeta(time,node)')
    if len(nc.dimensions['time']) != 1:
        raise ValueError('TPXO restart requires exactly one time record')
    for name in ('lon', 'lat') + (('lonc', 'latc') if currents else ()):
        if name not in nc.variables or nc[name].ndim != 1:
            raise ValueError(f'Restart is missing 1-D {name} coordinates')
    lon = np.asarray(nc['lon'][:], dtype=np.float64)
    lat = np.asarray(nc['lat'][:], dtype=np.float64)
    if lon.shape != lat.shape or lon.size != len(nc.dimensions['node']):
        raise ValueError('Restart node coordinates have invalid dimensions')
    lonc = latc = None
    if currents:
        lonc = np.asarray(nc['lonc'][:], dtype=np.float64)
        latc = np.asarray(nc['latc'][:], dtype=np.float64)
        if (lonc.shape != latc.shape or lonc.size != len(nc.dimensions['nele'])):
            raise ValueError('Restart element coordinates have invalid dimensions')
        for name, dims in (
            ('u', ('time', 'siglay', 'nele')),
            ('v', ('time', 'siglay', 'nele')),
            ('ua', ('time', 'nele')),
            ('va', ('time', 'nele')),
        ):
            if name not in nc.variables or nc[name].dimensions != dims:
                raise ValueError(f'Restart must contain {name}{dims} for tidal currents')
    if not all(np.isfinite(x).all() for x in (lon, lat) + ((lonc, latc) if currents else ())):
        raise ValueError('Restart contains nonfinite target coordinates')
    return t, lon, lat, lonc, latc


def _velocity_is_geographic(nc, latlon):
    """Use explicit restart metadata, never guess a projected velocity CRS."""
    declared = getattr(nc, 'glorys_velocity_coordinates', None)
    inferred = None
    if declared == 'geographic east/north':
        inferred = True
    elif declared == 'FVCOM projected x/y':
        inferred = False
    elif declared is not None:
        raise ValueError(f'Unrecognized glorys_velocity_coordinates={declared!r}')
    if latlon is None:
        if inferred is None:
            raise ValueError('Set latlon=True/False for tidal currents: restart velocity coordinates are not documented')
        return inferred
    if inferred is not None and bool(latlon) != inferred:
        raise ValueError('latlon disagrees with glorys_velocity_coordinates in the restart')
    return bool(latlon)


def _tpxo_depth_at_elements(grid_file, component, lon, lat):
    """Interpolate the relevant TPXO C-grid depth (hu/hv), in metres.

    Use named axes to avoid silently transposing square TPXO grids.
    """
    name = {'u': 'hu', 'v': 'hv'}[component]
    with xr.open_dataset(grid_file) as ds:
        lon_name, lat_name = f'lon_{component}', f'lat_{component}'
        if any(k not in ds.variables for k in (lon_name, lat_name, name)):
            raise ValueError(f'TPXO grid lacks {lon_name}/{lat_name}/{name}')
        source_lon, source_lat = tpxo._rectilinear_axes(
            ds[lon_name].values, ds[lat_name].values)
        da = ds[name]
        if da.ndim != 2:
            raise ValueError(f'TPXO {name} must have two spatial dimensions')
        # TPXO coordinate arrays can be 1-D or two separable 2-D axes.
        lon_var, lat_var = ds[lon_name], ds[lat_name]
        if lon_var.ndim == 1 and lat_var.ndim == 1:
            xdim, ydim = lon_var.dims[0], lat_var.dims[0]
        elif lon_var.ndim == lat_var.ndim == 2:
            # Determine which dimension actually varies with longitude.
            lon2 = np.asarray(lon_var.values)
            if np.allclose(lon2, lon2[0][None, :], equal_nan=True):
                xdim, ydim = lon_var.dims[1], lon_var.dims[0]
            else:
                xdim, ydim = lon_var.dims[0], lon_var.dims[1]
        else:
            raise ValueError('Incompatible TPXO longitude/latitude dimensions')
        if xdim == ydim or set(da.dims) != {xdim, ydim}:
            raise ValueError(f'Cannot match TPXO {name} dimensions to longitude/latitude')
        h = np.ma.filled(np.ma.asarray(da.transpose(ydim, xdim).values,
                                        dtype=np.float64), np.nan)
        if h.shape != (len(source_lat), len(source_lon)):
            raise ValueError(f'Incompatible TPXO {name} bathymetry shape')
        units = str(da.attrs.get('units', '')).lower().strip()
        if units in ('cm', 'centimeter', 'centimeters'):
            h *= 0.01
        elif units not in ('m', 'meter', 'meters'):
            raise ValueError(f'Unexpected TPXO {name} bathymetry units: {units!r}')
    ilon, ilat = np.argsort(source_lon), np.argsort(source_lat)
    source_lon, source_lat = source_lon[ilon], source_lat[ilat]
    h = h[np.ix_(ilat, ilon)]
    if (np.any(np.diff(source_lon) <= 0) or np.any(np.diff(source_lat) <= 0)):
        raise ValueError('TPXO bathymetry longitude/latitude coordinates are duplicated')
    x = np.asarray(lon, dtype=float)
    x = x + 360.0 * np.round((np.median(source_lon) - x) / 360.0)
    positions = np.column_stack((lat, x))
    depth = RegularGridInterpolator((source_lat, source_lon), h,
                                    bounds_error=True)(positions)
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError('TPXO depth is missing or non-positive at FVCOM elements')
    return depth


def _predict_one_time(harmonics, constituents, time_mjd, latitudes, *, chunk_size=10_000):
    """Vectorized, UTide-equivalent harmonic prediction for one FVCOM restart time.

    The standard eight astronomical constituents have no shallow-water
    dependencies. Apply UTide's exact satellite/nodal factors and astronomical
    arguments in chunks, avoiding one Python UTide reconstruct call per FVCOM
    node (an issue for meshes with hundreds of thousands of nodes).

    Fall back to the existing UTide reconstruction for nonstandard/shallow
    constituents. This preserves UTide's full shallow-water corrections.
    """
    h = np.asarray(harmonics, dtype=np.complex128)
    lat = np.asarray(latitudes, dtype=np.float64)
    t = np.asarray(time_mjd, dtype=np.float64)
    if t.shape != (1,) or lat.ndim != 1 or h.shape != (len(constituents), len(lat)):
        raise ValueError('Expect one time and harmonics (constituents, FVCOM points)')
    if not np.isfinite(h).all() or not np.isfinite(lat).all() or not np.isfinite(t).all():
        raise ValueError('Nonfinite TPXO coefficients, coordinates, or time')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')

    from utide.harmonics import sat, const
    from utide.astronomy import ut_astron

    names = [str(x).upper() for x in constituents]
    known = list(tpxo.ut_constants['const']['name'])
    indices = np.asarray([known.index(name) for name in names])
    shallow = ~np.ma.getmaskarray(np.ma.masked_invalid(const.ishallow))[indices]
    if np.any(shallow):
        return tpxo._predict_utide(h, constituents, t, lat)

    # UTide's epoch='1858-11-17' maps MJD to Python ordinal days.
    astro, _ = ut_astron(t + 678576.)
    satphase = np.exp(2j * np.pi * np.fmod(
        (sat.deldood @ astro[3:6] + sat.phcorr[:, None])[:, 0], 1.))
    V = np.fmod((const.doodson @ astro + const.semi[:, None])[indices, 0], 1.)
    astronomical = np.exp(2j * np.pi * V)
    iconst = np.asarray(sat.iconst, dtype=np.int64) - 1
    ilatfac = np.asarray(sat.ilatfac)
    ampl = np.asarray(sat.amprat, dtype=np.float64)

    selections = [np.flatnonzero(iconst == k) for k in indices]
    out = np.empty((1, len(lat)), dtype=np.float64)
    for start in range(0, len(lat), chunk_size):
        stop = min(start + chunk_size, len(lat))
        # UTide applies an equatorial 5-degree floor in its FUV routine.
        latitude = lat[start:stop].copy()
        near_equator = np.abs(latitude) < 5
        latitude[near_equator] = np.sign(latitude[near_equator]) * 5
        slat = np.sin(np.deg2rad(latitude))
        result = np.zeros(stop - start, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            factor1 = 0.36309 * (1. - 5. * slat**2) / slat
        factor2 = 2.59808 * slat
        for j, sel in enumerate(selections):
            if len(sel):
                factors = np.ones((len(sel), stop - start), dtype=np.float64)
                is_one, is_two = ilatfac[sel] == 1, ilatfac[sel] == 2
                factors[is_one] *= factor1[None, :]
                factors[is_two] *= factor2[None, :]
                # F*exp(i*U) from UTide FUV equals this complex sum.
                correction = 1. + np.sum(
                    (ampl[sel] * satphase[sel])[:, None] * factors, axis=0)
            else:
                correction = 1.
            result += np.real(h[j, start:stop] * correction * astronomical[j])
        out[0, start:stop] = result
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite UTide reconstruction')
    return out


def _rotate_currents(predictions, restart_lonc, restart_latc, fvcom_grid):
    """Rotate geographic TPXO E/N to FVCOM x/y using the same FVCOM method as GLORYS."""
    if fvcom_grid is None:
        raise ValueError('fvcom_grid is required for TPXO currents on projected FVCOM grids')
    # Lazy import so zeta-only corrections have no FVCOM grid dependency.
    from fvtools.grid.fvcom_grd import FVCOM_grid
    grid = FVCOM_grid(str(fvcom_grid))
    if (len(grid.xc) != len(restart_lonc)
            or not np.allclose(grid.lonc, restart_lonc, rtol=0, atol=1e-4)
            or not np.allclose(grid.latc, restart_latc, rtol=0, atol=1e-4)):
        raise ValueError('fvcom_grid element coordinates differ from the restart file')
    u, v = grid.rotate_vector_to_utm(
        predictions['u'], predictions['v'], np.asarray(grid.cell_utm_angle)[None, :])
    predictions['u'], predictions['v'] = u, v


def add_tpxo_tides_to_restart(
    restartfile, *, harmonics_files, constituents=DEFAULT_CONSTITUENTS,
    tpxo_bathymetry=None, fvcom_grid=None, latlon=None,
):
    """Add TPXO barotropic tides to an existing one-record FVCOM restart.

    ``harmonics_files`` contains ``'zeta'`` and optionally both ``'u'`` and
    ``'v'``. Transport files require ``tpxo_bathymetry`` (hu/hv). Tidal
    velocities are uniform through sigma and added to both u/v and ua/va.

    For projected FVCOM velocities, provide ``fvcom_grid`` to rotate E/N into
    FVCOM's x/y coordinates; ``latlon`` is inferred from the GLORYS restart
    metadata if present, otherwise it must be specified for currents.

    Atomic copy-on-write with a lock prevents a partial update or double add.
    The method does not modify temperature, salinity, or other restart state.
    """
    destination = Path(restartfile)
    if not destination.is_file():
        raise FileNotFoundError(destination)
    if not isinstance(harmonics_files, dict) or 'zeta' not in harmonics_files:
        raise ValueError("harmonics_files must contain 'zeta'")
    if set(harmonics_files) - {'zeta', 'u', 'v'}:
        raise ValueError('Only zeta, u and v TPXO components are supported')
    currents = 'u' in harmonics_files or 'v' in harmonics_files
    if currents and not {'u', 'v'} <= set(harmonics_files):
        raise ValueError('Provide BOTH u and v harmonics for a consistent tidal current vector')
    if currents and tpxo_bathymetry is None:
        raise ValueError('TPXO u/v transports require tpxo_bathymetry (hu and hv)')
    if not constituents or len(set(constituents)) != len(constituents):
        raise ValueError('Specify a nonempty sequence of unique TPXO constituents')

    with Dataset(destination, 'r') as nc:
        tpxo._assert_tpxo_not_applied(nc)
        time_mjd, lon, lat, lonc, latc = _restart_metadata(nc, currents)
        geographic = _velocity_is_geographic(nc, latlon) if currents else True

    predictions = {}
    for component in (('zeta', 'u', 'v') if currents else ('zeta',)):
        xx, yy = (lon, lat) if component == 'zeta' else (lonc, latc)
        harmonics = tpxo._load_harmonic_file(
            harmonics_files[component], component, xx, yy, constituents)
        if component != 'zeta':
            depth = _tpxo_depth_at_elements(
                tpxo_bathymetry, component, xx, yy)
            harmonics = harmonics / depth[None, :]
        predicted = _predict_one_time(harmonics, constituents, time_mjd, yy)
        if predicted.shape != (1, len(yy)) or not np.isfinite(predicted).all():
            raise ValueError(f'Invalid TPXO {component} prediction at restart time')
        predictions[component] = predicted

    if currents and not geographic:
        _rotate_currents(predictions, lonc, latc, fvcom_grid)

    # The existing TPXO transaction re-checks the marker under an exclusive
    # lock and atomically replaces a *copy* of the unchanged source on success.
    with tpxo._tpxo_transaction(destination) as staging:
        with Dataset(staging, 'r+') as nc:
            required = ('zeta', 'u', 'v', 'ua', 'va') if currents else ('zeta',)
            for name in required:
                original = np.ma.filled(np.ma.asarray(nc[name][:]), np.nan)
                if not np.isfinite(original).all():
                    raise ValueError(f'Nonfinite {name} in original FVCOM restart')
            tpxo._write_tpxo_predictions(nc, predictions, constituents)
            for name in required:
                updated = np.ma.filled(np.ma.asarray(nc[name][:]), np.nan)
                if not np.isfinite(updated).all():
                    raise ValueError(f'Nonfinite {name} after TPXO augmentation')
            nc.tpxo_application = 'FVCOM restart: one time, elevation and optional barotropic currents'
            if currents:
                nc.tpxo_current_coordinates = ('geographic east/north' if geographic
                                               else 'FVCOM projected x/y')
    return str(destination)


main = add_tpxo_tides_to_restart
