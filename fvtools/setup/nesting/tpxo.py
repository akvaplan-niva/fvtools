"""
Interpolate TPXO harmonics and augment a CMEMS FVCOM nesting file.

Inspired by PyFVCOM's add_nests_harmonics.
The harmonic interpolation uses REAL/IMAGINARY components, never angles.
The UTide dependency is optional and needed only for prediction.
"""
from __future__ import annotations
from contextlib import contextmanager as _tpxo_contextmanager

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from utide import reconstruct, ut_constants
from utide.utilities import Bunch

_COMPONENTS = {
    'zeta': ('hRe', 'hIm', 'lon_z', 'lat_z'),
    'u': ('URe', 'UIm', 'lon_u', 'lat_u'),
    'v': ('VRe', 'VIm', 'lon_v', 'lat_v'),
}

def _constituents(con):
    """Decode TPXO NetCDF 'con' as 1-D strings or 2-D character arrays."""
    con = np.asarray(con)
    if con.ndim == 2:
        return [b''.join(row).decode('ascii').strip().upper() if con.dtype.kind == 'S'
                else ''.join(row).strip().upper() for row in con]
    return [v.decode('ascii').strip().upper() if isinstance(v, bytes)
            else str(v).strip().upper() for v in con]

def _unit_factor(unit, *, transport=False):
    unit = (unit or '').strip().lower().replace(' ', '')
    # Deliberately refuse unknown units: silent cm / m mix gives wrong tides.
    if not transport:
        if unit in ('m', 'meter', 'meters'): return 1.0
        if unit in ('cm', 'centimeter', 'centimeters'): return 0.01
    else:
        if unit in (
            'm2/s', 'm^2/s', 'm2s-1', 'm^2s^-1',
            'meter^2/s', 'meters^2/s',
        ):
            return 1.0

        if unit in ('cm2/s', 'cm^2/s', 'cm2s-1', 'cm^2s^-1'):
            return 1e-4
    raise ValueError(f'Unknown TPXO {"transport" if transport else "height"} units {unit!r}; set units correctly in source file')

def _rectilinear_axes(lon, lat):
    lon, lat = np.asarray(lon), np.asarray(lat)
    if lon.ndim == lat.ndim == 1:
        return lon.astype(float), lat.astype(float)
    if lon.ndim == lat.ndim == 2 and lon.shape == lat.shape:
        # Determine whether longitude changes along the first or second axis.
        if np.allclose(lon, lon[0][None, :], equal_nan=True) and np.allclose(lat, lat[:, 0][:, None], equal_nan=True):
            return lon[0].astype(float), lat[:, 0].astype(float)
        if np.allclose(lon, lon[:, 0][:, None], equal_nan=True) and np.allclose(lat, lat[0][None, :], equal_nan=True):
            return lon[:, 0].astype(float), lat[0].astype(float)
    raise ValueError('TPXO coordinates are not separable/rectilinear: need a curvilinear-grid interpolator')


def _interpolate_complex(lon, lat, complex_grid, target_lon, target_lat, *, axes='yx'):
    """
    Bilinear interpolation of complex tidal coefficients (no phase wrapping).

    complex_grid shape (n_constituents, latitude, longitude) if axes='yx',
    or (n_constituents, longitude, latitude) if axes='xy'.
    """
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    z = np.asarray(complex_grid, dtype=complex)
    if axes == 'xy':
        z = z.transpose(0, 2, 1)
    if z.shape[1:] != (len(lat), len(lon)):
        raise ValueError('TPXO harmonic array shape does not match longitude and latitude axes')
    
    ilon = np.argsort(lon)
    ilat = np.argsort(lat)
    lon, lat, z = lon[ilon], lat[ilat], z[:, ilat][:, :, ilon]
    if np.any(np.diff(lon) <= 0) or np.any(np.diff(lat) <= 0):
        raise ValueError('TPXO coordinate axes must have unique values')
    x = np.asarray(target_lon, dtype=float).copy()
    y = np.asarray(target_lat, dtype=float).copy()
    if x.shape != y.shape:
        raise ValueError('Target lon/lat shape mismatch')
    
    # Interpret equivalent geographic longitudes in the source convention.
    x += 360.0 * np.round((np.median(lon) - x) / 360.0)
    xy = np.column_stack((y.ravel(), x.ravel()))
    result = []
    for field in z:
        good = np.isfinite(field)
        if not good.all():
            # Do not cross TPXO land masks by silently extrapolating.
            fi = np.where(good, field, np.nan + 1j * np.nan)
        else:
            fi = field
        interp = RegularGridInterpolator((lat, lon), fi, bounds_error=True)
        result.append(interp(xy).reshape(x.shape))
    result = np.asarray(result)
    if not np.isfinite(result).all():
        raise ValueError('TPXO harmonic interpolation intersects land/missing source data')
    return result


def _load_harmonic_file(filename, key, target_lon, target_lat, constituents):
    re_name, im_name, lon_name, lat_name = _COMPONENTS[key]
    with xr.open_dataset(filename, decode_cf=True) as ds:
        names = _constituents(ds['con'].values)
        indices = []
        for name in constituents:
            if name.upper() not in names:
                raise ValueError(f'TPXO constituent {name} absent from {filename}: {names}')
            indices.append(names.index(name.upper()))
        da = ds[re_name]
        db = ds[im_name]
        if da.dims != db.dims:
            raise ValueError('TPXO real/imag dimensions do not match')
        
        dcon = ds['con'].dims[0]
        space = [d for d in da.dims if d != dcon]
        if len(space) != 2:
            raise ValueError('Expected 2-D TPXO harmonic grid for each constituent')
        lon, lat = _rectilinear_axes(ds[lon_name].values, ds[lat_name].values)

        # Infer orientation from dimension lengths / coordinates. Ambiguous
        # square matrices must use named geographic axes.
        dims_lon = ds[lon_name].dims
        dims_lat = ds[lat_name].dims
        if len(dims_lon) == 1 and len(dims_lat) == 1:
            da, db = da.transpose(dcon, dims_lat[0], dims_lon[0]), db.transpose(dcon, dims_lat[0], dims_lon[0])
        elif len(dims_lon) == 2:
            if np.allclose(ds[lon_name].values, ds[lon_name].values[0][None, :]):
                da, db = da.transpose(dcon, dims_lon[0], dims_lon[1]), db.transpose(dcon, dims_lon[0], dims_lon[1])
            else:
                da, db = da.transpose(dcon, dims_lon[1], dims_lon[0]), db.transpose(dcon, dims_lon[1], dims_lon[0])
        else:
            raise ValueError('Unexpected TPXO coordinate layout')
        real = np.ma.filled(np.ma.asarray(da.isel({dcon: indices}).values), np.nan)
        imag = np.ma.filled(np.ma.asarray(db.isel({dcon: indices}).values), np.nan)
        scale = _unit_factor(da.attrs.get('units'), transport=(key != 'zeta'))
        harmonics = _interpolate_complex(lon, lat, scale*(real+1j*imag), target_lon, target_lat)
    return harmonics


def _predict_utide(harmonics, names, times_mjd, latitudes):
    """
    UTide astronomical reconstruction, using PyFVCOM's convention.
    """
    names = [str(c).upper() for c in names]
    known = list(ut_constants['const']['name'])
    idx = [known.index(name) for name in names]
    frequencies = ut_constants['const']['freq'][idx]
    times = np.asarray(times_mjd, dtype=float)
    coeff = np.asarray(harmonics)
    latitude = np.asarray(latitudes)
    if coeff.shape != (len(names), len(latitude)):
        raise ValueError('Harmonic coefficients must have shape (constituents, target points)')
    result = np.empty((len(times), len(latitude)))
    for i, lat in enumerate(latitude):
        opts = Bunch(twodim=False, nodsatlint=False, nodsatnone=False,
                     gwchlint=False, gwchnone=False, notrend=True,
                     prefilt=[], nodiagn=True)
        c = Bunch(name=names, mean=0, slope=0)
        c.aux = Bunch(reftime=729572.47916666674, lind=np.array(idx),
                      frq=frequencies, opt=opts, lat=float(lat))
        c.A = np.abs(coeff[:, i])
        c.g = -np.angle(coeff[:, i], deg=True)
        c.A_ci = np.zeros(len(names))
        c.g_ci = np.zeros(len(names))
        result[:, i] = reconstruct(times, c, epoch='1858-11-17', verbose=False)['h']
    return result


def add_tpxo_tides(nest_file, *, harmonics_files, constituents=('M2','S2','N2','K2','K1','O1','P1','Q1'),
                    tpxo_bathymetry=None):
    """
    Add TPXO barotropic tides to an existing FVCOM nest in-place.

    ``harmonics_files`` keys: 'zeta' (required), optionally 'u' and 'v'.
    Currents need ``tpxo_bathymetry`` explicitly supplied as TPXO grid file;
    accepted bathy names are 'hu' and 'hv', respectively. No unit guessing.
    """
    if 'zeta' not in harmonics_files:
        raise ValueError("harmonics_files must contain 'zeta'")
    if ('u' in harmonics_files or 'v' in harmonics_files) and not tpxo_bathymetry:
        raise ValueError('TPXO transport needs TPXO bathymetry to convert to m/s')
    
    # Open file for reading and process all harmonics BEFORE modifying values.
    with Dataset(nest_file, 'r') as nc:
        _assert_tpxo_not_applied(nc)
        lon, lat = np.asarray(nc['lon'][:]), np.asarray(nc['lat'][:])
        lonc, latc = np.asarray(nc['lonc'][:]), np.asarray(nc['latc'][:])
        t = np.asarray(nc['Itime'][:]) + np.asarray(nc['Itime2'][:])/86_400_000.
        projected = getattr(nc, 'interpolation_projection', 'degrees') != 'degrees'
        if projected and any(k in harmonics_files for k in ('u', 'v')):
            raise NotImplementedError('TPXO currents on projected FVCOM grids require velocity rotation')
        xcoords = np.asarray(nc['xc'][:]); ycoords = np.asarray(nc['yc'][:])
        widths = nc.dimensions['siglay'].size

    predictions = {}
    for key, path in harmonics_files.items():
        if key not in _COMPONENTS:
            raise ValueError(f'Unsupported tide component {key!r}')
        x, y = (lon, lat) if key == 'zeta' else (lonc, latc)
        coeff = _load_harmonic_file(path, key, x, y, constituents)
        if key != 'zeta':
            with xr.open_dataset(tpxo_bathymetry) as hds:
                name = {'u': 'hu', 'v': 'hv'}[key]
                _, _, lonname, latname = _COMPONENTS[key]
                lon_h, lat_h = _rectilinear_axes(hds[lonname].values, hds[latname].values)
                h = np.asarray(hds[name].values)
                if h.shape == (len(lon_h), len(lat_h)):
                    h = h.T
                elif h.shape != (len(lat_h), len(lon_h)):
                    raise ValueError(f'Incompatible shape for TPXO depth {name}')
                interp = RegularGridInterpolator((lat_h, lon_h), h, bounds_error=True)
                x2 = x + 360*np.round((np.median(lon_h)-x)/360)
                bathy_units = str(hds[name].attrs.get('units', '')).lower().strip()
                if bathy_units in ('cm', 'centimeter', 'centimeters'):
                    h = h * 0.01
                    interp = RegularGridInterpolator((lat_h, lon_h), h, bounds_error=True)
                elif bathy_units not in ('m', 'meter', 'meters'):
                    raise ValueError(f'Unknown TPXO depth units for {name}: {bathy_units!r}')
                depth = interp(np.column_stack((y,x2)))
                if not np.isfinite(depth).all() or np.any(depth <= 0):
                    raise ValueError('TPXO bathymetry is missing or non-positive on nesting boundary')
                coeff = coeff / depth[None, :]
        predictions[key] = _predict_utide(coeff, constituents, t, y)

    # Only mutate on success.  Barotropic tide is constant through sigma.
    with _tpxo_transaction(nest_file) as _tpxo_staging:
        with Dataset(_tpxo_staging, 'r+') as nc:
            nc['zeta'][:] = np.asarray(nc['zeta'][:]) + predictions['zeta']
            for key in ('u', 'v'):
                if key not in predictions:
                    continue
                nc[key][:] = np.asarray(nc[key][:]) + predictions[key][:, None, :]
                nc[{'u': 'ua', 'v': 'va'}[key]][:] = (np.asarray(nc[{'u': 'ua', 'v': 'va'}[key]][:]) + predictions[key])
            nc.tidal_source = 'TPXO'
            nc.tidal_constituents = ','.join(constituents)
            nc.tpxo_applied = 1
    return str(nest_file)


def _assert_tpxo_not_applied(nc):
    """Refuse both new TPXO markers and the legacy tidal_source metadata."""
    if ('tpxo_applied' in nc.ncattrs()
            or 'tpxo' in str(getattr(nc, 'tidal_source', '')).casefold()):
        raise ValueError(
            'TPXO tide has already been added to this nesting file. '
            'Use the original, tide-free nesting file instead.'
        )


@_tpxo_contextmanager
def _tpxo_transaction(nest_file):
    """Lock, recheck, modify a copy, then atomically replace original on success.

    Note: after an unexpected interpreter/process crash, a stale .tpxo.lock
    must be reviewed/removed manually before running again.
    """
    from pathlib import Path
    import os
    import shutil
    import tempfile
    from netCDF4 import Dataset

    source = Path(nest_file)
    lock_path = source.with_name(source.name + '.tpxo.lock')
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f'TPXO operation already in progress or stale lock: {lock_path}'
        ) from exc

    temp = None
    try:
        os.close(fd)
        # Recheck after obtaining lock: another call could have finished while
        # this one was calculating the tidal predictions.
        with Dataset(source, 'r') as nc:
            _assert_tpxo_not_applied(nc)

        tmp_fd, path = tempfile.mkstemp(
            dir=source.parent, prefix=f'.{source.name}.tpxo.', suffix='.partial'
        )
        os.close(tmp_fd)
        temp = Path(path)
        shutil.copy2(source, temp)
        yield temp
        os.replace(temp, source)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)
