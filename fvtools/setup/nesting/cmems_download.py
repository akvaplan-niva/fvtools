"""Download local GLORYS daily forcing and corresponding static deptho.

The dynamic field names come from :mod:`fvtools.setup.nesting.cmems`,
so the downloader and interpolator cannot silently disagree.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import os

import numpy as np
import xarray as xr

from fvtools.grid.fvcom_grd import FVCOM_grid

DYNAMIC_DATASET = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
STATIC_DATASET = 'cmems_mod_glo_phy_my_0.083deg_static'


def _datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError('start and stop must be date, datetime or ISO date strings')


def _bounds(model, buffer_degrees):
    if not np.isfinite(buffer_degrees) or buffer_degrees < 0:
        raise ValueError('buffer_degrees must be finite and non-negative')
    lon = np.asarray(model.lon, dtype=float)
    lat = np.asarray(model.lat, dtype=float)
    if (not lon.size or not lat.size or lon.shape != lat.shape or
            not np.isfinite(lon).all() or not np.isfinite(lat).all()):
        raise ValueError('FVCOM_grid must supply finite matching lon/lat arrays')
    west, east = float(lon.min() - buffer_degrees), float(lon.max() + buffer_degrees)
    south, north = float(lat.min() - buffer_degrees), float(lat.max() + buffer_degrees)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError('GLORYS subset bounds cross a longitude seam or latitude pole')
    return dict(minimum_longitude=west, maximum_longitude=east,
                minimum_latitude=south, maximum_latitude=north)


def _validate_file(path, *, variables, time_axis):
    """Check completed file structure; avoid treating interrupted files as cached."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'CMEMS result is missing or empty: {path}')
    try:
        with xr.open_dataset(path) as ds:
            missing = set(variables) - set(ds.data_vars)
            required = {'longitude', 'latitude'} | ({'time', 'depth'} if time_axis else set())
            missing_coords = required - set(ds.coords)
            if missing or missing_coords:
                raise ValueError(f'Missing variables {sorted(missing)} or coordinates '
                                 f'{sorted(missing_coords)} in {path}')
            if time_axis and ds.sizes.get('time', 0) == 0:
                raise ValueError(f'No time records in {path}')
    except (OSError, RuntimeError) as exc:
        raise ValueError(f'Unreadable CMEMS NetCDF: {path}') from exc


def _download_if_needed(cm, target, *, variables, time_axis, force, kwargs):
    if target.exists() and not force:
        _validate_file(target, variables=variables, time_axis=time_axis)
        return target

    # Copernicus writes to a temporary name. An interrupted download can
    # never masquerade as a complete cached file or overwrite a valid file.
    staging = target.with_name(f'.{target.stem}.{uuid4().hex}.partial.nc')
    try:
        cm.subset(
            **kwargs,
            variables=list(variables),
            output_directory=str(target.parent),
            output_filename=staging.name,
            overwrite=True,
        )
        _validate_file(staging, variables=variables, time_axis=time_axis)
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target


def download_cmems(
    model: FVCOM_grid,
    start,
    stop,
    *,
    output_directory='input/cmems',
    buffer_degrees=0.3,
    extra_days=3,
    force=False,
):
    """Download daily GLORYS forcing and static deptho for an FVCOM grid.

    Returns a mapping with ``dynamic`` and ``bathymetry`` Path entries.
    Repeated calls for the same region and interval reuse verified files.
    Set ``force=True`` to replace previous downloads.
    """
    import copernicusmarine as cm  # Optional dependency; needed only here.
    from fvtools.setup.nesting.cmems import VARIABLES

    start, stop = _datetime(start), _datetime(stop)
    if stop < start:
        raise ValueError('stop must not precede start')
    if not isinstance(extra_days, int) or extra_days < 0:
        raise ValueError('extra_days must be a non-negative integer')
    end = stop + timedelta(days=extra_days)
    extent = _bounds(model, buffer_degrees)
    outdir = Path(output_directory)
    outdir.mkdir(parents=True, exist_ok=True)

    # Spatially keyed cached files are never reused for an unrelated FVCOM grid.
    key = sha256(','.join(f'{value:.8f}' for value in extent.values()).encode()).hexdigest()[:12]
    date_key = f'{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}'
    dynamic = outdir / f'glorys_daily_{date_key}_{key}.nc'
    static = outdir / f'glorys_deptho_{key}.nc'
    spatial = {**extent, 'coordinates_selection_method': 'outside'}

    # Small static file first: if it fails, do not start the large request.
    _download_if_needed(
        cm, static, variables=('deptho',), time_axis=False, force=force,
        kwargs={**spatial, 'dataset_id': STATIC_DATASET, 'dataset_part': 'bathy'},
    )
    _download_if_needed(
        cm, dynamic, variables=tuple(VARIABLES.values()), time_axis=True,
        force=force,
        kwargs={**spatial, 'dataset_id': DYNAMIC_DATASET,
                'start_datetime': start, 'end_datetime': end},
    )
    return {'dynamic': dynamic, 'bathymetry': static}
