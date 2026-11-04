"""Download one GLORYS NetCDF containing dynamics and static deptho."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.glorys_grid import GLORYS

DYNAMIC_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
STATIC_DATASET = "cmems_mod_glo_phy_my_0.083deg_static"
DYNAMIC_VARIABLES = ("thetao", "so", "uo", "vo", "zos")


def _datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError("start and stop must be dates, datetimes or ISO strings")


def download_cmems(
    model: FVCOM_grid,
    start,
    stop,
    *,
    outfile="input/cmems/glorys.nc",
    buffer_degrees=0.3,
    extra_days=3,
    overwrite=False,
):
    """Return one NetCDF path containing thetao, so, uo, vo, zos and deptho.

    Download static and dynamic GLORYS subsets, align bathymetry to the
    dynamic grid, and append only the 2D deptho field. Large 4D variables
    are not loaded into memory. Publish only after both downloads succeed.

    Existing ``outfile`` is left untouched unless ``overwrite=True``.
    """
    import copernicusmarine as cm  # Optional dependency for downloads only.

    start, stop = _datetime(start), _datetime(stop)
    if stop < start:
        raise ValueError("stop must not precede start")
    if not isinstance(extra_days, int) or extra_days < 0:
        raise ValueError("extra_days must be a non-negative integer")
    if not np.isfinite(buffer_degrees) or buffer_degrees < 0:
        raise ValueError("buffer_degrees must be finite and non-negative")

    lon = np.asarray(model.lon, dtype=float)
    lat = np.asarray(model.lat, dtype=float)
    if (not lon.size or lon.shape != lat.shape or
            not np.isfinite(lon).all() or not np.isfinite(lat).all()):
        raise ValueError("FVCOM_grid requires matching, finite lon/lat arrays")

    bounds = dict(
        minimum_longitude=float(lon.min() - buffer_degrees),
        maximum_longitude=float(lon.max() + buffer_degrees),
        minimum_latitude=float(lat.min() - buffer_degrees),
        maximum_latitude=float(lat.max() + buffer_degrees),
        coordinates_selection_method="outside",
    )
    if not (-180 <= bounds["minimum_longitude"] < bounds["maximum_longitude"] <= 180
            and -90 <= bounds["minimum_latitude"] < bounds["maximum_latitude"] <= 90):
        raise ValueError("GLORYS bounds cross a longitude seam or latitude pole")

    destination = Path(outfile)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Both downloads and all modifications are confined to a temporary
    # directory on the destination filesystem. No partial final NetCDF file.
    with TemporaryDirectory(prefix=".cmems_", dir=destination.parent) as tmp:
        temp = Path(tmp)
        static = temp / "static.nc"
        daily = temp / "daily.nc"

        cm.subset(
            dataset_id=STATIC_DATASET,
            dataset_part="bathy",
            variables=["deptho"],
            output_directory=str(temp), output_filename=static.name,
            **bounds,
        )
        cm.subset(
            dataset_id=DYNAMIC_DATASET,
            variables=list(DYNAMIC_VARIABLES),
            start_datetime=start,
            end_datetime=stop + timedelta(days=extra_days),
            output_directory=str(temp), output_filename=daily.name,
            **bounds,
        )
        if not static.is_file() or not daily.is_file():
            raise FileNotFoundError("Copernicus did not return both requested files")

        # GLORYS already validates and aligns static deptho to the dynamic
        # grid. Reuse that logic instead of implementing interpolation twice.
        with xr.open_dataset(daily) as ds, xr.open_dataset(static) as bathy:
            missing = set(DYNAMIC_VARIABLES) - set(ds.data_vars)
            if missing or ds.sizes.get("time", 0) == 0:
                raise ValueError(f"Invalid dynamic GLORYS data (missing {sorted(missing)})")
            deptho = np.asarray(GLORYS(ds, bathymetry=bathy).h_rho, dtype="f4")

        # Append directly to the downloaded temporary file; do not rewrite
        # the (potentially very large) time-dependent variables via xarray.
        with Dataset(daily, "r+") as nc:
            var = nc.createVariable(
                "deptho", "f4", ("latitude", "longitude"),
                zlib=(nc.data_model == "NETCDF4"),
                fill_value=np.float32(1e20),
            )
            var[:] = np.ma.masked_invalid(deptho)
            var.units = "m"
            var.positive = "down"
            var.long_name = "GLORYS model seafloor depth"
            nc.deptho_source_dataset = STATIC_DATASET
            nc.deptho_source_part = "bathy"

        daily.replace(destination)

    return destination
