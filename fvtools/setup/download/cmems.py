"""Download combined CMEMS forcing files for GLORYS and NWSHELF.

The GLORYS downloader preserves the existing daily product workflow.
The NWSHELF downloader combines its separate hourly datasets into one
NetCDF file that can be read by ``fvtools.setup.nesting.cmems``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
import copernicusmarine as cm  # Optional dependency for downloads only.

from netCDF4 import Dataset

from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.glorys_grid import GLORYS

DYNAMIC_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
STATIC_DATASET = "cmems_mod_glo_phy_my_0.083deg_static"
DYNAMIC_VARIABLES = ("thetao", "so", "uo", "vo", "zos")

# Current dataset identifiers as listed in the Copernicus Marine catalogue.
NWSHELF_DATASETS = {
    "ssh": {
        "id": "cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT1H-i",
        "variables": ("zos",),
    },
    "current_3d": {
        "id": "cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i",
        "variables": ("uo", "vo", "wo", "ubar", "vbar"),
    },
    "salinity": {
        "id": "cmems_mod_nws_phy-sal_anfc_1.5km-3D_PT1H-i",
        "variables": ("so",),
    },
    "temperature": {
        "id": "cmems_mod_nws_phy-tem_anfc_1.5km-3D_PT1H-i",
        "variables": ("thetao",),
    },
    "static": {
        "id": "cmems_mod_nws_phy_anfc_1.5km_static",
        "part": "bathy",
        "variables": ("deptho",),
    },
}

NWSHELF_DOMAIN = {
    "minimum_longitude": -16.0,
    "maximum_longitude": 13.0,
    "minimum_latitude": 46.0,
    "maximum_latitude": 62.75,
}


def _datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError("start and stop must be dates, datetimes or ISO strings")


def download_glorys(
    model: FVCOM_grid,
    start,
    stop,
    *,
    outfile="input/cmems/glorys.nc",
    buffer_degrees=0.3,
    extra_days=3,
    overwrite=False,
):
    """Download one GLORYS NetCDF containing dynamics and static deptho.

    Download static and dynamic GLORYS subsets, align bathymetry to the
    dynamic grid, and append only the 2D deptho field. Large 4D variables
    are not loaded into memory. Publish only after both downloads succeed.

    Existing ``outfile`` is left untouched unless ``overwrite=True``.
    """
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


# ---------------------------------------------------------------------------
# NWSHELF helpers
# ---------------------------------------------------------------------------


def _nwshelf_bounds(model, buffer_degrees):
    """Return a buffered subset box, clipped to the NWSHELF product domain."""
    if not np.isfinite(buffer_degrees) or buffer_degrees < 0:
        raise ValueError("buffer_degrees must be finite and non-negative")

    lon = getattr(model, "lon")
    lat = getattr(model, "lat")
    lon = lon() if callable(lon) else lon
    lat = lat() if callable(lat) else lat
    lon = np.ma.asarray(lon, dtype=float).filled(np.nan)
    lat = np.ma.asarray(lat, dtype=float).filled(np.nan)
    if (not lon.size or lon.shape != lat.shape or
            not np.isfinite(lon).all() or not np.isfinite(lat).all()):
        raise ValueError("FVCOM_grid requires matching, finite lon/lat arrays")

    # Accept FVCOM longitude arrays expressed either in [-180, 180] or [0, 360].
    lon = (lon + 180.0) % 360.0 - 180.0
    grid_bounds = {
        "minimum_longitude": float(lon.min()),
        "maximum_longitude": float(lon.max()),
        "minimum_latitude": float(lat.min()),
        "maximum_latitude": float(lat.max()),
    }
    if (
        grid_bounds["minimum_longitude"] < NWSHELF_DOMAIN["minimum_longitude"]
        or grid_bounds["maximum_longitude"] > NWSHELF_DOMAIN["maximum_longitude"]
        or grid_bounds["minimum_latitude"] < NWSHELF_DOMAIN["minimum_latitude"]
        or grid_bounds["maximum_latitude"] > NWSHELF_DOMAIN["maximum_latitude"]
    ):
        raise ValueError(
            "FVCOM grid lies partly outside the NWSHELF product domain "
            "(-16 to 13 degrees longitude, 46 to 62.75 degrees latitude)"
        )

    return {
        "minimum_longitude": max(
            NWSHELF_DOMAIN["minimum_longitude"],
            grid_bounds["minimum_longitude"] - buffer_degrees,
        ),
        "maximum_longitude": min(
            NWSHELF_DOMAIN["maximum_longitude"],
            grid_bounds["maximum_longitude"] + buffer_degrees,
        ),
        "minimum_latitude": max(
            NWSHELF_DOMAIN["minimum_latitude"],
            grid_bounds["minimum_latitude"] - buffer_degrees,
        ),
        "maximum_latitude": min(
            NWSHELF_DOMAIN["maximum_latitude"],
            grid_bounds["maximum_latitude"] + buffer_degrees,
        ),
        "coordinates_selection_method": "outside",
    }


def _download_nwshelf_dataset(
    cm, name, config, bounds, start, stop, directory, extra_days
):
    """Download one NWSHELF dataset to a temporary NetCDF file."""
    path = directory / f"{name}.nc"
    kwargs = {
        "dataset_id": config["id"],
        "variables": list(config["variables"]),
        "output_directory": str(directory),
        "output_filename": path.name,
        **bounds,
    }
    if "part" in config:
        kwargs["dataset_part"] = config["part"]
    else:
        kwargs["start_datetime"] = start
        kwargs["end_datetime"] = stop + timedelta(days=extra_days)

    print(f"Downloading NWSHELF {name}: {config['id']}")
    cm.subset(**kwargs)
    if not path.is_file():
        raise FileNotFoundError(
            f"Copernicus did not return the requested NWSHELF file: {path.name}"
        )
    return path


def _decoded_times(path):
    """Read and validate a NetCDF time coordinate as datetime64[ns]."""
    with xr.open_dataset(path, decode_times=True) as ds:
        if "time" not in ds.coords:
            raise ValueError(f"{path.name}: missing time coordinate")
        values = np.asarray(ds.time.values)

    try:
        times = values.astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name}: unsupported time coordinate") from exc

    if times.ndim != 1 or not len(times) or np.isnat(times).any():
        raise ValueError(f"{path.name}: invalid time coordinate")
    if np.any(times[1:] <= times[:-1]):
        raise ValueError(f"{path.name}: time must be strictly increasing")
    return times


def _common_time_indices(files):
    """Return common timestamps and source indices for every dynamic file."""
    source_times = {name: _decoded_times(path) for name, path in files.items()
                    if name != "static"}
    common = next(iter(source_times.values()))
    for times in list(source_times.values())[1:]:
        common = np.intersect1d(common, times)

    if not len(common):
        raise ValueError(
            "The requested period has no timestamps common to all NWSHELF datasets. "
            "The 3D temperature, salinity and current products have a shorter "
            "time coverage than some 2D products."
        )

    indices = {}
    for name, times in source_times.items():
        idx = np.searchsorted(times, common)
        if np.any(idx >= len(times)) or not np.array_equal(times[idx], common):
            raise ValueError(f"Could not align the common time axis for {name}")
        indices[name] = idx.astype(np.int64)
    return common, indices


def _assert_nwshelf_coordinates(files):
    """Validate common horizontal and vertical coordinates before merging."""
    dynamic_names = [name for name in files if name != "static"]
    reference_name = "temperature"
    if reference_name not in files:
        raise ValueError("NWSHELF temperature dataset is required")

    with xr.open_dataset(files[reference_name], decode_times=True) as reference:
        reference_coords = {
            name: np.asarray(reference[name].values)
            for name in ("latitude", "longitude", "depth")
        }

    for name in dynamic_names:
        with xr.open_dataset(files[name], decode_times=True) as ds:
            for coord in ("latitude", "longitude"):
                if coord not in ds.coords or ds[coord].ndim != 1:
                    raise ValueError(f"{name}: expected 1D {coord} coordinate")
                values = np.asarray(ds[coord].values)
                expected = reference_coords[coord]
                if values.shape != expected.shape or not np.allclose(
                    values, expected, rtol=0.0, atol=1e-8, equal_nan=False
                ):
                    raise ValueError(
                        f"NWSHELF {coord} coordinates do not match in {name}"
                    )

            if name in ("temperature", "salinity", "current_3d"):
                if "depth" not in ds.coords or ds.depth.ndim != 1:
                    raise ValueError(f"{name}: expected 1D depth coordinate")
                values = np.asarray(ds.depth.values, dtype=float)
                expected = np.asarray(reference_coords["depth"], dtype=float)
                if values.shape != expected.shape or not np.allclose(
                    values, expected, rtol=0.0, atol=1e-8, equal_nan=False
                ):
                    raise ValueError(
                        f"NWSHELF depth coordinates do not match in {name}"
                    )

    depth = np.asarray(reference_coords["depth"], dtype=float)
    if (depth.ndim != 1 or len(depth) < 2 or not np.isfinite(depth).all()
            or depth[0] < 0 or np.any(np.diff(depth) <= 0)):
        raise ValueError(
            "NWSHELF depth must be positive, finite and ordered shallow-to-deep"
        )


def _variable_chunks(shape, dimensions):
    """Yield bounded slices, keeping NetCDF variable copies memory efficient."""
    axis_chunks = []
    for size, dimension in zip(shape, dimensions):
        if dimension == "time":
            chunk_size = 1
        elif dimension == "depth":
            chunk_size = min(size, 8)
        elif dimension in ("latitude", "longitude"):
            chunk_size = min(size, 256)
        else:
            chunk_size = size
        axis_chunks.append([
            slice(start, min(start + chunk_size, size))
            for start in range(0, size, chunk_size)
        ])
    return product(*axis_chunks)


def _copy_variable_data(source, destination, time_indices=None):
    """Copy a NetCDF variable in small blocks, optionally selecting common times."""
    dimensions = source.dimensions
    shape = source.shape

    if "time" in dimensions:
        time_axis = dimensions.index("time")
        other_axes = [axis for axis in range(len(dimensions)) if axis != time_axis]
        other_chunks = []
        for axis in other_axes:
            dim = dimensions[axis]
            size = shape[axis]
            chunk_size = min(size, 8 if dim == "depth" else 256)
            other_chunks.append([
                slice(start, min(start + chunk_size, size))
                for start in range(0, size, chunk_size)
            ])

        for output_time, source_time in enumerate(time_indices):
            for blocks in product(*other_chunks):
                src_key = [slice(None)] * len(dimensions)
                dst_key = [slice(None)] * len(dimensions)
                src_key[time_axis] = slice(int(source_time), int(source_time) + 1)
                dst_key[time_axis] = slice(output_time, output_time + 1)
                for axis, block in zip(other_axes, blocks):
                    src_key[axis] = block
                    dst_key[axis] = block
                destination[tuple(dst_key)] = source[tuple(src_key)]
        return

    for blocks in _variable_chunks(shape, dimensions):
        destination[blocks] = source[blocks]


def _copy_netcdf_variable(source, destination_file, name, *, output_name=None,
                          time_indices=None, compression_level=4):
    """Create and copy one variable, preserving its metadata."""
    output_name = output_name or name
    fill_value = source.getncattr("_FillValue") if "_FillValue" in source.ncattrs() else None
    kwargs = {}
    if np.issubdtype(source.dtype, np.number) and source.ndim:
        kwargs.update(zlib=True, complevel=compression_level, shuffle=True)
        chunksizes = []
        for size, dim in zip(source.shape, source.dimensions):
            if dim == "time":
                chunksizes.append(1)
            elif dim == "depth":
                chunksizes.append(min(size, 8))
            elif dim in ("latitude", "longitude"):
                chunksizes.append(min(size, 256))
            else:
                chunksizes.append(size)
        kwargs["chunksizes"] = tuple(chunksizes)
    if fill_value is not None:
        kwargs["fill_value"] = fill_value

    target = destination_file.createVariable(
        output_name, source.dtype, source.dimensions, **kwargs
    )
    for attribute in source.ncattrs():
        if attribute != "_FillValue":
            target.setncattr(attribute, source.getncattr(attribute))

    _copy_variable_data(source, target, time_indices=time_indices)
    return target


def _merge_nwshelf(files, outfile, *, compression_level=4):
    """Merge NWSHELF subsets without loading the large 4D fields into RAM."""
    _assert_nwshelf_coordinates(files)
    common_times, time_indices = _common_time_indices(files)

    # Reuse the existing GLORYS grid reader for the common rectilinear CMEMS
    # grid and its strict static-deptho alignment checks. Its grid handling is
    # independent of whether the source product is GLORYS or NWSHELF.
    with xr.open_dataset(files["salinity"], decode_times=True) as salinity, \
            xr.open_dataset(files["static"], decode_times=True) as bathymetry:
        deptho = np.asarray(
            GLORYS(salinity, bathymetry=bathymetry).h_rho,
            dtype=np.float32,
        )

    temporary_output = outfile.with_name(f".{outfile.name}.partial")
    try:
        with Dataset(files["temperature"], "r") as base, \
                Dataset(temporary_output, "w", format="NETCDF4") as output:

            # The temperature subset provides the canonical time/depth/grid
            # dimensions and coordinate variables.
            for dimension in ("time", "depth", "latitude", "longitude"):
                if dimension not in base.dimensions:
                    raise ValueError(
                        f"NWSHELF temperature file is missing {dimension!r} dimension"
                    )
                size = (
                    len(common_times)
                    if dimension == "time"
                    else len(base.dimensions[dimension])
                )
                output.createDimension(dimension, size)

            for coordinate in ("time", "depth", "latitude", "longitude"):
                if coordinate not in base.variables:
                    raise ValueError(
                        f"NWSHELF temperature file is missing {coordinate!r} coordinate"
                    )
                _copy_netcdf_variable(
                    base.variables[coordinate], output, coordinate,
                    time_indices=time_indices["temperature"] if coordinate == "time" else None,
                    compression_level=0,
                )

            # Copy global metadata from the canonical source file.
            for attribute in base.ncattrs():
                output.setncattr(attribute, base.getncattr(attribute))

            # The 3D current dataset also contains the barotropic 2D
            # components ubar/vbar. Keep both those and the 3D uo/vo fields.
            variables = {
                "ssh": (("zos", "zos"),),
                "current_3d": (
                    ("uo", "uo"), ("vo", "vo"), ("wo", "wo"),
                    ("ubar", "ubar"), ("vbar", "vbar"),
                ),
                "salinity": (("so", "so"),),
                "temperature": (("thetao", "thetao"),),
            }

            for dataset_name, mappings in variables.items():
                with Dataset(files[dataset_name], "r") as source_file:
                    for source_name, output_name in mappings:
                        if source_name not in source_file.variables:
                            raise ValueError(
                                f"{dataset_name}: missing variable {source_name!r}"
                            )
                        source_variable = source_file.variables[source_name]
                        _validate_nwshelf_variable_dimensions(
                            dataset_name, source_name, source_variable.dimensions
                        )
                        _copy_netcdf_variable(
                            source_variable,
                            output,
                            source_name,
                            output_name=output_name,
                            time_indices=time_indices[dataset_name],
                            compression_level=compression_level,
                        )

            # Add static bathymetry on the dynamic NWSHELF grid.
            bathy = output.createVariable(
                "deptho", "f4", ("latitude", "longitude"),
                zlib=True, complevel=compression_level, shuffle=True,
                chunksizes=(min(deptho.shape[0], 256), min(deptho.shape[1], 256)),
                fill_value=np.float32(1e20),
            )
            bathy[:] = np.ma.masked_invalid(deptho)
            bathy.units = "m"
            bathy.positive = "down"
            bathy.long_name = "NWSHELF model seafloor depth"
            bathy.standard_name = "sea_floor_depth_below_geoid"
            bathy.source_dataset = NWSHELF_DATASETS["static"]["id"]
            bathy.source_part = NWSHELF_DATASETS["static"]["part"]

            output.source_product = "NWSHELF_ANALYSISFORECAST_PHY_004_013"
            output.source_datasets = "; ".join(
                config["id"] for config in NWSHELF_DATASETS.values()
            )
            output.history = (
                "Combined from Copernicus Marine NWSHELF hourly datasets; "
                "time coordinate restricted to timestamps common to all subsets."
            )
            output.current_3d_variables = "uo, vo, wo"
            output.barotropic_current_variables = "ubar, vbar"

        temporary_output.replace(outfile)
    finally:
        temporary_output.unlink(missing_ok=True)


def _validate_nwshelf_variable_dimensions(dataset_name, variable_name, dimensions):
    """Fail early if a Copernicus dataset does not have the expected layout."""
    expected = {
        ("ssh", "zos"): ("time", "latitude", "longitude"),
        ("current_3d", "uo"): ("time", "depth", "latitude", "longitude"),
        ("current_3d", "vo"): ("time", "depth", "latitude", "longitude"),
        ("current_3d", "wo"): ("time", "depth", "latitude", "longitude"),
        ("current_3d", "ubar"): ("time", "latitude", "longitude"),
        ("current_3d", "vbar"): ("time", "latitude", "longitude"),
        ("salinity", "so"): ("time", "depth", "latitude", "longitude"),
        ("temperature", "thetao"): ("time", "depth", "latitude", "longitude"),
    }
    required = expected[(dataset_name, variable_name)]
    if tuple(dimensions) != required:
        raise ValueError(
            f"NWSHELF {dataset_name}.{variable_name}: expected dimensions "
            f"{required}, got {tuple(dimensions)}"
        )


def download_nwshelf(
    model: FVCOM_grid,
    start,
    stop,
    *,
    outfile="input/cmems/nwshelf.nc",
    buffer_degrees=0.3,
    extra_days=0,
    overwrite=False,
    compression_level=4,
):
    """Download NWSHELF hourly fields and combine them into one NetCDF.

    The 3D-current dataset includes both 3D ``uo``/``vo``/``wo`` and the
    vertically integrated 2D barotropic components ``ubar``/``vbar``. The
    latter are retained directly, so no separate surface-current dataset is
    needed for the requested 2D and 3D velocity fields.

    The output contains the variables required by
    ``fvtools.setup.nesting.cmems`` with the expected names and dimensions:

    - ``zos(time, latitude, longitude)``
    - ``thetao/so/uo/vo(time, depth, latitude, longitude)``
    - ``deptho(latitude, longitude)``

    Additional fields are retained:

    - ``wo`` from the 3D current dataset
    - ``ubar`` and ``vbar`` (2D barotropic currents) from the 3D current dataset

    The 2D SSH and 3D source products do not have identical historical
    coverage. The output therefore contains only timestamps common to all
    requested dynamic datasets. Large 4D variables are copied in bounded blocks.

    Parameters
    ----------
    model : FVCOM_grid
        FVCOM grid object with ``lon`` and ``lat`` arrays.
    start, stop : date, datetime or ISO string
        Requested time interval. ``extra_days`` optionally extends the
        download beyond ``stop``; it defaults to zero for NWSHELF.
    outfile : str or Path
        Combined output NetCDF filename.
    buffer_degrees : float
        Geographic buffer around the FVCOM grid. The buffer is clipped to the
        NWSHELF product extent.
    extra_days : int
        Number of extra days to download after ``stop``.
    overwrite : bool
        Replace an existing output only when True. Failed downloads never
        replace an existing output.
    compression_level : int
        NetCDF compression level from 0 (none) to 9 (maximum).

    Returns
    -------
    pathlib.Path
        Path to the combined NetCDF file.
    """
    start, stop = _datetime(start), _datetime(stop)
    if stop < start:
        raise ValueError("stop must not precede start")
    if not isinstance(extra_days, int) or extra_days < 0:
        raise ValueError("extra_days must be a non-negative integer")
    if not isinstance(compression_level, int) or not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be an integer between 0 and 9")

    bounds = _nwshelf_bounds(model, buffer_degrees)
    destination = Path(outfile)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".nwshelf_", dir=destination.parent) as tmp:
        temporary_directory = Path(tmp)
        files = {}
        for name, config in NWSHELF_DATASETS.items():
            files[name] = _download_nwshelf_dataset(
                cm, name, config, bounds, start, stop,
                temporary_directory, extra_days,
            )

        _merge_nwshelf(
            files,
            destination,
            compression_level=compression_level,
        )

    return destination
