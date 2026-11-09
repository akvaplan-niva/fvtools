"""Offline tests for the combined NWSHELF CMEMS downloader."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import xarray as xr

from fvtools.grid.glorys_grid import GLORYS
from fvtools.interpolators.glorys_fields import _open_cmems
from fvtools.setup.download.cmems import NWSHELF_DATASETS, download_nwshelf


TIMES_3D = np.array(
    ["2024-09-17T00:00:00", "2024-09-17T01:00:00"],
    dtype="datetime64[s]",
)
TIMES_2D = np.array(
    ["2024-09-16T23:00:00", "2024-09-17T00:00:00", "2024-09-17T01:00:00"],
    dtype="datetime64[s]",
)
DEPTH = np.array([0.5, 3.0, 5.0], dtype="f4")
LATITUDE = np.array([55.5, 55.6, 55.7], dtype="f4")
LONGITUDE = np.array([4.0, 4.1, 4.2], dtype="f4")


def _dynamic_3d_dataset(*, times=TIMES_3D, longitude=LONGITUDE):
    shape = (len(times), len(DEPTH), len(LATITUDE), len(longitude))
    coords = {
        "time": times,
        "depth": DEPTH,
        "latitude": LATITUDE,
        "longitude": longitude,
    }
    fields = {
        name: (("time", "depth", "latitude", "longitude"),
               np.full(shape, value, dtype="f4"))
        for name, value in (("uo", 0.1), ("vo", 0.2), ("wo", 0.003))
    }
    surface_shape = (len(times), len(LATITUDE), len(longitude))
    fields["ubar"] = (("time", "latitude", "longitude"),
                      np.full(surface_shape, 0.01, dtype="f4"))
    fields["vbar"] = (("time", "latitude", "longitude"),
                      np.full(surface_shape, 0.02, dtype="f4"))
    return xr.Dataset(fields, coords=coords)


def _temperature_dataset():
    ds = _dynamic_3d_dataset()
    return xr.Dataset(
        {"thetao": ds["uo"].rename("thetao") + 10.0},
        coords=ds.coords,
    )


def _salinity_dataset():
    ds = _dynamic_3d_dataset()
    return xr.Dataset(
        {"so": ds["uo"].rename("so") + 34.0},
        coords=ds.coords,
    )


def _ssh_dataset():
    values = np.full((len(TIMES_2D), len(LATITUDE), len(LONGITUDE)), 0.15, dtype="f4")
    return xr.Dataset(
        {"zos": (("time", "latitude", "longitude"), values)},
        coords={"time": TIMES_2D, "latitude": LATITUDE, "longitude": LONGITUDE},
    )


def _static_dataset(*, shifted=False):
    longitude = LONGITUDE + (0.02 if shifted else 0.0)
    deptho = np.array(
        [[100.0, 110.0, 120.0], [130.0, 140.0, 150.0], [160.0, 170.0, 180.0]],
        dtype="f4",
    )
    return xr.Dataset(
        {"deptho": (("latitude", "longitude"), deptho, {"units": "m"})},
        coords={"latitude": LATITUDE, "longitude": longitude},
    )


def mock_copernicus(monkeypatch, *, shifted_static=False, fail_dataset=None):
    calls = []

    def subset(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["output_directory"]) / kwargs["output_filename"]
        dataset_id = kwargs["dataset_id"]

        if dataset_id == NWSHELF_DATASETS["static"]["id"]:
            ds = _static_dataset(shifted=shifted_static)
        elif dataset_id == NWSHELF_DATASETS["ssh"]["id"]:
            ds = _ssh_dataset()
        elif dataset_id == NWSHELF_DATASETS["current_3d"]["id"]:
            ds = _dynamic_3d_dataset()
        elif dataset_id == NWSHELF_DATASETS["salinity"]["id"]:
            ds = _salinity_dataset()
        elif dataset_id == NWSHELF_DATASETS["temperature"]["id"]:
            ds = _temperature_dataset()
        else:
            raise AssertionError(f"Unexpected dataset ID: {dataset_id}")

        if dataset_id == fail_dataset:
            path.write_bytes(b"partial download")
            raise OSError("synthetic download error")

        ds.to_netcdf(path)

    monkeypatch.setitem(
        sys.modules,
        "copernicusmarine",
        SimpleNamespace(subset=subset),
    )
    return calls


def model():
    return SimpleNamespace(
        lon=np.array([4.0, 4.2]),
        lat=np.array([55.5, 55.7]),
    )


def test_nwshelf_download_combines_all_fields_and_common_times(tmp_path, monkeypatch):
    calls = mock_copernicus(monkeypatch)
    output = tmp_path / "nwshelf.nc"

    result = download_nwshelf(
        model(),
        "2024-09-17T00:00:00",
        "2024-09-17T01:00:00",
        outfile=output,
        buffer_degrees=0.0,
        extra_days=0,
    )

    assert result == output
    assert len(calls) == 5
    assert sorted(path.name for path in tmp_path.iterdir()) == ["nwshelf.nc"]
    assert all(call["coordinates_selection_method"] == "outside" for call in calls)

    with xr.open_dataset(output) as ds:
        required = {"zos", "thetao", "so", "uo", "vo", "deptho"}
        additional = {"wo", "ubar", "vbar"}
        assert required | additional <= set(ds.data_vars)
        assert ds.zos.dims == ("time", "latitude", "longitude")
        assert ds.thetao.dims == ("time", "depth", "latitude", "longitude")
        assert ds.so.dims == ("time", "depth", "latitude", "longitude")
        assert ds.uo.dims == ("time", "depth", "latitude", "longitude")
        assert ds.vo.dims == ("time", "depth", "latitude", "longitude")
        assert ds.deptho.dims == ("latitude", "longitude")
        assert ds.ubar.dims == ("time", "latitude", "longitude")
        assert ds.vbar.dims == ("time", "latitude", "longitude")
        assert ds.sizes["time"] == 2
        np.testing.assert_array_equal(
            ds.time.values.astype("datetime64[h]"),
            np.array(["2024-09-17T00", "2024-09-17T01"], dtype="datetime64[h]"),
        )
        assert ds.source_product == "NWSHELF_ANALYSISFORECAST_PHY_004_013"
        np.testing.assert_allclose(ds.uo.values, 0.1)
        np.testing.assert_allclose(ds.ubar.values, 0.01)

    # The merged file must pass the same input checks as a GLORYS file.
    compatible = _open_cmems(output)
    try:
        assert {"thetao", "so", "uo", "vo", "zos", "deptho"} <= set(compatible.variables)
        assert GLORYS(compatible).bathymetry_source == "deptho"
    finally:
        compatible.close()


def test_nwshelf_static_grid_must_match_dynamic_grid(tmp_path, monkeypatch):
    mock_copernicus(monkeypatch, shifted_static=True)
    output = tmp_path / "nwshelf.nc"

    with pytest.raises(ValueError, match="coordinates do not match"):
        download_nwshelf(
            model(), "2024-09-17", "2024-09-17",
            outfile=output, buffer_degrees=0.0, extra_days=0,
        )

    assert not output.exists()


def test_nwshelf_failed_download_does_not_replace_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "nwshelf.nc"
    target.write_bytes(b"original")
    mock_copernicus(
        monkeypatch,
        fail_dataset=NWSHELF_DATASETS["temperature"]["id"],
    )

    with pytest.raises(FileExistsError):
        download_nwshelf(model(), "2024-09-17", "2024-09-17", outfile=target)

    with pytest.raises(OSError, match="synthetic download error"):
        download_nwshelf(
            model(), "2024-09-17", "2024-09-17",
            outfile=target, overwrite=True,
        )

    assert target.read_bytes() == b"original"
    assert [path.name for path in tmp_path.iterdir()] == ["nwshelf.nc"]


def test_nwshelf_rejects_grid_outside_product_domain(tmp_path, monkeypatch):
    calls = mock_copernicus(monkeypatch)
    outside = SimpleNamespace(
        lon=np.array([13.0, 13.1]),
        lat=np.array([55.5, 55.7]),
    )

    with pytest.raises(ValueError, match="outside the NWSHELF product domain"):
        download_nwshelf(
            outside, "2024-09-17", "2024-09-17",
            outfile=tmp_path / "nwshelf.nc",
        )

    assert calls == []
