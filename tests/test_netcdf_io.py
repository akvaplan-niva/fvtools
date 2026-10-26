"""Test public NetCDF loading without requiring model output or Olivia."""

import numpy as np
import pytest
from netCDF4 import Dataset

from fvtools.grid.fvcom_grd import FVCOM_grid


@pytest.fixture
def tiny_output(tmp_path):
    path = tmp_path / "tiny_output.nc"
    with Dataset(path, "w") as ds:
        for name, length in (
            ("time", 2), ("node", 4), ("nele", 2), ("siglay", 2)
        ):
            ds.createDimension(name, length)

        wet_nodes = ds.createVariable("wet_nodes", "i1", ("time", "node"))
        wet_nodes[:] = [[1, 1, 0, 1], [1, 1, 1, 1]]
        wet_cells = ds.createVariable("wet_cells", "i1", ("time", "nele"))
        wet_cells[:] = [[1, 0], [1, 1]]

        temp = ds.createVariable("temp", "f4", ("time", "siglay", "node"))
        temp[:] = [
            [[10, 11, 12, 13], [14, 15, 16, 17]],
            [[20, 21, 22, 23], [24, 25, 26, 27]],
        ]
        u = ds.createVariable("u", "f4", ("time", "siglay", "nele"))
        u[:] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        v = ds.createVariable("v", "f4", ("time", "siglay", "nele"))
        v[:] = [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
        pv = ds.createVariable("pv", "f4", ("time", "node"))
        pv[:] = [[100, 101, 102, 103], [200, 201, 202, 203]]
    return path


def test_netcdf_loader_preserves_node_and_cell_staggering(small_grid_file, tiny_output):
    grid = FVCOM_grid(str(small_grid_file))
    values = grid.load_netCDF(str(tiny_output), ("temp", "u", "v", "pv"), 0, sig=0)

    assert set(values) == {"temp", "u", "v", "pv"}
    assert values["temp"].shape == (4,)
    assert values["pv"].shape == (4,)
    assert values["u"].shape == (2,)
    assert values["v"].shape == (2,)
    np.testing.assert_allclose(values["temp"], [10, 11, np.nan, 13], equal_nan=True)
    np.testing.assert_allclose(values["u"], [1, np.nan], equal_nan=True)
    np.testing.assert_allclose(values["v"], [9, np.nan], equal_nan=True)
    np.testing.assert_allclose(values["pv"], [100, 101, np.nan, 103], equal_nan=True)


def test_netcdf_session_does_not_reuse_previous_timestep(small_grid_file, tiny_output):
    grid = FVCOM_grid(str(small_grid_file))
    with grid.netcdf_session():
        first = grid.load_netCDF(str(tiny_output), ("u", "v"), 0, sig=0)
        second = grid.load_netCDF(str(tiny_output), ("u", "v"), 1, sig=0)

    np.testing.assert_allclose(first["u"], [1, np.nan], equal_nan=True)
    np.testing.assert_allclose(second["u"], [5, 6], equal_nan=True)
    np.testing.assert_allclose(second["v"], [13, 14], equal_nan=True)


def test_sigma_and_z_cannot_be_requested_in_one_netcdf_read(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    with pytest.raises(ValueError, match="either sig or z"):
        grid.load_netCDF("unused.nc", "temp", 0, sig=0, z=1.0)
