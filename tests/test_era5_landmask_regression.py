"""Offline regression tests for ERA5 land fraction and FVCOM atmospheric forcing.

Run: uv run --no-sync python -m pytest tests/test_era5_landmask_regression.py -q
"""

import inspect
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
import xarray as xr

from fvtools.grid import era5_grid
from fvtools.interpolators.horizontal_interpolator import N4Coefficients
from fvtools.setup.atm import read_era5


def test_era5_lsm_is_inverted_to_sea_points(monkeypatch):
    lsm = np.array([[0.0, 0.3, 0.8, 1.0, np.nan]])
    ds = xr.Dataset(
        {'lsm': (('time', 'lat', 'lon'), lsm[None, :, :])},
        coords={'time': [np.datetime64('2023-08-01')],
                'lat': [65.0], 'lon': np.arange(5.0)},
    )
    monkeypatch.setattr(era5_grid.xr, 'open_dataset', lambda _: ds)
    e = era5_grid.ERA5('synthetic.nc', Proj=lambda x, y: (x, y))
    np.testing.assert_array_equal(
        e.land_mask, np.array([[True, True, False, False, False]]))
    assert e.land_mask.dtype == np.bool_


def _n4(wet):
    coarse = SimpleNamespace(
        land_mask=np.asarray(wet, dtype=bool),
        x=np.array([[0., 1.], [0., 1.]]),
        y=np.array([[0., 0.], [1., 1.]]),
    )
    n4 = N4Coefficients(None, coarse)
    # All four source points belong to the synthetic domain crop.
    n4.__dict__['fv_domain_mask'] = np.ones((2, 2), dtype=bool)
    return n4


def test_ocean_interpolation_renormalizes_partial_land_and_uses_nearest_sea():
    n4 = _n4([[True, False], [True, False]])
    indices = np.array([[0, 1, 2, 3], [1, 3, 1, 3]])
    coef = np.full((2, 4), 0.25)
    with np.errstate(divide='raise', invalid='raise'):
        out_index, out_coef = n4._remove_land_points(indices, coef)
    np.testing.assert_allclose(out_coef[0], [.5, 0., .5, 0.])
    np.testing.assert_allclose(out_coef[1], [.25] * 4)
    assert np.unique(out_index[1]).size == 1
    assert out_index[1, 0] in (0, 2)
    np.testing.assert_allclose(out_coef.sum(axis=1), 1.)
    assert np.isfinite(out_coef).all()


def test_no_unnecessary_nearest_ocean_query(monkeypatch):
    n4 = _n4([[True, True], [True, True]])
    monkeypatch.setattr(n4, '_find_nearest_ocean_neighbor',
                        lambda *args: pytest.fail('Unexpected nearest-sea query'))
    indices = np.array([[0, 1, 2, 3]])
    coef = np.full((1, 4), .25)
    out_idx, out_coef = n4._remove_land_points(indices, coef)
    np.testing.assert_array_equal(out_idx, indices)
    np.testing.assert_allclose(out_coef, coef)


def test_missing_ocean_points_give_clear_error_not_empty_kdtree():
    n4 = _n4([[False, False], [False, False]])
    with pytest.raises(ValueError, match='No sea points in the cropped atmospheric grid'):
        n4._remove_land_points(np.array([[0, 1, 2, 3]]),
                               np.full((1, 4), .25))


def test_wind_rotation_uses_written_era5_variable_names():
    source = inspect.getsource(read_era5.main)
    assert "d['uwind_speed'][i, :]" in source
    assert "d['vwind_speed'][i, :]" in source
    assert "d['u_wind']" not in source
    assert "d['v_wind']" not in source
    u, v = read_era5._rotate_vector(np.array([2.]), np.array([3.]),
                                     np.array([np.pi / 2]))
    np.testing.assert_allclose(u, [3.])
    np.testing.assert_allclose(v, [-2.])


def test_era5_output_coordinates_and_wind_metadata(tmp_path):
    m = SimpleNamespace(
        x=np.array([10., 20., 30.]),
        y=np.array([40., 50., 60.]),
        lon=np.array([7., 8., 9.]),
        lat=np.array([60., 61., 62.]),
        xc=np.array([20.]), yc=np.array([50.]),
        lonc=np.array([8.]), latc=np.array([61.]),
        tri=np.array([[0, 1, 2]]),
    )
    output = str(tmp_path / 'era5_atm_test.nc')
    read_era5.create_surface_forcing_file(m, output)
    with netCDF4.Dataset(output) as nc:
        np.testing.assert_allclose(nc['x'][:], m.x)
        np.testing.assert_allclose(nc['y'][:], m.y)
        assert nc['lon'].units == 'degrees_east'
        assert nc['lat'].units == 'degrees_north'
        assert nc['uwind_speed'].shape == (0, 1)
        assert nc['vwind_speed'].shape == (0, 1)
        assert 'north' in nc['vwind_speed'].long_name
