"""Regression tests: GLORYS grid, strict MatchTopo, and native CMEMS timestamps."""
from types import SimpleNamespace
import numpy as np
import pytest
import xarray as xr
from scipy.spatial import cKDTree

from fvtools.grid.glorys_grid import GLORYS
from fvtools.setup.nesting import cmems
from fvtools.setup.nesting.matchtopo import MatchTopo


def _source(*, dry=False):
    z = np.empty((1, 2, 3, 3), dtype=np.float64)
    z[:] = 34.5
    if dry:
        z[0, :, -1, -1] = np.nan
    return xr.Dataset(
        {'so': (('time', 'depth', 'latitude', 'longitude'), z)},
        coords={'time': np.array(['2023-07-01'], dtype='datetime64[D]'),
                'depth': np.array([0.5, 12.]),
                'latitude': np.array([2., 1., 0.]),
                'longitude': np.array([2., 1., 0.])},
    )


def test_glorys_dataset_keeps_sorted_mask_aligned():
    ds = _source(dry=True).sortby('latitude').sortby('longitude')
    grid = GLORYS(ds)
    assert grid.lon.shape == (3, 3)
    assert grid.land_mask.shape == (3, 3)
    assert grid.land_mask[0, 0]
    assert not grid.land_mask[-1, -1]
    np.testing.assert_array_equal(grid.depth, [0.5, 12.])


def test_glorys_can_still_be_loaded_from_a_netcdf_path(tmp_path):
    ds = _source().sortby('latitude').sortby('longitude')
    file = tmp_path / 'glorys_grid.nc'
    ds.to_netcdf(file, engine='scipy')
    from_file = GLORYS(file)
    from_ds = GLORYS(ds)
    np.testing.assert_array_equal(from_file.lon, from_ds.lon)
    np.testing.assert_array_equal(from_file.land_mask, from_ds.land_mask)
    np.testing.assert_array_equal(from_file.h_rho, from_ds.h_rho)


def test_native_times_preserved_and_filtered_without_interpolation():
    src = np.array(['2023-07-01T00', '2023-07-02T00', '2023-07-04T00'],
                   dtype='datetime64[h]')
    idx = cmems._native_time_indices(src,
        start_time='2023-07-01T12', stop_time='2023-07-04T00')
    np.testing.assert_array_equal(idx, [1, 2])
    np.testing.assert_array_equal(src[idx], src[[1, 2]])
    assert not hasattr(cmems, '_time_bracket')
    assert not hasattr(cmems, '_output_times')
    with pytest.raises(ValueError, match='strictly increasing'):
        cmems._native_time_indices(src[[0, 0]])
    with pytest.raises(ValueError, match='coverage'):
        cmems._native_time_indices(src, stop_time='2023-07-05')


class _Grid:
    def __init__(self):
        self.x = np.array([0.5, 0.75, 1., 1.25, 1.5])
        self.y = np.full(5, 0.5)
        self.h = np.full(5, 40.)
        self.obc_nodes = np.array([0])
        self.info = {'casename': 'test'}
        self.read_sigma_count = 0
        self.saved = False
        self.Proj = lambda lon, lat: (lon, lat)

    def read_sigma(self):
        self.read_sigma_count += 1

    def find_nearest(self, x, y, grid='node'):
        return cKDTree(np.column_stack((self.x, self.y))).query(
            np.column_stack((x, y)))[1]

    def write_bath(self, **kwargs):
        self.write_kwargs = kwargs

    def to_npy(self):
        self.saved = True


class _Nest:
    def __init__(self):
        self.R = 0.3
        self.x = np.array([0.5, 0.75])
        self.y = np.array([0.5, 0.5])
        self.h = None
        self.inherit_calls = 0

    def inherit_sig(self, M):
        self.inherit_calls += 1


def test_matchtopo_glorys_updates_depth_sigma_and_source_flag():
    grid = GLORYS(_source().sortby('latitude').sortby('longitude'))
    M, NEST = _Grid(), _Nest()
    # The deepest wet GLORYS z-level is 12 m at all source points.
    matcher = MatchTopo(grid, M, NEST, latlon=True)
    matcher.add_GLORYS_bathymetry_to_FVCOM_and_NEST()
    np.testing.assert_allclose(M.h[:3], [12., 12., 12.])
    np.testing.assert_allclose(NEST.h, [12., 12.])
    assert M.info.get('true if updated by glorys') is True
    assert not M.info.get('true if updated by roms', False)
    assert M.saved and M.read_sigma_count == 1 and NEST.inherit_calls == 1
    matcher.add_GLORYS_bathymetry_to_FVCOM_and_NEST()
    assert M.read_sigma_count == 1  # no repeated modification


def test_glorys_missing_wet_cell_fails_before_modifying_depth():
    grid = GLORYS(_source(dry=True).sortby('latitude').sortby('longitude'))
    M, NEST = _Grid(), _Nest()
    matcher = MatchTopo(grid, M, NEST, latlon=True)
    with pytest.raises(ValueError, match='interpolation|outside|No complete'):
        matcher.add_GLORYS_bathymetry_to_FVCOM_and_NEST()
    np.testing.assert_array_equal(M.h, np.full(5, 40.))
    assert not M.saved


def test_roms_matching_uses_original_roms_flag():
    M, NEST = _Grid(), _Nest()
    M.info['true if updated by roms'] = True
    matcher = MatchTopo(GLORYS(_source()), M, NEST, latlon=True)
    with pytest.raises(ValueError, match='ROMS-matched'):
        matcher.add_GLORYS_bathymetry_to_FVCOM_and_NEST()
    np.testing.assert_array_equal(M.h, np.full(5, 40.))
