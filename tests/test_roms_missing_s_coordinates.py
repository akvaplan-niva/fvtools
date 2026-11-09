"""Synthetic ROMS vertical-coordinate tests. No THREDDS connections."""
import numpy as np
import pytest
from netCDF4 import Dataset

from fvtools.grid.roms_grid import (
    ROMSbase,
    METNorKystV3Hindcast,
    compute_roms_s_coordinates,
    read_roms_s_coordinates,
)


def test_souza_vstretching5_matches_roms_40_levels():
    # Expected rho coordinates from a real 40-level Vstretching=5 ROMS file.
    sr, sw = compute_roms_s_coordinates(40, 5)
    np.testing.assert_allclose(
        sr[:4], [-0.97990385, -0.94028846, -0.90144231, -0.86336538],
        atol=5e-9, rtol=0,
    )
    np.testing.assert_allclose(sr[-1], -0.004903846153846154, atol=1e-14)
    np.testing.assert_allclose(sw[[0, 1, -2, -1]], [-1, -0.96, -0.01, 0])
    assert sr.shape == (40,)
    assert sw.shape == (41,)
    assert np.all(np.diff(sr) > 0)
    assert np.all(np.diff(sw) > 0)


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_other_stretching_versions_use_uniform_s_coordinates(version):
    sr, sw = compute_roms_s_coordinates(4, version)
    np.testing.assert_allclose(sr, [-0.875, -0.625, -0.375, -0.125])
    np.testing.assert_allclose(sw, [-1, -0.75, -0.5, -0.25, 0])


@pytest.mark.parametrize('n,version,match', [
    (0, 5, 'Invalid number'),
    (1, 5, 'at least two'),
    (4, 6, 'Unsupported ROMS Vstretching'),
])
def test_invalid_coordinate_parameters(n, version, match):
    with pytest.raises(ValueError, match=match):
        compute_roms_s_coordinates(n, version)


def _create_native_roms(path, *, include_rho=True, include_w=True,
                        version=5, version_name='Vstretching'):
    """Create the metadata/geometry needed by ROMSbase.load_grid_from_nc."""
    with Dataset(path, 'w') as ds:
        for name, count in (
            ('eta_rho', 2), ('xi_rho', 3), ('eta_u', 2), ('xi_u', 2),
            ('eta_v', 1), ('xi_v', 3), ('s_rho', 40), ('s_w', 41),
        ):
            ds.createDimension(name, count)
        for loc in ('rho', 'u', 'v'):
            dims = (f'eta_{loc}', f'xi_{loc}')
            shape = tuple(len(ds.dimensions[d]) for d in dims)
            for name in ('lon', 'lat'):
                ds.createVariable(f'{name}_{loc}', 'f8', dims)[:] = np.ones(shape)
            ds.createVariable(f'mask_{loc}', 'i4', dims)[:] = 1
        ds.createVariable('h', 'f8', ('eta_rho', 'xi_rho'))[:] = 100.
        ds.createVariable('angle', 'f8', ('eta_rho', 'xi_rho'))[:] = 0.
        ds.createVariable('Cs_r', 'f8', ('s_rho',))[:] = np.linspace(-1, 0, 40)
        ds.createVariable('Cs_w', 'f8', ('s_w',))[:] = np.linspace(-1, 0, 41)
        ds.createVariable('hc', 'f8')[()] = 10.0
        if version_name is not None:
            ds.createVariable(version_name, 'i4')[()] = version
        if include_rho:
            ds.createVariable('s_rho', 'f8', ('s_rho',))[:] = np.linspace(-.9, -.1, 40)
        if include_w:
            ds.createVariable('s_w', 'f8', ('s_w',))[:] = np.linspace(-1, 0, 41)


@pytest.mark.parametrize('include_rho,include_w', [
    (True, True), (True, False), (False, True), (False, False),
])
def test_native_grid_reads_present_levels_and_calculates_only_missing(
    tmp_path, include_rho, include_w,
):
    file = tmp_path / 'roms.nc'
    _create_native_roms(file, include_rho=include_rho, include_w=include_w)
    grid = ROMSbase()
    grid.load_grid_from_nc(file)
    calculated_rho, calculated_w = compute_roms_s_coordinates(40, 5)
    expected_rho = (np.linspace(-.9, -.1, 40)
                    if include_rho else calculated_rho)
    expected_w = (np.linspace(-1, 0, 41)
                  if include_w else calculated_w)
    np.testing.assert_array_equal(grid.s_rho, expected_rho)
    np.testing.assert_array_equal(grid.s_w, expected_w)
    assert grid.hc == 10.0
    assert grid.Vstretch == 5
    assert grid.h_rho.shape == (2, 3)
    assert not np.any(grid.rho_mask)


def test_native_grid_supports_legacy_vstretch_name(tmp_path):
    path = tmp_path / 'legacy.nc'
    _create_native_roms(path, include_rho=False, include_w=False,
                        version=4, version_name='Vstretch')
    grid = ROMSbase()
    grid.load_grid_from_nc(path)
    np.testing.assert_allclose(grid.s_w, np.linspace(-1, 0, 41))
    assert grid.Vstretch == 4


def test_stored_coordinates_need_no_vstretching_metadata(tmp_path):
    path = tmp_path / 'complete.nc'
    _create_native_roms(path, version_name=None)
    with Dataset(path) as ds:
        # No extra coordinate calculations when both variables are present.
        sr, sw = read_roms_s_coordinates(ds)
        np.testing.assert_array_equal(sr, np.linspace(-.9, -.1, 40))
        np.testing.assert_array_equal(sw, np.linspace(-1, 0, 41))


def test_missing_levels_require_stretching_metadata(tmp_path):
    path = tmp_path / 'missing_metadata.nc'
    with Dataset(path, 'w') as ds:
        ds.createDimension('s_rho', 4)
        ds.createDimension('s_w', 5)
        ds.createVariable('Cs_r', 'f8', ('s_rho',))[:] = np.zeros(4)
        ds.createVariable('Cs_w', 'f8', ('s_w',))[:] = np.zeros(5)
    with Dataset(path) as ds:
        with pytest.raises(ValueError, match='Vstretching/Vstretch metadata'):
            read_roms_s_coordinates(ds)


def test_collocated_hindcast_also_calculates_missing_s_w(tmp_path):
    path = tmp_path / 'hindcast.nc'
    with Dataset(path, 'w') as ds:
        for name, count in [('Y', 2), ('X', 3), ('s_rho', 40), ('s_w', 41)]:
            ds.createDimension(name, count)
        for name in ('lon', 'lat', 'h', 'sea_mask'):
            value = 1 if name == 'sea_mask' else 100
            ds.createVariable(name, 'f8', ('Y', 'X'))[:] = value
        ds.createVariable('Cs_r', 'f8', ('s_rho',))[:] = np.linspace(-1, 0, 40)
        ds.createVariable('Cs_w', 'f8', ('s_w',))[:] = np.linspace(-1, 0, 41)
        ds.createVariable('s_rho', 'f8', ('s_rho',))[:] = np.linspace(-.9, -.1, 40)
        ds.createVariable('hc', 'f8')[()] = 10
        ds.createVariable('Vstretching', 'i4')[()] = 5
    grid = METNorKystV3Hindcast()
    grid.load_grid_from_nc(path)
    np.testing.assert_array_equal(grid.s_rho, np.linspace(-.9, -.1, 40))
    np.testing.assert_allclose(grid.s_w, compute_roms_s_coordinates(40, 5)[1])
    assert grid.hc == 10
