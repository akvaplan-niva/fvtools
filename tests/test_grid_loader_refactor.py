"""Regression tests for extracting GridLoader from fvcom_grd.py.

Tests existing loading behavior, not a rewritten or modified parser. The public
FVCOM_grid API and all 1-based/0-based indexing conventions are preserved.
"""

import numpy as np
import pytest
from netCDF4 import Dataset

from fvtools.grid.fvcom_grd import FVCOM_grid, GridLoader


def test_direct_initialization_produces_node_and_cell_coordinates(small_grid_file):
    original = FVCOM_grid(str(small_grid_file))
    grid = FVCOM_grid(
        x=original.x.copy(), y=original.y.copy(),
        tri=original.tri.copy(), lon=original.lon.copy(), lat=original.lat.copy(),
        h=original.h.copy(), siglay=original.siglay.copy(),
        siglev=original.siglev.copy(), obc_nodes=[0, 1],
    )
    assert grid.node_number == 4
    assert grid.cell_number == 2
    assert grid.obc_nodes == [0, 1]
    np.testing.assert_array_equal(grid.tri, original.tri)
    np.testing.assert_allclose(grid.xc, original.xc)
    np.testing.assert_allclose(grid.yc, original.yc)
    np.testing.assert_allclose(grid.hc, original.hc)
    np.testing.assert_allclose(grid.siglay, original.siglay)
    np.testing.assert_allclose(grid.siglev, original.siglev)


@pytest.mark.parametrize('mutation, message', [
    ('one_based_tri', 'Triangulation is invalid'),
    ('negative_depth', 'Minimum depth is invalid'),
    ('obc_not_list', 'obc_nodes must be given as a list'),
])
def test_direct_initialization_still_rejects_invalid_inputs(
    small_grid_file, mutation, message,
):
    grid = FVCOM_grid(str(small_grid_file))
    kwargs = dict(
        x=grid.x.copy(), y=grid.y.copy(), tri=grid.tri.copy(),
        h=grid.h.copy(), obc_nodes=[0, 1],
    )
    if mutation == 'one_based_tri':
        kwargs['tri'] = kwargs['tri'] + 1
    elif mutation == 'negative_depth':
        kwargs['h'][0] = -1.0
    else:
        kwargs['obc_nodes'] = (0, 1)
    with pytest.raises(AssertionError, match=message):
        grid._direct_initialization(**kwargs)


def test_2dm_loader_retains_nodestrings_and_zero_based_connectivity():
    from pathlib import Path
    mesh = Path(__file__).resolve().parent / 'data' / 'mini_nodestrings.2dm'
    grid = FVCOM_grid(str(mesh))
    np.testing.assert_array_equal(grid.tri, [[0, 1, 2], [0, 2, 3]])
    np.testing.assert_array_equal(grid.obc_nodes, [0, 1, 2, 3])
    assert [list(ns) for ns in grid.nodestrings] == [[0, 1], [2, 3]]


def test_txt_loader_retains_zero_based_indices(tmp_path):
    filename = tmp_path / 'simple.txt'
    filename.write_text(
        '4\n'
        '500000 7700000\n'
        '500100 7700000\n'
        '500100 7700100\n'
        '500000 7700100\n'
        '2\n'
        '0 1 2\n'
        '0 2 3\n', encoding='utf-8',
    )
    grid = FVCOM_grid(str(filename))
    assert grid.node_number == 4
    assert grid.cell_number == 2
    np.testing.assert_array_equal(grid.tri, [[0, 1, 2], [0, 2, 3]])
    np.testing.assert_allclose(grid.x, [500000, 500100, 500100, 500000])
    np.testing.assert_allclose(grid.y, [7700000, 7700000, 7700100, 7700100])


def test_netcdf_grid_loader_transposes_one_based_connectivity_and_sigma(
    tmp_path, small_grid_file,
):
    original = FVCOM_grid(str(small_grid_file))
    path = tmp_path / 'tiny_mesh.nc'
    with Dataset(path, 'w') as ds:
        ds.createDimension('node', original.node_number)
        ds.createDimension('nele', original.cell_number)
        ds.createDimension('three', 3)
        ds.createDimension('siglay', original.siglay.shape[1])
        ds.createDimension('siglev', original.siglev.shape[1])
        ds.createDimension('nobc', 2)
        for name, values in (
            ('x', original.x), ('y', original.y),
            ('lon', original.lon), ('lat', original.lat), ('h', original.h),
        ):
            ds.createVariable(name, 'f8', ('node',))[:] = values
        ds.createVariable('nv', 'i4', ('three', 'nele'))[:] = original.tri.T + 1
        ds.createVariable('obc_nodes', 'i4', ('nobc',))[:] = [1, 3]
        ds.createVariable('siglev', 'f8', ('siglev', 'node'))[:] = original.siglev.T
        ds.createVariable('siglay', 'f8', ('siglay', 'node'))[:] = original.siglay.T

    loaded = FVCOM_grid(str(path))
    np.testing.assert_array_equal(loaded.tri, original.tri)
    np.testing.assert_array_equal(loaded.obc_nodes, [0, 2])
    np.testing.assert_allclose(loaded.x, original.x)
    np.testing.assert_allclose(loaded.y, original.y)
    np.testing.assert_allclose(loaded.h, original.h)
    np.testing.assert_allclose(loaded.siglev, original.siglev)
    np.testing.assert_allclose(loaded.siglay, original.siglay)


def test_sigma_file_parser_supports_named_parameters_and_comments(tmp_path):
    path = tmp_path / 'sigma.dat'
    path.write_text(
        '! FVCOM config\n'
        'NUMBER OF SIGMA LEVELS = 5  ! comment\n'
        'SIGMA COORDINATE TYPE = GEOMETRIC # comment\n'
        'P_SIGMA = 2\n', encoding='utf-8',
    )
    kb, stype, params = GridLoader._read_sigma_configuration(path)
    assert (kb, stype) == (5, 'geometric')
    assert GridLoader._sigma_options(stype, params) == {'p_sigma': 2.0}


def test_read_sigma_geometric_has_correct_interfaces_and_midpoints(
    tmp_path, small_grid_file,
):
    grid = FVCOM_grid(str(small_grid_file))
    path = tmp_path / 'sigma.dat'
    path.write_text(
        'NUMBER OF SIGMA LEVELS = 5\n'
        'SIGMA COORDINATE TYPE = GEOMETRIC\n'
        'P_SIGMA = 2.0\n', encoding='utf-8',
    )
    grid.read_sigma(path)
    expected = np.array([0.0, -0.125, -0.5, -0.875, -1.0])
    assert grid.siglev.shape == (grid.node_number, 5)
    assert grid.siglay.shape == (grid.node_number, 4)
    np.testing.assert_allclose(grid.siglev, np.tile(expected, (grid.node_number, 1)))
    np.testing.assert_allclose(
        grid.siglay,
        np.tile(0.5 * (expected[:-1] + expected[1:]), (grid.node_number, 1)),
    )
    np.testing.assert_allclose(grid.siglev[:, 0], 0.0)
    np.testing.assert_allclose(grid.siglev[:, -1], -1.0)


def test_sigma_file_parser_rejects_invalid_kb_and_missing_parameters(tmp_path):
    path = tmp_path / 'sigma.dat'
    path.write_text('KB = 2\nSTYPE = UNIFORM\n', encoding='utf-8')
    with pytest.raises(ValueError, match='KB must be at least 3'):
        GridLoader._read_sigma_configuration(path)
    path.write_text('KB = 5\nSTYPE = GEOMETRIC\n', encoding='utf-8')
    kb, stype, params = GridLoader._read_sigma_configuration(path)
    assert (kb, stype) == (5, 'geometric')
    with pytest.raises(ValueError, match='GEOMETRIC sigma requires'):
        GridLoader._sigma_options(stype, params)


def test_real_grid_sigma_read_preserves_nodal_dimension(reference_paths, tmp_path):
    _, npy_file = reference_paths
    grid = FVCOM_grid(str(npy_file))
    original_nodes = grid.node_number
    path = tmp_path / 'uniform_sigma.dat'
    path.write_text('KB = 4\nSTYPE = UNIFORM\n', encoding='utf-8')
    grid.read_sigma(path)
    assert grid.node_number == original_nodes
    assert grid.siglev.shape == (original_nodes, 4)
    assert grid.siglay.shape == (original_nodes, 3)
    np.testing.assert_allclose(grid.siglev[0], [0.0, -1 / 3, -2 / 3, -1.0])
    np.testing.assert_allclose(grid.siglev[-1], [0.0, -1 / 3, -2 / 3, -1.0])
