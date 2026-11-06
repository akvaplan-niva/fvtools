"""Regression checks for extracting the small compatibility and nesting classes.

All synthetic tests use tiny meshes. Optional reference test uses the same
trusted M.npy file as the existing FVCOM grid regression suite.
"""

import numpy as np
import pytest

from fvtools.grid.fvcom_grd import (
    FVCOM_grid, NEST_grid,
    LegacyPropertyAliases, LegacyFunctions, OutputLoader,
    LegacyNestPropertyAliases, LoadNest, NestROMS2FVCOM,
)
from fvtools.grid.fvcom.netcdf_io import FVCOMNetCDFIO
from fvtools.grid.fvcom._components.compatibility import (
    LegacyPropertyAliases as MovedAliases,
    LegacyFunctions as MovedFunctions,
    OutputLoader as MovedOutputLoader,
)
from fvtools.grid.fvcom._components.nesting import (
    LegacyNestPropertyAliases as MovedNestAliases,
    LoadNest as MovedLoadNest,
    NestROMS2FVCOM as MovedNestROMS2FVCOM,
    NEST_grid as MovedNestGrid,
)


@pytest.fixture
def mother(small_grid_file):
    return FVCOM_grid(str(small_grid_file))


@pytest.fixture
def nest_file(tmp_path, mother):
    """Five nesting nodes over a square: four corners and one central node."""
    x = np.r_[mother.x, mother.x.mean()]
    y = np.r_[mother.y, mother.y.mean()]
    tri = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=int)
    payload = {
        'xn': x, 'yn': y, 'nv': tri,
        'h': np.r_[mother.h, mother.h.mean()],
        'nid': np.array([0, 1, 2, 3, 0]),
        'cid': np.array([0, 0, 1, 1]),
        'R': np.array([0.25, 0.75]),
        'oend1': np.array([0, 1]),
        'oend2': np.array([2, 3]),
        'info': {'reference': mother.reference, 'casename': 'tiny_nest'},
    }
    path = tmp_path / 'ngrd.npy'
    np.save(path, payload)
    return path


@pytest.fixture
def nest(nest_file, mother):
    return NEST_grid(str(nest_file), M=mother)


def test_public_compatibility_class_aliases_and_fvcom_mro():
    assert LegacyPropertyAliases is MovedAliases
    assert LegacyFunctions is MovedFunctions
    assert OutputLoader is MovedOutputLoader
    assert issubclass(OutputLoader, FVCOMNetCDFIO)
    assert issubclass(FVCOM_grid, LegacyPropertyAliases)
    assert issubclass(FVCOM_grid, LegacyFunctions)
    assert issubclass(FVCOM_grid, OutputLoader)
    assert FVCOM_grid.__mro__[1].__name__ == 'GridLoader'
    assert OutputLoader.load_netCDF is FVCOMNetCDFIO.load_netCDF


def test_public_nesting_class_aliases_and_nest_mro():
    assert NEST_grid is MovedNestGrid
    assert LoadNest is MovedLoadNest
    assert NestROMS2FVCOM is MovedNestROMS2FVCOM
    assert LegacyNestPropertyAliases is MovedNestAliases
    assert NEST_grid.__mro__[1] is LoadNest
    assert issubclass(NEST_grid, LegacyNestPropertyAliases)
    assert issubclass(NEST_grid, LegacyPropertyAliases)
    assert issubclass(NEST_grid, NestROMS2FVCOM)


def test_legacy_sigma_aliases_read_current_cell_coordinates(mother):
    np.testing.assert_allclose(mother.siglev_center, mother.siglev_c)
    np.testing.assert_allclose(mother.siglay_center, mother.siglay_c)
    assert mother.siglev_center.shape == (mother.cell_number, mother.siglev.shape[1])


def test_cell2node_has_equal_cell_weights(mother):
    # tri = [[0, 1, 2], [0, 2, 3]], so nodes 0 and 2 are shared.
    got = mother.cell2node(np.array([2., 6.]))
    np.testing.assert_allclose(got, [4., 2., 4., 6.])


@pytest.mark.parametrize('values', [
    np.array([1.]),
    np.array([1., 2., 3.]),
    np.ones((2, 1)),
])
def test_cell2node_rejects_wrong_shape(mother, values):
    with pytest.raises(ValueError, match='one value per cell'):
        mother.cell2node(values)


def test_cell2node_returns_nan_for_isolated_node(mother):
    mother.x = np.r_[mother.x, 500500.]
    mother.y = np.r_[mother.y, 7700500.]
    got = mother.cell2node(np.array([2., 6.]))
    np.testing.assert_allclose(got[:4], [4., 2., 4., 6.])
    assert np.isnan(got[4])


def test_nest_npy_load_preserves_geometry_and_connectivity(nest, nest_file, mother):
    payload = np.load(nest_file, allow_pickle=True).item()  # created locally in fixture
    assert nest.filepath == str(nest_file)
    np.testing.assert_array_equal(nest.x, payload['xn'])
    np.testing.assert_array_equal(nest.y, payload['yn'])
    np.testing.assert_array_equal(nest.tri, payload['nv'])
    np.testing.assert_array_equal(nest.nid, payload['nid'])
    np.testing.assert_array_equal(nest.cid, payload['cid'])
    np.testing.assert_allclose(nest.h, payload['h'])
    assert nest.x.shape == (5,)
    assert nest.xc.shape == (4,)
    assert nest.Proj is mother.Proj


def test_nest_npy_coordinate_aliases_are_column_vectors(nest):
    assert nest.lonn.shape == (5, 1)
    assert nest.latn.shape == (5, 1)
    np.testing.assert_allclose(nest.lonn[:, 0], nest.lon)
    np.testing.assert_allclose(nest.latn[:, 0], nest.lat)
    assert nest.lonc.shape == (4,)
    assert nest.latc.shape == (4,)


def test_nest_inherits_sigma_from_nearest_mother_nodes(nest, mother):
    nearest = mother.find_nearest(nest.x, nest.y)
    np.testing.assert_allclose(nest.siglay, mother.siglay[nearest])
    np.testing.assert_allclose(nest.siglev, mother.siglev[nearest])
    np.testing.assert_allclose(nest.siglay_center, np.mean(nest.siglay[nest.tri], axis=1))
    np.testing.assert_allclose(nest.zeta, np.zeros(nest.h.shape))


def test_nest_can_load_without_mother_grid(nest_file):
    bare = NEST_grid(str(nest_file))
    assert bare.x.shape == (5,)
    assert bare.tri.shape == (4, 3)
    assert bare.reference.lower() == 'epsg:32633'
    assert bare.lon.shape == (5,)
    assert bare.lat.shape == (5,)


def test_nest_loader_ignores_missing_optional_fields(tmp_path):
    x = np.array([500000., 500100., 500000.])
    y = np.array([7700000., 7700000., 7700100.])
    payload = {'xn': x, 'yn': y, 'nv': np.array([[0, 1, 2]]), 'h': np.ones(3)}
    path = tmp_path / 'minimal.npy'
    np.save(path, payload)
    result = NEST_grid(str(path))
    np.testing.assert_array_equal(result.x, x)
    np.testing.assert_array_equal(result.tri, payload['nv'])
    assert result.casename == 'FVCOM experiment'


def test_nest_nearest_obc_distance_uses_projected_meters(nest):
    nest.x_obc = np.array([nest.x[0], nest.x[2]])
    nest.y_obc = np.array([nest.y[0], nest.y[2]])
    distances = nest._distance_to_nearest_obc(nest.x, nest.y)
    expected = np.minimum(np.hypot(nest.x - nest.x[0], nest.y - nest.y[0]),
                          np.hypot(nest.x - nest.x[2], nest.y - nest.y[2]))
    np.testing.assert_allclose(distances, expected, atol=1e-12)


def test_nest_linear_weight_interpolation(nest):
    node_distances = np.array([0., 2.5, 5., 7.5, 10.])
    cell_distances = np.array([0., 5., 7.5, 12.])
    nest._compute_weights(.1, .02, node_distances, cell_distances)
    np.testing.assert_allclose(nest.weight_node, [.1, .08, .06, .04, .02])
    np.testing.assert_allclose(nest.weight_cell, [.1, .06, .04, .02])


def test_nest_obc_cell_and_node_weights_propagate(nest, mother):
    nest.get_obc_nodes(mother)
    np.testing.assert_array_equal(nest.node_obc_in_nest, [0, 1, 2, 3])
    nest.get_cube_connected_to_obc()
    np.testing.assert_array_equal(nest.cell_obc_to_one, [0, 1, 2, 3])
    np.testing.assert_array_equal(nest.node_obc_to_one, [0, 1, 2, 3, 4])
    nest.weight_node = np.full(5, .1)
    nest.weight_cell = np.full(4, .05)
    nest.set_weights_in_obc_cube_to_one()
    np.testing.assert_allclose(nest.weight_node, 1.)
    np.testing.assert_allclose(nest.weight_cell, 1.)


def test_real_fvcom_grid_can_supply_nest_sigma(reference_paths, tmp_path):
    _, npy = reference_paths
    mother = FVCOM_grid(str(npy))
    first = np.asarray(mother.tri[0], dtype=int)
    payload = {
        'xn': mother.x[first].copy(),
        'yn': mother.y[first].copy(),
        'nv': np.array([[0, 1, 2]], dtype=int),
        'h': mother.h[first].copy(),
        'info': {'reference': mother.reference},
    }
    path = tmp_path / 'reference_nest.npy'
    np.save(path, payload)
    nest = NEST_grid(str(path), M=mother)
    np.testing.assert_allclose(nest.siglev, mother.siglev[first])
    np.testing.assert_allclose(nest.siglay, mother.siglay[first])
    assert nest.tri.shape == (1, 3)
