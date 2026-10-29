"""Regression tests for extracting FVCOM coastline and open-boundary mixins.

The tests exercise both the small synthetic mesh and the optional real-world
48,803-node mesh supplied separately under tests/reference/.
"""

import numpy as np
import pytest

from fvtools.grid.fvcom_grd import FVCOM_grid, CoastLine, OBC
from fvtools.grid._components.coastline import CoastLine as CoastLineImplementation
from fvtools.grid._components.obc import OBC as OBCImplementation


def test_public_mixins_remain_identical_and_keep_mro():
    assert CoastLine is CoastLineImplementation
    assert OBC is OBCImplementation
    mro = FVCOM_grid.__mro__
    assert mro.index(CoastLine) < mro.index(OBC)
    assert FVCOM_grid._get_nodestrings is OBC._get_nodestrings
    assert FVCOM_grid.get_land_polygons is CoastLine.get_land_polygons


def test_existing_nodestring_order_and_adjacent_cellstrings(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    assert grid.n_obc == 2
    assert [s.tolist() for s in grid.nodestrings] == [[0, 1], [2, 3]]
    assert grid.cellstrings == [[0], [1]]
    assert grid.cellstrings is grid.cellstrings  # cached once per mesh


def test_obc_coordinate_views_follow_nodestring_indices(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    for name, node_values in (
        ('x_obc', grid.x), ('y_obc', grid.y),
        ('lat_obc', grid.lat), ('lon_obc', grid.lon),
    ):
        result = getattr(grid, name)
        assert len(result) == len(grid.nodestrings)
        for positions, segment in zip(result, grid.nodestrings):
            np.testing.assert_allclose(np.asarray(positions, dtype=float),
                                       node_values[segment])


def test_nodestring_reconstruction_follows_real_boundary_edges(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    segments = grid._get_nodestrings([2, 0, 1])
    assert len(segments) == 1
    np.testing.assert_array_equal(segments[0], [0, 1, 2])
    # Two separate OBC edges on the four-node square remain disconnected.
    np.testing.assert_array_equal(grid._get_nodestrings([0, 1])[0], [0, 1])
    np.testing.assert_array_equal(grid._get_nodestrings([3, 2])[0], [2, 3])


@pytest.mark.parametrize('indices', [[-1], [4], [0, 2]])
def test_nodestring_reconstruction_rejects_invalid_or_interior_edges(
    small_grid_file, indices,
):
    grid = FVCOM_grid(str(small_grid_file))
    with pytest.raises(ValueError):
        grid._get_nodestrings(indices)


def test_closed_boundary_is_one_circular_nodestring(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    nodestrings = grid._get_nodestrings([0, 1, 2, 3])
    assert len(nodestrings) == 1
    # SMS represents a circular OBC by repeating its first node.
    np.testing.assert_array_equal(nodestrings[0], [0, 1, 2, 3, 0])
    # Every step must still be an actual mesh-boundary edge.
    boundary_edges = {
        tuple(sorted(map(int, edge))) for edge in grid.boundary_nodes
    }
    for left, right in zip(nodestrings[0][:-1], nodestrings[0][1:]):
        assert tuple(sorted((int(left), int(right)))) in boundary_edges


def test_boundary_segments_match_exact_exterior_triangle_edges(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    segments = grid.boundary_nodes
    actual = {tuple(sorted(map(int, edge))) for edge in segments}
    expected = {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert actual == expected
    assert grid.boundary_nodes is segments


def test_coastline_includes_open_boundary_endpoints(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    grid.obc_nodes = [0, 1]
    grid.nodestrings = [np.array([0, 1], dtype=int)]
    coast = np.asarray(grid.coastline_nodes, dtype=int)
    assert set(coast) == {0, 1, 2, 3}
    assert {0, 1}.issubset(set(coast))
    np.testing.assert_array_equal(grid.full_model_boundary, [0, 1, 2, 3])


def test_polygon_and_is_on_mesh_on_simple_square(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    polygons = grid.get_land_polygons()
    assert len(polygons) == 1
    np.testing.assert_allclose(polygons[0].area, 10000.0, atol=1e-8)
    np.testing.assert_allclose(grid.model_boundary.length, 400.0, atol=1e-8)

    x = np.array([500025., 500050., 500150.])
    y = np.array([7700025., 7700050., 7700050.])
    found = grid.is_on_mesh(x, y)
    assert found.shape == (2, 2)
    np.testing.assert_allclose(found, [[500025., 500050.], [7700025., 7700050.]])


def test_mesh_reassignment_clears_coastal_and_obc_spatial_caches(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    _ = grid.boundary_nodes
    _ = grid.coastline_nodes
    _ = grid.cellstrings
    for key in ('boundary_nodes', 'coastline_nodes', 'cellstrings'):
        assert key in grid.__dict__

    # Switching the diagonal changes connectivity, but not the square outline.
    grid.tri = np.array([[0, 1, 3], [1, 2, 3]], dtype=int)
    for key in ('boundary_nodes', 'coastline_nodes', 'cellstrings'):
        assert key not in grid.__dict__
    exterior = {tuple(sorted(map(int, edge))) for edge in grid.boundary_nodes}
    assert exterior == {(0, 1), (1, 2), (2, 3), (0, 3)}


def test_reference_nodestrings_follow_mesh_boundary_edges(reference_paths):
    _, npy_path = reference_paths
    grid = FVCOM_grid(str(npy_path))  # The user-provided M.npy is trusted.
    assert grid.n_obc == 3
    assert [len(s) for s in grid.nodestrings] == [25, 57, 17]
    edges = {tuple(sorted(map(int, edge))) for edge in grid.boundary_nodes}
    for segment in grid.nodestrings:
        for left, right in zip(segment[:-1], segment[1:]):
            assert tuple(sorted((int(left), int(right)))) in edges
