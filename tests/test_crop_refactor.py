"""Regression tests for the CropGrid mixin extracted from fvcom_grd.

A five-node, four-element fan has one exposed edge per triangle. That
avoids conflating the existing two-exterior-edge filtering rule with
remapping, vertical data transfer or child-grid initialization.
"""

import matplotlib.tri
import numpy as np
import pytest

from fvtools.grid.fvcom_grd import CropGrid, FVCOM_grid
from fvtools.grid._components.cropping import CropGrid as ExtractedCropGrid


@pytest.fixture
def fan_grid():
    # One interior node and four exterior nodes, anticlockwise triangles.
    x = np.array([500000.0, 500100.0, 500000.0, 499900.0, 500000.0])
    y = np.array([7700000.0, 7700000.0, 7700100.0, 7700000.0, 7699900.0])
    tri = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=int)
    h = np.array([20.0, 24.0, 28.0, 32.0, 36.0])
    siglev = np.tile([0.0, -0.5, -1.0], (5, 1))
    siglay = (siglev[:, :-1] + siglev[:, 1:]) / 2
    grid = FVCOM_grid(
        x=x, y=y, tri=tri, h=h, siglev=siglev, siglay=siglay,
        reference='epsg:32633', obc_nodes=[],
    )
    grid.casename = 'fan_mesh'
    return grid


def test_cropgrid_public_identity_and_mro():
    assert CropGrid is ExtractedCropGrid
    assert issubclass(FVCOM_grid, CropGrid)
    assert FVCOM_grid.subgrid is CropGrid.subgrid


def test_default_crop_mapping_and_setters(fan_grid):
    grid = fan_grid
    np.testing.assert_array_equal(grid.cropped_nodes, np.array(0))
    np.testing.assert_array_equal(grid.cropped_cells, np.array(0))
    grid.cropped_nodes = np.array([4, 2], dtype=int)
    grid.cropped_cells = np.array([3, 1], dtype=int)
    np.testing.assert_array_equal(grid.cropped_nodes, [4, 2])
    np.testing.assert_array_equal(grid.cropped_cells, [3, 1])


def test_triangle_filter_preserves_one_exterior_edge_fan(fan_grid):
    keep = fan_grid._return_valid_triangles(fan_grid.x, fan_grid.y, fan_grid.tri)
    np.testing.assert_array_equal(keep, [0, 1, 2, 3])
    # Two triangles forming a square each have exactly two exposed sides.
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    np.testing.assert_array_equal(fan_grid._return_valid_triangles(x, y, tri), [])


def test_remap_keeps_cell_order_and_rebuilds_node_connectivity(fan_grid):
    grid = fan_grid
    cells = np.array([3, 1, 0, 2], dtype=int)
    x, y, tri, old_nodes, old_cells = grid.remap(grid.x, grid.y, grid.tri, cells)
    np.testing.assert_array_equal(old_cells, cells)
    np.testing.assert_array_equal(old_nodes, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(x, grid.x[old_nodes])
    np.testing.assert_allclose(y, grid.y[old_nodes])
    np.testing.assert_array_equal(old_nodes[tri], grid.tri[old_cells])
    np.testing.assert_array_equal(tri, grid.tri[cells])


def test_subgrid_preserves_coordinates_connectivity_and_sigma(fan_grid):
    grid = fan_grid
    parent_x = grid.x.copy()
    parent_tri = grid.tri.copy()
    parent_h = grid.h.copy()
    parent_sig = grid.siglay.copy()
    parent_z = grid.siglev.copy()
    cells = np.array([3, 0, 1, 2], dtype=int)

    sub = grid.subgrid(cells=cells)
    assert isinstance(sub, FVCOM_grid)
    assert sub is not grid
    assert sub.reference == grid.reference
    assert sub.casename == 'Cropped fan_mesh'
    np.testing.assert_array_equal(sub.cropped_nodes, np.arange(grid.node_number))
    np.testing.assert_array_equal(sub.cropped_cells, cells)
    np.testing.assert_array_equal(sub.cropped_nodes[sub.tri], grid.tri[cells])
    np.testing.assert_allclose(sub.h, grid.h[sub.cropped_nodes])
    np.testing.assert_allclose(sub.siglay, grid.siglay[sub.cropped_nodes])
    np.testing.assert_allclose(sub.siglev, grid.siglev[sub.cropped_nodes])
    np.testing.assert_allclose(sub.x, grid.x[sub.cropped_nodes])
    np.testing.assert_allclose(sub.y, grid.y[sub.cropped_nodes])
    np.testing.assert_allclose(sub.lon, grid.lon[sub.cropped_nodes], atol=1e-10)
    np.testing.assert_allclose(sub.lat, grid.lat[sub.cropped_nodes], atol=1e-10)
    # The operation must not mutate the source mesh or its vertical fields.
    np.testing.assert_array_equal(grid.x, parent_x)
    np.testing.assert_array_equal(grid.tri, parent_tri)
    np.testing.assert_array_equal(grid.h, parent_h)
    np.testing.assert_array_equal(grid.siglay, parent_sig)
    np.testing.assert_array_equal(grid.siglev, parent_z)


def test_subgrid_box_selection_matches_cell_selection(fan_grid):
    grid = fan_grid
    by_box = grid.subgrid(
        xlim=[grid.x.min(), grid.x.max()],
        ylim=[grid.y.min(), grid.y.max()],
    )
    np.testing.assert_array_equal(by_box.cropped_cells, np.arange(grid.cell_number))
    np.testing.assert_array_equal(by_box.cropped_nodes[by_box.tri], grid.tri)


def test_subgrid_repeated_cropping_keeps_parent_mesh_indices(fan_grid):
    grid = fan_grid
    first = grid.subgrid(cells=np.array([2, 0, 3, 1]))
    second = first.subgrid(cells=np.array([3, 1, 0, 2]))
    np.testing.assert_array_equal(
        second.cropped_nodes[second.tri], grid.tri[second.cropped_cells]
    )
    np.testing.assert_allclose(second.h, grid.h[second.cropped_nodes])
    np.testing.assert_allclose(second.siglay, grid.siglay[second.cropped_nodes])
    assert second.casename.startswith('Cropped')


def test_subgrid_has_independent_kdtree_caches(fan_grid):
    parent = fan_grid
    parent_tree = parent.node_tree
    child = parent.subgrid(cells=np.arange(parent.cell_number))
    assert child.node_tree is not parent_tree
    np.testing.assert_array_equal(
        child.find_nearest(parent.x, parent.y), np.arange(parent.node_number)
    )
    child.x = child.x + 1.0
    assert 'node_tree' not in child.__dict__
    assert parent.node_tree is parent_tree


def test_coastline_helper_updates_only_selected_nodes(fan_grid):
    grid = fan_grid
    mask = np.array([False, True, True, False, False])
    grid._update_coastline_nodes(mask)
    assert grid.coastline_nodes == [1, 2]
    np.testing.assert_array_equal(grid.tri, [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])


def test_ocean_boundary_helpers_identify_noncoastal_boundary_nodes(fan_grid):
    grid = fan_grid
    grid._update_coastline_nodes(np.array([False, True, True, False, False]))
    ocean, touching_cells = grid._find_ocean_obc_nodes()
    np.testing.assert_array_equal(np.sort(ocean), [3, 4])
    np.testing.assert_array_equal(touching_cells, np.arange(grid.cell_number))
    # An OBC point on triangle (0, 2, 3) connects to coastal node 2.
    connected = grid._connect_ocean_obc_to_land(np.array([3]), touching_cells)
    np.testing.assert_array_equal(np.sort(connected), [2, 3])


def test_reference_mesh_fan_remap_preserves_original_indices(reference_paths):
    # The 48k-node reference mesh is a trusted uploaded fixture. Find an
    # interior star: each triangle surrounding that node has one exposed edge.
    _, npy_path = reference_paths
    grid = FVCOM_grid(str(npy_path))
    counts = np.bincount(grid.tri.ravel(), minlength=grid.node_number)
    selected = None
    for node in np.flatnonzero((counts >= 5) & (counts <= 15))[:250]:
        cells = np.flatnonzero(np.any(grid.tri == node, axis=1))
        vertices = np.unique(grid.tri[cells])
        local = np.searchsorted(vertices, grid.tri[cells])
        neighbors = matplotlib.tri.Triangulation(
            grid.x[vertices], grid.y[vertices], local,
        ).neighbors
        if np.all(np.count_nonzero(neighbors == -1, axis=1) == 1):
            selected = cells
            break
    if selected is None:
        pytest.fail('No well-formed interior node fan among first 250 candidates')
    x, y, tri, nodes, cells = grid.remap(grid.x, grid.y, grid.tri, selected)
    np.testing.assert_array_equal(cells, selected)
    np.testing.assert_array_equal(nodes[tri], grid.tri[cells])
    np.testing.assert_allclose(x, grid.x[nodes])
    np.testing.assert_allclose(y, grid.y[nodes])
    assert len(cells) >= 5
