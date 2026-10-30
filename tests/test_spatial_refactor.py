"""Regression tests for the relocated GridProximity and CellTriangulation.

Tests exercise the production FVCOM_grid mixins, not reimplemented algorithms.
Small direct-initialization fixtures avoid requiring a big mesh on every run.
"""

import matplotlib.tri as mtri
import numpy as np
import pytest

from fvtools.grid.fvcom_grd import FVCOM_grid, GridProximity, CellTriangulation


@pytest.fixture
def spatial_grid():
    """A square 3x3-node mesh with eight anti-clockwise FVCOM elements."""
    x = 500_000.0 + 100.0 * np.tile(np.arange(3), 3)
    y = 7_700_000.0 + 100.0 * np.repeat(np.arange(3), 3)
    tri = []
    for j in range(2):
        for i in range(2):
            bl = 3 * j + i
            br, tl, tr = bl + 1, bl + 3, bl + 4
            tri += [[bl, br, tr], [bl, tr, tl]]
    tri = np.asarray(tri, dtype=np.int64)
    n = len(x)
    sigma_levels = np.tile([0.0, -0.5, -1.0], (n, 1))
    grid = FVCOM_grid(
        x=x, y=y, tri=tri,
        lon=np.full(n, 15.0), lat=np.full(n, 69.0),
        h=np.full(n, 20.0),
        siglev=sigma_levels,
        siglay=0.5 * (sigma_levels[:, :-1] + sigma_levels[:, 1:]),
    )
    return grid


def test_find_nearest_nodes_and_cells_scalar_and_array(spatial_grid):
    grid = spatial_grid
    assert grid.node_number == 9
    assert grid.cell_number == 8
    assert int(grid.find_nearest(grid.x[0], grid.y[0])) == 0
    np.testing.assert_array_equal(
        grid.find_nearest(grid.x[[0, 4, 8]], grid.y[[0, 4, 8]]), [0, 4, 8]
    )
    np.testing.assert_array_equal(
        grid.find_nearest(grid.xc[[0, 3, 7]], grid.yc[[0, 3, 7]], grid="cell"),
        [0, 3, 7],
    )


def test_find_within_unique_sorted_and_validates_arguments(spatial_grid):
    grid = spatial_grid
    indices = grid.find_within(
        [grid.x[4], grid.x[0], grid.x[4]],
        [grid.y[4], grid.y[0], grid.y[4]],
        r=1e-6,
    )
    np.testing.assert_array_equal(indices, [0, 4])

    cell_indices = grid.find_within(
        grid.xc[[1, 6]], grid.yc[[1, 6]], r=1e-6, grid="cell"
    )
    np.testing.assert_array_equal(cell_indices, [1, 6])
    assert grid.find_within(-100.0, -100.0, r=1.0).size == 0

    for radius in (None, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            grid.find_within(grid.x[0], grid.y[0], r=radius)
    with pytest.raises(ValueError, match="grid"):
        grid.find_within(grid.x[0], grid.y[0], r=1.0, grid="unknown")


def test_bounds_and_cell_box_lookup(spatial_grid):
    grid = spatial_grid
    assert bool(grid.isinside(grid.x[0], grid.y[0]))
    assert not bool(grid.isinside(grid.x.min() - 1.0, grid.y[0]))
    assert bool(grid.isinside(grid.x.min() - 1.0, grid.y[0], x_buffer=2.0))
    np.testing.assert_array_equal(
        grid.isinside(
            [grid.x.min() - 1.0, grid.x.max() + 1.0],
            [grid.y.min(), grid.y.max()],
        ),
        [False, False],
    )
    xlim = [grid.xc.min() - 1, grid.xc.mean()]
    ylim = [grid.yc.min() - 1, grid.yc.max() + 1]
    expected = np.where(
        (grid.xc >= min(xlim)) & (grid.xc <= max(xlim))
        & (grid.yc >= min(ylim)) & (grid.yc <= max(ylim))
    )[0]
    np.testing.assert_array_equal(grid._find_cells_in_box(xlim, ylim), expected)
    np.testing.assert_array_equal(
        grid._find_cells_in_box(xlim[::-1], ylim[::-1]), expected
    )


def test_clockwise_connectivity_is_reoriented(spatial_grid):
    grid = spatial_grid
    original = grid.tri.copy()
    assert grid._test_nv(grid.x, grid.y, original).size == 0
    grid.tri = original[:, [0, 2, 1]]
    assert grid._test_nv(grid.x, grid.y, grid.tri).size == grid.cell_number
    grid._check_nv()
    np.testing.assert_array_equal(grid.tri, original)
    assert grid._test_nv(grid.x, grid.y, grid.tri).size == 0


def test_spatial_trees_are_lazy_cached_and_invalidated(spatial_grid):
    grid = spatial_grid
    assert "node_tree" not in grid.__dict__
    assert "cell_tree" not in grid.__dict__
    node_tree, cell_tree = grid.node_tree, grid.cell_tree
    assert node_tree is grid.node_tree
    assert cell_tree is grid.cell_tree

    grid.x = grid.x + np.linspace(0.0, 2.0, grid.node_number)
    assert "node_tree" not in grid.__dict__
    assert "cell_tree" not in grid.__dict__
    assert grid.node_tree is not node_tree
    assert grid.cell_tree is not cell_tree
    assert int(grid.find_nearest(grid.x[8], grid.y[8])) == 8

    _ = grid.node_tree, grid.cell_tree
    grid.tri = grid.tri[:, [0, 2, 1]]
    assert "node_tree" not in grid.__dict__
    assert "cell_tree" not in grid.__dict__


def test_cell_triangulation_mask_and_lazy_cache(spatial_grid, monkeypatch):
    import fvtools.util.tools as tools

    grid = spatial_grid
    unmasked = mtri.Triangulation(grid.xc, grid.yc).triangles
    captured = []

    def fake_mask_land_tris(x, y, tri, ctri):
        np.testing.assert_array_equal(x, grid.x)
        np.testing.assert_array_equal(y, grid.y)
        np.testing.assert_array_equal(tri, grid.tri)
        np.testing.assert_array_equal(ctri, unmasked)
        captured.append(True)
        return ctri[::2]

    monkeypatch.setattr(tools, "mask_land_tris", fake_mask_land_tris)
    np.testing.assert_array_equal(grid.ctri, unmasked[::2])
    first = grid.ctri
    assert grid.ctri is first
    assert len(captured) == 1
    np.testing.assert_array_equal(grid.mtri_cells.triangles, first)
    assert grid.mtri_cells is grid.mtri_cells


def test_ctri_setter_and_mesh_edit_invalidate_cell_triangulation(spatial_grid, monkeypatch):
    import fvtools.util.tools as tools

    # Avoid compiling geometry routines: this is a cache lifecycle test.
    monkeypatch.setattr(tools, "mask_land_tris", lambda x, y, tri, ctri: ctri)
    grid = spatial_grid
    first_ctri = grid.ctri
    first_mtri = grid.mtri_cells
    assert "_ctri" in grid.__dict__
    assert "mtri_cells" in grid.__dict__

    custom_ctri = first_ctri[:2].copy()
    grid.ctri = custom_ctri
    assert "mtri_cells" not in grid.__dict__
    np.testing.assert_array_equal(grid.mtri_cells.triangles, custom_ctri)
    assert grid.mtri_cells is not first_mtri

    _ = grid.node_tree, grid.cell_tree
    grid.y = grid.y + np.linspace(0.0, 3.0, grid.node_number)
    for name in ("_ctri", "mtri_cells", "node_tree", "cell_tree"):
        assert name not in grid.__dict__
    assert grid.ctri is not first_ctri
    np.testing.assert_array_equal(
        grid.mtri_cells.triangles, grid.ctri
    )


def test_reference_mesh_proximity_smoke(reference_paths):
    """Optional: no expensive cell-centre Delaunay, just real-world KD queries."""
    _, npy_path = reference_paths
    grid = FVCOM_grid(str(npy_path))
    nodes = np.array([0, grid.node_number // 2, grid.node_number - 1])
    np.testing.assert_array_equal(
        grid.find_nearest(grid.x[nodes], grid.y[nodes]), nodes
    )
    cells = np.array([0, grid.cell_number // 2, grid.cell_number - 1])
    np.testing.assert_array_equal(
        grid.find_nearest(grid.xc[cells], grid.yc[cells], grid="cell"), cells
    )
    np.testing.assert_array_equal(
        grid.find_within(grid.x[nodes], grid.y[nodes], r=0.0), nodes
    )
