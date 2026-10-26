"""Regression tests for extracting PlotFVCOM and ControlVolumePlotter.

Run headlessly; do not access map/WMS services or mutate user files.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import pytest

from fvtools.grid.fvcom_grd import (
    FVCOM_grid,
    PlotFVCOM,
    ControlVolumePlotter,
)

from fvtools.grid._components.plotting import (
    PlotFVCOM as PlottingMixin,
)

from fvtools.grid._components.control_volumes import (
    ControlVolumePlotter as ControlVolumeMixin,
    _find_cv_walls,
    _connect_cv_walls,
)


@pytest.fixture(autouse=True)
def close_test_figures():
    yield
    plt.close('all')


@pytest.fixture
def multi_cell_grid():
    """A three-by-three node mesh with eight cells and no external files."""
    coords = np.array([(i, j) for j in range(3) for i in range(3)], dtype=float)
    xy = coords * 100.0 + np.array([500000.0, 7700000.0])
    tri = []
    for j in range(2):
        for i in range(2):
            bottom_left = j * 3 + i
            bottom_right = bottom_left + 1
            top_left = bottom_left + 3
            top_right = top_left + 1
            tri.extend(((bottom_left, bottom_right, top_right),
                        (bottom_left, top_right, top_left)))
    tri = np.asarray(tri, dtype=np.int64)
    n = len(xy)
    sigma_levels = np.tile([0.0, -0.5, -1.0], (n, 1))
    sigma_layers = (sigma_levels[:, 1:] + sigma_levels[:, :-1]) / 2
    grid = FVCOM_grid(
        x=xy[:, 0], y=xy[:, 1], tri=tri,
        h=np.full(n, 20.0), siglev=sigma_levels, siglay=sigma_layers,
        obc_nodes=[],
    )
    grid.casename = 'synthetic_plotting'
    return grid


def test_plotting_classes_and_cv_helper_imports_still_work():
    assert PlotFVCOM is PlottingMixin
    assert ControlVolumePlotter is ControlVolumeMixin
    assert issubclass(FVCOM_grid, PlottingMixin)
    assert issubclass(FVCOM_grid, ControlVolumeMixin)

def test_plot_triangulations_are_cached_separately_and_reset(multi_cell_grid):
    grid = multi_cell_grid
    nodes = grid.mtri_nodes
    assert grid.mtri_nodes is nodes
    np.testing.assert_array_equal(nodes.triangles, grid.tri)

    # Explicit cell connectivity also avoids depending on land-mask heuristics
    # already exercised in the CellTriangulation regression tests.
    grid.ctri = mtri.Triangulation(grid.xc, grid.yc).triangles
    cells = grid.mtri_cells
    assert grid.mtri_cells is cells
    assert cells is not nodes
    assert len(cells.x) == grid.cell_number
    grid.ctri = grid.ctri.copy()
    assert 'mtri_cells' not in grid.__dict__
    assert grid.mtri_cells is not cells

    grid.x = grid.x + 1.0
    assert 'mtri_nodes' not in grid.__dict__
    assert 'mtri_cells' not in grid.__dict__
    assert grid.mtri_nodes is not nodes


def test_plot_grid_without_display(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    fig, ax = plt.subplots()
    assert grid.plot_grid(ax=ax, show=False) is None  # historical API
    assert ax.get_aspect() == 1.0  # equal
    assert len(ax.lines) >= 1


def test_plot_obc_uses_each_nodestring(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    fig, ax = plt.subplots()
    assert grid.plot_obc(ax=ax) is None
    assert len(ax.collections) == 2 * len(grid.nodestrings)


def test_nodal_contours_reuse_triangulation(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    fig, ax = plt.subplots()
    first = grid.plot_contour(np.array([0.0, 1.0, 3.0, 2.0]), ax=ax, show=False)
    triangulation = grid.mtri_nodes
    second = grid.plot_contour(np.array([1.0, 3.0, 6.0, 2.0]), ax=ax, show=False)
    assert first is not None and second is not None
    assert grid.mtri_nodes is triangulation
    assert triangulation.mask is None


def test_nodal_mask_changes_between_frames_without_mutating_mesh(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    original = grid.tri.copy()
    fig, ax = plt.subplots()
    # Node 1 belongs to the first triangle only.
    grid.plot_contour(np.array([0.0, np.nan, 3.0, 2.0]), ax=ax, show=False)
    cached = grid.mtri_nodes
    np.testing.assert_array_equal(cached.mask, [True, False])
    grid.plot_contour(np.array([0.0, 1.0, 3.0, 2.0]), ax=ax, show=False)
    assert grid.mtri_nodes is cached
    assert cached.mask is None
    np.testing.assert_array_equal(grid.tri, original)


def test_cell_contours_use_cell_centres_and_connectivity(multi_cell_grid):
    grid = multi_cell_grid
    # Assign explicit Delaunay connectivity so this test only checks the
    # plotting layer, not the land triangle filtering logic.
    grid.ctri = mtri.Triangulation(grid.xc, grid.yc).triangles
    fig, ax = plt.subplots()
    values = (grid.xc - grid.xc.min()) / 100.0 + (grid.yc - grid.yc.min()) / 100.0
    artist = grid.plot_contour(values, ax=ax, show=False)
    assert artist is not None
    assert grid.mtri_cells.x.shape == (grid.cell_number,)
    assert grid.mtri_nodes.x.shape == (grid.node_number,)


def test_contour_validation_and_all_dry_frames(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='one-dimensional'):
        grid.plot_contour(np.ones((3, 4)), ax=ax, show=False)
    with pytest.raises(ValueError, match='does not match'):
        grid.plot_contour(np.ones(3), ax=ax, show=False)
    with pytest.raises(ValueError, match='on_empty'):
        grid.plot_contour(np.ones(4), ax=ax, show=False, on_empty='maybe')
    assert grid.plot_contour(np.full(4, np.nan), ax=ax, show=False,
                             on_empty='skip') is None
    with pytest.raises(ValueError, match='No valid triangles'):
        grid.plot_contour(np.full(4, np.nan), ax=ax, show=False)
    # The frame after a fully dry frame must be able to draw again.
    assert grid.plot_contour(np.array([1., 3., 2., 4.]),
                             ax=ax, show=False) is not None


def _mock_control_volume_patches(grid):
    return [Polygon(np.array([(grid.x[i], grid.y[i]),
                              (grid.x[i] + 2, grid.y[i]),
                              (grid.x[i], grid.y[i] + 2)], dtype=float),
                    closed=True) for i in range(grid.node_number)]


def test_nodal_control_volume_plot_is_patch_collection(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    grid._cv = _mock_control_volume_patches(grid)
    fig, ax = plt.subplots()
    plt.sca(ax)  # plot_cvs uses pyplot's current axes by design
    field = np.array([10.0, 12.0, 14.0, 16.0])
    artist = grid.plot_cvs(field, cmap='viridis', edgecolor='k')
    assert isinstance(artist, PatchCollection)
    assert len(artist.get_paths()) == grid.node_number
    np.testing.assert_allclose(artist.get_array(), field)
    assert artist in ax.collections


def test_control_volume_cache_is_lazy_and_reset_on_grid_change(
        small_grid_file, monkeypatch):
    grid = FVCOM_grid(str(small_grid_file))
    calls = []
    patches = _mock_control_volume_patches(grid)

    def make_patches():
        calls.append(1)
        return patches

    monkeypatch.setattr(grid, '_get_cvs', make_patches)
    assert grid.cv is patches
    assert grid.cv is patches
    assert len(calls) == 1
    grid.tri = grid.tri.copy()
    assert '_cv' not in grid.__dict__
    assert grid.cv is patches
    assert len(calls) == 2


def test_cv_polygon_numba_helpers_still_function():
    x = np.array([0., 1., 0.])
    y = np.array([0., 0., 1.])
    xc = np.array([1. / 3., 1. / 3.])
    yc = np.array([1. / 3., 1. / 3.])
    xwall, ywall = _find_cv_walls(
        0, np.array([1, 2, 0], dtype=np.int64),
        np.array([0, 1, 1], dtype=np.int64), x, y, xc, yc)
    np.testing.assert_allclose(xwall, [.5, 1. / 3., 0., 0.])
    np.testing.assert_allclose(ywall, [0., 1. / 3., .5, 0.])


def test_control_volume_dispatch_passes_node_connectivity(monkeypatch):
    import fvtools.grid._components.control_volumes as cv_module

    seen = []

    def walls(node, adjacent_nodes, adjacent_cells, *_):
        seen.append((int(node), adjacent_nodes.copy(), adjacent_cells.copy()))
        return [float(node), float(node) + 1], [0.0, 1.0]

    monkeypatch.setattr(cv_module, '_find_cv_walls', walls)
    nbsn = np.array([[1, 2, -1], [-1, -1, -1]], dtype=int)
    nbve = np.array([[0, 1, -1], [-1, -1, -1]], dtype=int)
    xx, yy = ControlVolumePlotter._get_control_volumes(
        nbsn, nbve, np.array([0., 1., 2.]), np.array([0., 1., 2.]),
        np.array([.5, 1.5]), np.array([.5, 1.5]), np.array([0]))
    assert len(xx) == len(yy) == 1
    assert seen[0][0] == 0
    np.testing.assert_array_equal(seen[0][1], [1, 2])
    np.testing.assert_array_equal(seen[0][2], [0, 1])


def test_reference_mesh_plot_triangulation_smoke(reference_paths):
    _, npy_path = reference_paths
    grid = FVCOM_grid(str(npy_path))
    fig, ax = plt.subplots()
    grid.plot_grid(ax=ax, show=False)
    assert len(grid.mtri_nodes.triangles) == grid.cell_number
    assert len(ax.lines) >= 1
