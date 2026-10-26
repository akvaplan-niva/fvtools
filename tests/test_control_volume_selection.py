"""Regression tests for plot_cvs node-index selection and value alignment."""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import pytest

from fvtools.grid.fvcom_grd import FVCOM_grid


@pytest.fixture
def grid(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    # Isolate the plot_cvs selection logic from expensive TGE calculations.
    # Deliberately use visibly different polygons to verify their order.
    grid._cv = [
        Polygon([(i * 10.0, 0.0), (i * 10.0 + 2.0, 0.0),
                 (i * 10.0, 2.0)], closed=True)
        for i in range(grid.node_number)
    ]
    return grid


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def assert_selected(artist, field, expected):
    assert isinstance(artist, PatchCollection)
    expected = np.asarray(expected, dtype=int)
    np.testing.assert_array_equal(artist.get_array(), np.asarray(field)[expected])
    assert len(artist.get_paths()) == len(expected)
    for path, index in zip(artist.get_paths(), expected):
        np.testing.assert_allclose(path.vertices[0], [index * 10.0, 0.0])


def test_cv_explicit_numpy_indices_selects_correct_patches(grid):
    field = np.array([10.0, 20.0, 30.0, 40.0])
    artist = grid.plot_cvs(field, inds=np.array([3, 1]))
    assert_selected(artist, field, [3, 1])


def test_cv_full_length_permutation_does_not_silently_plot_wrong_patches(grid):
    field = np.array([10.0, 20.0, 30.0, 40.0])
    artist = grid.plot_cvs(field, inds=[2, 0, 3, 1])
    assert_selected(artist, field, [2, 0, 3, 1])


def test_cv_thresholds_intersect_explicit_node_subset(grid):
    field = np.array([10.0, 20.0, 30.0, 40.0])
    artist = grid.plot_cvs(field, inds=np.array([0, 2, 3]), cmin=15, cmax=35)
    assert_selected(artist, field, [2])


def test_cv_thresholds_without_inds_select_expected_nodes(grid):
    field = [10.0, 20.0, 30.0, 40.0]
    artist = grid.plot_cvs(field, cmin=20, cmax=30)
    assert_selected(artist, field, [1, 2])


def test_cv_boolean_mask_indices_and_empty_selection(grid):
    field = np.array([10.0, 20.0, 30.0, 40.0])
    artist = grid.plot_cvs(field, inds=[True, False, True, False])
    assert_selected(artist, field, [0, 2])
    artist = grid.plot_cvs(field, inds=[0, 1], cmin=100)
    assert_selected(artist, field, [])


@pytest.mark.parametrize('inds,error', [
    ([5], IndexError),
    ([-1], IndexError),
    ([0.5, 1.5], TypeError),
    ([[0, 1]], ValueError),
    ([True, False], ValueError),
])
def test_cv_bad_indices_fail_explicitly(grid, inds, error):
    with pytest.raises(error):
        grid.plot_cvs(np.arange(4.0), inds=inds)


def test_cv_requires_one_nodal_value_per_patch(grid):
    with pytest.raises(ValueError, match='one value per node'):
        grid.plot_cvs(np.ones(2))
