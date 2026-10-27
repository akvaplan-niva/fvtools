"""Numerical regression tests for fvtools.grid.tools.make_interpolation_matrices.

Explicitly test the actual function, not a monkeypatched stand-in.  For a
nodal/cell tracer, the legacy postprocessing scripts compute:

    np.sum(weights * values, axis=1)

All tests are headless and write no output files.
"""

import numpy as np
import pytest

from fvtools.grid.tools import make_interpolation_matrices
from fvtools.grid.fvcom_grd import FVCOM_grid


@pytest.fixture
def varying_sigma_levels():
    """Three water columns, with distinct but monotonic negative z-levels."""
    return np.array([
        [-1.0, -3.0, -5.0],
        [-2.0, -4.0, -8.0],
        [-0.5, -3.5, -6.5],
    ])


@pytest.mark.parametrize('depth', [-2.5, -3.0, -4.0])
def test_linear_field_is_interpolated_to_the_requested_depth(
    varying_sigma_levels, depth,
):
    zlevels = varying_sigma_levels
    original = zlevels.copy()
    weights = make_interpolation_matrices(zlevels, depth)

    assert isinstance(weights, np.ndarray), 'Function must return the weight matrix'
    assert weights.shape == zlevels.shape
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)

    # A linear field must be reproduced exactly by linear sigma interpolation.
    field = 3 * zlevels + np.arange(len(zlevels))[:, None]
    output = np.sum(weights * field, axis=1)
    np.testing.assert_allclose(output, 3 * depth + np.arange(len(zlevels)), atol=1e-13)
    np.testing.assert_array_equal(zlevels, original)


def test_exact_level_selects_one_sigma_level():
    levels = np.tile([-1.0, -3.0, -5.0], (3, 1))
    weights = make_interpolation_matrices(levels, -3.0)
    np.testing.assert_array_equal(weights, np.tile([0.0, 1.0, 0.0], (3, 1)))


def test_above_shallowest_sigma_level_uses_first_value(varying_sigma_levels):
    levels = varying_sigma_levels
    weights = make_interpolation_matrices(levels, 0.0)
    np.testing.assert_array_equal(weights, np.tile([1., 0., 0.], (3, 1)))
    field = np.arange(9., dtype=float).reshape(3, 3)
    np.testing.assert_array_equal(np.sum(weights * field, axis=1), field[:, 0])


def test_below_bottom_returns_nan_only_for_affected_columns(varying_sigma_levels):
    levels = varying_sigma_levels
    weights = make_interpolation_matrices(levels, -6.0)
    assert np.isnan(weights[0]).all()
    assert np.isfinite(weights[1:]).all()
    np.testing.assert_allclose(weights[1:].sum(axis=1), 1.0)
    np.testing.assert_allclose(np.sum(weights[1:] * levels[1:], axis=1), -6.0)


def test_ts_mixin_generates_weights_used_by_postpro(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    grid.make_interpolation_matrices_TS([-3.0])
    weights = grid.interpolation_matrix_TS_3_m
    levels = grid.siglayz
    assert weights.shape == levels.shape == (grid.node_number, 2)
    # Emulates postpro/fvcom_data_extract_TS.py (node, sigma) multiplication.
    temperature = 2 * levels + np.arange(grid.node_number)[:, None]
    np.testing.assert_allclose(
        np.sum(weights * temperature, axis=1),
        -6.0 + np.arange(grid.node_number),
        atol=1e-12,
    )


def test_uv_mixin_generates_cell_based_weights_used_by_postpro(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    grid.make_interpolation_matrices_uv([-3.0])
    weights = grid.interpolation_matrix_uv_3_m
    levels = grid.siglayz_uv
    assert weights.shape == levels.shape == (grid.cell_number, 2)
    # u and v in FVCOM are cell-centred, never nodal.
    u = 2 * levels + np.arange(grid.cell_number)[:, None]
    v = -3 * levels
    np.testing.assert_allclose(
        np.sum(weights * u, axis=1), -6.0 + np.arange(grid.cell_number), atol=1e-12,
    )
    np.testing.assert_allclose(np.sum(weights * v, axis=1), 9.0, atol=1e-12)


def test_original_function_on_reference_sigma_levels(reference_paths):
    """Check real-world sigma profiles when the optional M.npy fixture exists."""
    _, path = reference_paths
    grid = FVCOM_grid(str(path))
    levels = grid.siglayz[:200]
    # -5 m may lie below the last layer in shallow water.
    weights = make_interpolation_matrices(levels, -5.0)
    assert weights.shape == levels.shape
    finite = np.isfinite(weights).all(axis=1)
    assert finite.any(), 'At least one reference profile should reach -5 m'
    np.testing.assert_allclose(weights[finite].sum(axis=1), 1.0, atol=1e-12)
    expected = np.where(levels[finite, 0] < -5.0, levels[finite, 0], -5.0)
    np.testing.assert_allclose(
        np.sum(weights[finite] * levels[finite], axis=1), expected, atol=1e-10,
    )
