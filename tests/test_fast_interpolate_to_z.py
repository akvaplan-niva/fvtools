"""Regression tests for accelerated FVCOM sigma-to-z interpolation."""
import numpy as np
import pytest

from fvtools.grid.fvcom._components.interpolation import (
    InterpolateToZ, _interpolate_to_z_numba,
)
from fvtools.grid.fvcom.netcdf_io import NetCDFFieldProcessing


class Grid(InterpolateToZ, NetCDFFieldProcessing):
    pass


def reference(f, z, depths):
    """Independent vectorized oracle preserving the original selection rules."""
    f = np.ma.asarray(f, dtype=float).filled(np.nan)
    depths = np.asarray(depths, dtype=float)
    valid = np.isfinite(f) & np.isfinite(depths)
    above = np.where(valid & (depths <= z), depths, -np.inf)
    below = np.where(valid & (depths >= z), depths, np.inf)
    upper_index = np.argmax(above, axis=0)
    lower_index = np.argmin(below, axis=0)
    loc = np.arange(f.shape[1])
    upper_depth, lower_depth = above[upper_index, loc], below[lower_index, loc]
    upper_value = f[upper_index, loc]
    lower_value = f[lower_index, loc]
    out = np.full(f.shape[1], np.nan, dtype=float)
    exact = np.isfinite(upper_depth) & np.isfinite(lower_depth) & (upper_depth == lower_depth)
    out[exact] = upper_value[exact]
    interpolate = (np.isfinite(upper_depth) & np.isfinite(lower_depth)
                   & (lower_depth > upper_depth))
    out[interpolate] = (
        upper_value[interpolate] * (lower_depth[interpolate] - z)
        + lower_value[interpolate] * (z - upper_depth[interpolate])
    ) / (lower_depth[interpolate] - upper_depth[interpolate])
    shallower = (~np.isfinite(upper_depth)) & np.isfinite(lower_depth)
    out[shallower] = lower_value[shallower]
    return out


@pytest.mark.parametrize('field_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('depth_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('z', [0.0, 2.5, 11.0, 200.0])
def test_matches_old_algorithm(field_dtype, depth_dtype, order, z):
    rng = np.random.default_rng(42)
    nlev, npoints = 18, 133
    depths = np.linspace(.1, 30, nlev)[:, None] * rng.uniform(.3, 1.8, (1, npoints))
    field = 12. - .35 * depths + rng.normal(0, .02, depths.shape)
    # Exercise invalid values, duplicate depths, and out-of-order levels.
    depths[2, :6] = depths[1, :6]
    depths[[3, 8], :9] = depths[[8, 3], :9].copy()
    field[4, 3:14] = np.nan
    depths[12, 5:15] = np.nan
    field[6, 17] = np.inf
    f = np.array(field, dtype=field_dtype, order=order)
    d = np.array(depths, dtype=depth_dtype, order=order)
    got = Grid().interpolate_to_z(f, z, depths=d)
    expected = reference(f, z, d)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, rtol=2e-14, atol=2e-14, equal_nan=True)


def test_masked_and_implicit_nodal_depths():
    grid = Grid()
    depths = np.array([[.5, .5, .5], [2., 2., 2.], [4., 4., 4.]])
    grid.siglayz = -depths.T
    field = np.ma.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                        mask=[[0, 1, 0], [0, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(grid.interpolate_to_z(field, 1.25),
                               reference(field, 1.25, depths), equal_nan=True)


@pytest.mark.parametrize('z', [-1., np.nan, np.inf])
def test_invalid_z_rejected(z):
    with pytest.raises(ValueError):
        Grid().interpolate_to_z(np.ones((3, 4)), z, depths=np.ones((3, 4)))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        Grid().interpolate_to_z(np.ones((3, 4)), 1., depths=np.ones((4, 4)))


def test_numba_compiles_in_nopython_mode():
    f = np.arange(9., dtype=np.float32).reshape(3, 3)
    d = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    np.testing.assert_allclose(_interpolate_to_z_numba(f, 3., d),
                               reference(f, 3., d), equal_nan=True)
    assert _interpolate_to_z_numba.nopython_signatures
