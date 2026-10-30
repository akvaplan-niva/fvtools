"""Functional, numeric, edge-case regression checks for phase 5b.

Import the installed fvtools module under its permanent package name.
Numba caching depends on the importable module name.
"""
from __future__ import annotations

import numpy as np
import pytest

@pytest.fixture(scope='module')
def vi():
    # Import under its permanent package name. Numba caches the module name.
    from fvtools.prepro.nesting import vertical_interpolation
    return vertical_interpolation


def reference_weights(z1, z2):
    """Independent NumPy reference reproducing old nearest argmin + bracketing."""
    if z1.ndim != 2 or z2.ndim != 2 or z1.shape[1] != z2.shape[1] or z1.shape[0] == 0:
        raise ValueError('Vertical interpolation requires two 2-D arrays with matching columns and at least one ROMS level')
    ix1 = np.zeros(z2.shape, dtype=np.int64)
    ix2 = np.zeros(z2.shape, dtype=np.int64)
    w1 = np.zeros(z2.shape)
    w2 = np.zeros(z2.shape)
    for n in range(z1.shape[1]):
        parent = z1[:, n]
        child = z2[:, n]
        if not (np.all(np.isfinite(parent)) and np.all(np.isfinite(child))):
            raise ValueError('Vertical interpolation depths must be finite')
        if np.any(np.diff(parent) < 0):
            raise ValueError('ROMS vertical levels must be sorted shallow-to-deep')
        nearest = np.argmin(np.abs(child[:, None] - parent[None, :]), axis=1)
        for k, m in enumerate(nearest):
            m = int(m)
            ix1[k, n] = m
            depth = child[k]
            if depth <= parent[0] or depth >= parent[-1] or depth == parent[m]:
                w1[k, n] = 1.0
            elif parent[m] > depth:
                other = m - 1
                while other >= 0 and parent[other] == parent[m]:
                    other -= 1
                if other < 0:
                    raise ValueError('Cannot bracket FVCOM depth with ROMS sigma levels')
                ix2[k, n] = other
                dz = parent[m] - parent[other]
                w2[k, n] = (parent[m] - depth) / dz
                w1[k, n] = (depth - parent[other]) / dz
            else:
                other = m + 1
                while other < len(parent) and parent[other] == parent[m]:
                    other += 1
                if other == len(parent):
                    raise ValueError('Cannot bracket FVCOM depth with ROMS sigma levels')
                ix2[k, n] = other
                dz = parent[other] - parent[m]
                w2[k, n] = (depth - parent[m]) / dz
                w1[k, n] = (parent[other] - depth) / dz
    return ix1, ix2, w1, w2


def assert_same(vi, a, b):
    ref = reference_weights(a, b)
    got = vi.calc_interp_matrices(a, b)
    for x, y in zip(ref, got):
        np.testing.assert_array_equal(x, y)
    assert got[0].dtype == got[1].dtype == np.int64
    assert got[2].dtype == got[3].dtype == np.float64


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('npoints', [1, 9, 200])
def test_random_vertical_grid(vi, dtype, order, npoints):
    rng = np.random.default_rng(112)
    parent = np.cumsum(rng.uniform(0.1, 5, (40, npoints)), axis=0).astype(dtype)
    child = rng.uniform(-2, 210, (21, npoints)).astype(dtype)
    assert_same(vi, np.array(parent, order=order), np.array(child, order=order))


@pytest.mark.parametrize('parent,child', [
    ([0., 5., 10.], [-1, 0, 2.5, 5, 7.5, 10, 11]),
    ([0., 0., 5., 10., 10.], [-1, 0, 1, 2.5, 5, 7.5, 10, 11]),
    ([5., 5., 5.], [0, 5, 10]),
    ([7.], [-1, 7, 12]),
    ([0, 3, 3, 5, 9], [1.5, 3, 4, 7, 8.5]),
])
def test_boundaries_duplicates_midpoint_ties(vi, parent, child):
    assert_same(vi, np.array(parent, dtype=float)[:, None],
                np.array(child, dtype=float)[:, None])


@pytest.mark.parametrize('invalid_parent,invalid_child', [
    ([[0], [1], [float('nan')]], [[0], [1]]),
    ([[0], [float('inf')]], [[0], [1]]),
    ([[0], [2], [1]], [[0], [1]]),
    ([[0], [1]], [[0], [float('inf')]]),
])
def test_invalid_depth_profiles(vi, invalid_parent, invalid_child):
    with pytest.raises(ValueError):
        vi.calc_interp_matrices(np.array(invalid_parent, dtype=float),
                                np.array(invalid_child, dtype=float))


def test_empty_columns_and_levels(vi):
    assert_same(vi, np.empty((20, 0)), np.empty((10, 0)))
    assert_same(vi, np.array([[1., 2.], [2., 3.]]), np.empty((0, 2)))
    with pytest.raises(ValueError, match='2-D arrays'):
        vi.calc_interp_matrices(np.empty((0, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError, match='2-D arrays'):
        vi.calc_interp_matrices(np.ones((3, 4)), np.ones((2, 3)))


def test_no_changes_to_public_helpers(vi):
    # Keep any callers of the old helper routines working.
    assert callable(vi.numba_absdiff)
    assert callable(vi.find_inds_and_weights)
    assert callable(vi.prepare_z_dimensions)
    assert callable(vi.add_vertical_interpolation2N4)


def test_medium_grid_matches_original(vi):
    rng = np.random.default_rng(209)
    n = 2_000
    parent = np.cumsum(rng.uniform(.1, 1, size=(30, n)), axis=0)
    child = rng.uniform(-2, 35, size=(21, n))
    assert_same(vi, parent, child)
