"""Regression tests for the Numba-compatible CV polygon duplicate check.

No FVCOM input files or Jupyter kernel are required. These tests also compile
both public Numba helper functions in nopython mode.
"""
import numpy as np
import pytest

from fvtools.grid.fvcom._components.control_volumes import (
    _connect_cv_walls, _find_cv_walls,
)


def original_connect_python(xmid, ymid, xcell, ycell, x, y,
                            nbsnhere, elemshere, node):
    """Independent legacy implementation; deliberately not JIT compiled."""
    xcv = []
    ycv = []
    for i in range(len(xmid)):
        xcv.append(xmid[i])
        ycv.append(ymid[i])
        boundary = False
        if i < len(xmid) - 1:
            xcvmid = (xmid[i] + xmid[i + 1]) / 2
            ycvmid = (ymid[i] + ymid[i + 1]) / 2
            dst = np.sqrt((xcell - xcvmid)**2 + (ycell - ycvmid)**2)
            ind = np.argwhere(dst == dst.min())[0]
            if nbsnhere[i + 1] == node:
                boundary = True
                xcv.append(x[nbsnhere[i + 1]])
                ycv.append(y[nbsnhere[i + 1]])
            elif nbsnhere[i] == node:
                boundary = True
        if not boundary:
            xcv.append(xcell[ind][0])
            ycv.append(ycell[ind][0])
        uxcv = np.array(xcv)
        if boundary and len(np.unique(uxcv)) < len(xcv):
            break
    return xcv, ycv


def test_original_cv_polygon_coordinates_exactly_preserved():
    x = np.array([0., 1., 0.])
    y = np.array([0., 0., 1.])
    xc = np.array([1. / 3., 1. / 3.])
    yc = np.array([1. / 3., 1. / 3.])
    xwall, ywall = _find_cv_walls(
        0, np.array([1, 2, 0], dtype=np.int64),
        np.array([0, 1, 1], dtype=np.int64), x, y, xc, yc)
    np.testing.assert_array_equal(xwall, [.5, 1. / 3., 0., 0.])
    np.testing.assert_array_equal(ywall, [0., 1. / 3., .5, 0.])
    assert _find_cv_walls.nopython_signatures
    assert _connect_cv_walls.nopython_signatures


@pytest.mark.parametrize('seed', [0, 1, 19, 731])
@pytest.mark.parametrize('n', [3, 4, 7, 12])
def test_numba_polygons_equal_independent_legacy_reference(seed, n):
    rng = np.random.default_rng(seed)
    # Repeated coordinate values deliberately exercise boundary early exit.
    for _ in range(12):
        x = rng.choice([-2., -1., 0., 0.5, 2.], size=n)
        y = rng.choice([-2., -1., 0., 1., 2.], size=n)
        xc = rng.choice([-1., 0., .25, 1., 3.], size=n + 1)
        yc = rng.choice([-2., -.5, 0., 1., 3.], size=n + 1)
        nbsn = np.arange(n, dtype=np.int64)
        nbsn[-1] = 0
        elements = rng.integers(0, n + 1, size=n, dtype=np.int64)
        xmid = (x[0] + x[nbsn]) / 2
        ymid = (y[0] + y[nbsn]) / 2
        xcell, ycell = xc[elements], yc[elements]
        expected = original_connect_python(
            xmid, ymid, xcell, ycell, x, y, nbsn, elements, 0)
        actual = _connect_cv_walls(
            xmid, ymid, xcell, ycell, x, y, nbsn, elements, 0)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
