"""A globally valid local simplex must be preferred over remote hull support."""

from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator
from fvtools.setup._buildcase import tiled_bathymetry as mod


@pytest.mark.parametrize('shift', [(0., 0.), (530870.847, 6252885.830)])
def test_exact_local_simplex_needs_no_distant_hull(tmp_path, monkeypatch, shift):
    # The target lies within a genuine 60-m Delaunay triangle; the remaining
    # 55 points are outside its circumcircle. This resembles the diagnosed
    # query with 58 points within 300 m and no global circle intruders.
    rng = np.random.default_rng(20261101)
    angles = rng.uniform(-np.pi, np.pi, 55)
    radii = rng.uniform(100.0, 290.0, 55)
    surrounding = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)))
    vertices = np.array([[60., 0.], [-30., 51.961524227],
                         [-30., -51.961524227]])
    # Far hull points create the geometry that made the old fallback costly.
    hull = np.array([[-100_000.,-100_000.], [100_000.,-100_000.],
                     [100_000.,100_000.], [-100_000.,100_000.]])
    origin = np.asarray(shift)
    points = np.vstack((vertices, surrounding, hull)) + origin
    values = 0.003 * (points[:, 0] - origin[0]) + 0.1 * (
        points[:, 1] - origin[1]) + np.cos((points[:, 0] - origin[0])/100.)
    src = np.column_stack((points, values))
    folder = tmp_path/'bins'
    folder.mkdir()
    bounds = tuple((origin + [-105000.,-105000.]).tolist() +
                   (origin + [105000.,105000.]).tolist())
    idx = mod._DiskBins(src, bounds, folder,
                        mod.TilingOptions(source_chunk_rows=100,
                                          bins_per_axis=8))
    opts = mod.TilingOptions(max_triangulation_points=200)

    def unexpected_hull(*args, **kwargs):
        raise AssertionError('Globally certified local simplex must not need remote hull')
    monkeypatch.setattr(mod, '_hull_support_for_query', unexpected_hull)

    got, steps, triangle = mod._interpolate_dense_single(
        origin, idx, opts, return_triangle=True)
    reference = float(LinearNDInterpolator(points-origin, values)(
        np.array([[0.,0.]]))[0])
    assert steps <= 4
    assert triangle.shape == (3, 3)
    np.testing.assert_allclose(got, reference, atol=1e-8, rtol=1e-10)


def test_local_certification_never_accepts_circle_intruders(tmp_path):
    # If a purported local triangle is not globally valid, preserve exactness.
    rng = np.random.default_rng(111)
    pts = rng.uniform(-100.,100.,(200,2))
    src = np.column_stack((pts, np.sin(pts[:,0]/15.) + pts[:,1]/19.))
    folder = tmp_path/'bins'
    folder.mkdir()
    idx = mod._DiskBins(src, (-110.,-110.,110.,110.), folder,
                        mod.TilingOptions(source_chunk_rows=100,bins_per_axis=8))
    q = np.array([3.13,9.27])
    value, steps = mod._interpolate_dense_single(
        q, idx, mod.TilingOptions(max_triangulation_points=160))
    expected = float(LinearNDInterpolator(src[:,:2], src[:,2])(q[None,:])[0])
    np.testing.assert_allclose(value, expected, atol=1e-10, rtol=1e-10)
