"""Regression tests for bounded-memory BuildCase bathymetry interpolation.

No external topography is needed. SciPy's original global interpolator is the
numerical reference for all sufficiently small synthetic point clouds.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator

from fvtools.prepro._buildcase import tiled_bathymetry as tiled
from fvtools.prepro._buildcase.depth import DepthHandler


def _reference(data, queries):
    lo = queries.min(axis=0) - 5000
    hi = queries.max(axis=0) + 5000
    selected = np.logical_and.reduce((
        data[:, 0] >= lo[0], data[:, 0] <= hi[0],
        data[:, 1] >= lo[1], data[:, 1] <= hi[1],
    ))
    return LinearNDInterpolator(data[selected, :2], data[selected, 2])(queries)


def _compare(tmp_path, data, queries):
    source = tmp_path / 'topography.npy'
    np.save(source, data)
    opts = tiled.TilingOptions(
        source_chunk_rows=233, bins_per_axis=13,
        nodes_per_tile=32, max_triangulation_points=12_000,
    )
    got = tiled.interpolate_unstructured_npy(source, queries, options=opts)
    expected = _reference(data, queries)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    return got


def test_random_unstructured_bathymetry_matches_global_delaunay(tmp_path):
    rng = np.random.default_rng(42)
    xy = rng.uniform(0, 100, (4000, 2))
    depths = 2 * xy[:, 0] + 0.01 * xy[:, 1] ** 2 + np.sin(xy[:, 0]/4)
    targets = rng.uniform(5, 95, (320, 2))
    _compare(tmp_path, np.c_[xy, depths], targets)


def test_tiled_edges_including_regular_cocircular_source(tmp_path):
    source = np.array([(x, y, x + 2*y) for x in range(-1, 32)
                       for y in range(-1, 32)], dtype=float)
    target = np.c_[np.array([0, 1, 1e-8, 5.5, 29.5, 30, -0.1, 30.1]),
                   np.array([0, 4, 1e-8, 20.2, 29.5, 29, 16, 22])]
    _compare(tmp_path, source, target)


def test_sparse_and_highly_concentrated_source_points(tmp_path):
    rng = np.random.default_rng(62)
    xy = np.r_[rng.normal([60, 50], 2, size=(9000, 2)),
               rng.uniform(0, 100, size=(250, 2))]
    depths = np.sin(xy[:, 0]/20) + xy[:, 1]/40
    target = np.r_[rng.normal([60, 50], 2, size=(120, 2)),
                   rng.uniform(10, 90, size=(70, 2))]
    _compare(tmp_path, np.c_[xy, depths], target)


def test_float32_topography(tmp_path):
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 20, (3000, 2)).astype(np.float32)
    values = np.sin(xy[:, 0]) * np.cos(xy[:, 1])
    target = rng.uniform(2, 18, (87, 2))
    _compare(tmp_path, np.c_[xy, values].astype(np.float32), target)


def test_nan_outside_global_convex_hull(tmp_path):
    source = np.array([[0, 0, 7], [10, 0, 8], [5, 10, 9]], float)
    target = np.array([[5., 3.], [9., 9.], [10., 10.]])
    got = _compare(tmp_path, source, target)
    assert np.isfinite(got[0])
    assert np.isnan(got[1]) and np.isnan(got[2])


def test_hard_cap_on_triangulation_size(tmp_path, monkeypatch):
    rng = np.random.default_rng(78)
    xy = rng.uniform(0, 400, (35_000, 2))
    z = (np.sin(xy[:, 0]/40) + np.cos(xy[:, 1]/80)).astype('f4')
    source = np.c_[xy.astype('f4'), z]
    targets = rng.uniform(25, 375, (200, 2))
    path = tmp_path / 'large.npy'
    np.save(path, source)
    options = tiled.TilingOptions(source_chunk_rows=800, bins_per_axis=29,
                                  nodes_per_tile=25, max_triangulation_points=2400)
    original = tiled.Delaunay
    sampled = []

    def guarded_delaunay(points):
        sampled.append(len(points))
        assert len(points) <= options.max_triangulation_points
        return original(points)

    monkeypatch.setattr(tiled, 'Delaunay', guarded_delaunay)
    result = tiled.interpolate_unstructured_npy(path, targets, options=options)
    expected = _reference(source, targets)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    assert len(sampled) > 1 and max(sampled) <= 2400


def test_invalid_source_shape_and_invalid_options(tmp_path):
    path = tmp_path / 'bad.npy'
    np.save(path, np.ones((8, 2)))
    with pytest.raises(ValueError, match='columns'):
        tiled.interpolate_unstructured_npy(path, np.array([[0., 0.]]))
    np.save(path, np.zeros((30, 3)))
    with pytest.raises(ValueError, match='must be > 0'):
        tiled.interpolate_unstructured_npy(path, np.array([[0., 0.]]),
                                           options=tiled.TilingOptions(bins_per_axis=0))


def test_outside_source_extent_still_rejected(tmp_path):
    path = tmp_path / 'data.npy'
    np.save(path, np.array([[0., 0., 2.], [1., 0., 3.], [0., 1., 4.]]))
    with pytest.raises(ValueError, match='cover'):
        tiled.interpolate_unstructured_npy(path, np.array([[4., 4.], [5., 5.]]))


def test_depthhandler_npy_path_preserves_min_depth_and_source_file(tmp_path):
    xy = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.], [5., 5.]])
    source = np.array([[0., 0., 1.], [10., 0., 8.],
                       [0., 10., 3.], [10., 10., 12.], [5., 5., 2.]])
    filename = tmp_path/'bathy.npy'
    np.save(filename, source)
    obj = SimpleNamespace(depth_file=str(filename), x=xy[:, 0],
                          xy_grid_at_depth_points=xy, min_depth=4.)
    DepthHandler.make_depth(obj)
    expected = _reference(source, xy)
    expected[np.isnan(expected) | (expected < 4.)] = 4.
    np.testing.assert_allclose(obj.h_raw, expected)
    np.testing.assert_array_equal(np.load(filename), source)


def test_flat_and_text_bathymetry_workflow_unchanged():
    flat = SimpleNamespace(depth_file=17., x=np.zeros(4), min_depth=3.)
    DepthHandler.make_depth(flat)
    np.testing.assert_array_equal(flat.h_raw, [17., 17., 17., 17.])

    class Text:
        depth_file = 'topography.txt'
        x = np.zeros(3)
        min_depth = 3.
        def load_textbath(self, filename):
            assert filename == self.depth_file
            return 'raw'
        def crop(self, data):
            assert data == 'raw'
            return 'cropped'
        def interpolate_to_mesh(self, data):
            assert data == 'cropped'
            return np.array([2., 4., 6.])
    obj = Text()
    DepthHandler.make_depth(obj)
    np.testing.assert_array_equal(obj.h_raw, [3., 4., 6.])


def test_extra_unused_source_columns_are_ignored(tmp_path):
    source = np.array([[0,0,10,99],[10,0,20,99],[0,10,30,99],[10,10,40,99]],dtype=float)
    target = np.array([[2.,3.],[3.,7.],[4.,5.]])
    path=tmp_path/'four_columns.npy'
    np.save(path,source)
    got=tiled.interpolate_unstructured_npy(path,target)
    exp=LinearNDInterpolator(source[:,:2],source[:,2])(target)
    np.testing.assert_allclose(got,exp,rtol=1e-10,atol=1e-10)
