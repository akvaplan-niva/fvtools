"""Regression tests for minimum depth in BuildCase tiled and text interpolation.

These tests only use synthetic data and do not read model bathymetry files.
"""
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from fvtools.prepro._buildcase.depth import DepthHandler


def test_tiled_npy_enforces_min_depth_and_preserves_input(tmp_path):
    source = np.array([[0., 0., 1.], [10., 0., 8.],
                       [0., 10., 3.], [10., 10., 12.], [5., 5., 2.]])
    filename = tmp_path / 'bathy.npy'
    np.save(filename, source)
    xy = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.], [5., 5.]])
    obj = SimpleNamespace(depth_file=str(filename), x=xy[:, 0],
                          xy_grid_at_depth_points=xy, min_depth=4.)
    DepthHandler.make_depth(obj)
    expected = LinearNDInterpolator(source[:, :2], source[:, 2])(xy)
    np.maximum(expected, 4., out=expected)
    np.testing.assert_allclose(obj.h_raw, expected)
    np.testing.assert_array_equal(np.load(filename), source)


def test_tiled_npy_outside_convex_hull_becomes_min_depth(tmp_path):
    source = np.array([[0., 0., 1.], [10., 0., 10.], [0., 10., 7.]])
    filename = tmp_path / 'triangle.npy'
    np.save(filename, source)
    xy = np.array([[2., 2.], [9., 9.]])
    obj = SimpleNamespace(depth_file=str(filename), x=xy[:, 0],
                          xy_grid_at_depth_points=xy, min_depth=4.)
    DepthHandler.make_depth(obj)
    np.testing.assert_allclose(obj.h_raw, [4., 4.])


def test_text_depth_enforces_minimum_while_flat_depth_stays_unchanged():
    class Text:
        depth_file = 'bathy.txt'
        x = np.zeros(3)
        min_depth = 3.
        def load_textbath(self, filename):
            assert filename == self.depth_file
            return 'loaded'
        def crop(self, data):
            assert data == 'loaded'
            return 'cropped'
        def interpolate_to_mesh(self, data):
            assert data == 'cropped'
            return np.array([2., 4., 6.])

    text = Text()
    DepthHandler.make_depth(text)
    np.testing.assert_array_equal(text.h_raw, [3., 4., 6.])

    # Numeric constant bathymetry took a separate path in the original code.
    flat = SimpleNamespace(depth_file=2., x=np.zeros(3), min_depth=4.)
    DepthHandler.make_depth(flat)
    np.testing.assert_array_equal(flat.h_raw, [2., 2., 2.])
