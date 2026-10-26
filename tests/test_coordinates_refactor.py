"""Compatibility tests for coordinate mixins moved out of fvcom_grd.py."""
import numpy as np

from fvtools.grid.fvcom_grd import FVCOM_grid, Coordinates, InputCoordinates
from fvtools.grid._components.coordinates import (
    Coordinates as ImplementationCoordinates,
    InputCoordinates as ImplementationInputCoordinates,
)


def test_public_coordinate_class_imports_remain_identical():
    assert Coordinates is ImplementationCoordinates
    assert InputCoordinates is ImplementationInputCoordinates
    assert issubclass(FVCOM_grid, Coordinates)
    assert issubclass(FVCOM_grid, InputCoordinates)


def test_coordinate_derived_fields_preserve_shape_and_values(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    np.testing.assert_allclose(grid.xc, grid.x[grid.tri].mean(axis=1))
    np.testing.assert_allclose(grid.yc, grid.y[grid.tri].mean(axis=1))
    np.testing.assert_allclose(grid.hc, grid.h[grid.tri].mean(axis=1))
    np.testing.assert_allclose(grid.tri_area, [5000, 5000])
    np.testing.assert_allclose(grid.delta_sigma, 0.5)
    np.testing.assert_allclose(grid.siglay_c, grid.siglay[grid.tri].mean(axis=1))


def test_assigning_xy_invalidates_spatial_cache(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    _ = grid.node_tree
    assert 'node_tree' in grid.__dict__
    original_xc = grid.xc.copy()
    grid.x = grid.x + np.array([0.0, 10.0, 0.0, 0.0])
    assert 'node_tree' not in grid.__dict__
    assert not np.array_equal(grid.xc, original_xc)
