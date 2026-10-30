"""Regressions for BuildRivers after removing FVCOM grid_metrics.py."""

from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.prepro._buildrivers.forcing import FVCOM_rivers


def _forcing_with_boundary(nodestrings):
    # A square around one interior node; an interior OBC must not leak into
    # the set of true mesh-perimeter nodes.
    m = SimpleNamespace(
        boundary_nodes=np.array([[2, 3], [1, 2], [3, 0], [0, 1]], dtype=int),
        nodestrings=[np.asarray(s, dtype=int) for s in nodestrings],
        x=np.array([10., 20., 30., 40., 50., 60.]),
        y=np.array([11., 21., 31., 41., 51., 61.]),
    )
    obj = FVCOM_rivers.__new__(FVCOM_rivers)
    obj.M = m
    obj.info = {'iloc': 'node'}
    return obj


@pytest.mark.parametrize('strings,land,obc', [
    ([[1, 2]], [0, 3], [1, 2]),
    ([[0, 1, 2, 3, 0]], [], [0, 1, 2, 3]),
    ([[0, 1], [2, 3]], [], [0, 1, 2, 3]),
    ([], [0, 1, 2, 3], []),
    ([[4, 5, 4]], [0, 1, 2, 3], []),
    ([[0, 1], [4, 5, 4]], [2, 3], [0, 1]),
])
def test_river_node_locations_match_old_perimeter_classification(strings, land, obc):
    obj = _forcing_with_boundary(strings)
    np.testing.assert_array_equal(obj._boundary_node_groups[0], land)
    np.testing.assert_array_equal(obj._boundary_node_groups[1], obc)
    np.testing.assert_allclose(obj.xy_land, np.column_stack((obj.M.x[land], obj.M.y[land])) if land else np.empty((0, 2)))
    np.testing.assert_allclose(obj.xy_obc, np.column_stack((obj.M.x[obc], obj.M.y[obc])) if obc else np.empty((0, 2)))


def test_boundary_node_groups_are_cached():
    obj = _forcing_with_boundary([[1, 2]])
    original = obj._boundary_node_groups
    assert obj._boundary_node_groups is original


def test_rivers_module_does_not_import_removed_grid_metrics():
    import ast
    import inspect
    from fvtools.prepro._buildrivers import forcing

    tree = ast.parse(inspect.getsource(forcing))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                alias.name != "fvtools.grid.fvcom.grid_metrics"
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            assert node.module != "fvtools.grid.fvcom.grid_metrics"

            if node.module == "fvtools.grid.fvcom":
                assert all(
                    alias.name != "grid_metrics"
                    for alias in node.names
                )