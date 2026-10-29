"""Regressions for warning handling, OBC caches and FVCOM_grid representation.

All tests use the standard tiny test grid; no network access or output files.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fvtools.grid.fvcom_grd import FVCOM_grid, NEST_grid


def test_grid_exports_and_inheritance_are_preserved():
    assert issubclass(FVCOM_grid, object)
    assert FVCOM_grid.__name__ == 'FVCOM_grid'
    assert NEST_grid.__name__ == 'NEST_grid'
    assert callable(FVCOM_grid.smooth)
    assert callable(FVCOM_grid._smooth)


def test_import_does_not_disable_runtime_warnings():
    """Test in a fresh process so import-time warning filters are observable."""
    check = (
        'import warnings\n'
        'warnings.simplefilter("error", RuntimeWarning)\n'
        'import fvtools.grid.fvcom_grd\n'
        'warnings.warn("fvtools_cleanup_warning_marker", RuntimeWarning)\n'
        'raise AssertionError("RuntimeWarning was globally suppressed")\n'
    )
    proc = subprocess.run(
        [sys.executable, '-c', check],
        cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, check=False,
    )
    assert proc.returncode != 0
    assert 'RuntimeWarning: fvtools_cleanup_warning_marker' in proc.stderr, (
        f'stdout={proc.stdout}\nstderr={proc.stderr}'
    )


def test_direct_mesh_has_default_casename_and_printable_repr(small_grid_file):
    loaded = FVCOM_grid(str(small_grid_file))
    grid = FVCOM_grid(
        x=loaded.x.copy(), y=loaded.y.copy(), tri=loaded.tri.copy(),
        h=loaded.h.copy(), obc_nodes=[],
    )
    assert grid.casename == 'FVCOM experiment'
    description = repr(grid)
    assert 'FVCOM experiment FVCOM grid object' in description
    assert f'Number of nodes:        {grid.node_number}' in description
    assert f'Number of triangles:    {grid.cell_number}' in description
    assert 'Number of sigma layers:' not in description
    assert 'Number of sigma levels:' not in description
    assert 'Reference:' in description


def test_loaded_grid_repr_reports_sigma_and_grid_metrics(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    description = repr(grid)
    assert f'Number of sigma layers: {grid.siglay.shape[1]}' in description
    assert f'Number of sigma levels: {grid.siglev.shape[1]}' in description
    assert 'Grid resolution:' in description
    assert 'Grid angles (deg):' in description
    assert grid.reference in description


def test_grid_repr_supports_only_one_sigma_array(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    grid.siglay = None
    description = repr(grid)
    assert 'Number of sigma layers:' not in description
    assert f'Number of sigma levels: {grid.siglev.shape[1]}' in description


def test_one_node_crop_is_recognized_even_when_node_index_is_zero(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    assert not grid.casename.startswith('Cropped ')
    grid.cropped_nodes = np.array([0], dtype=np.int64)
    assert grid.casename.startswith('Cropped ')
    assert repr(grid).startswith('Cropped ')


def test_obc_nodes_assignment_rebuilds_derived_caches(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    original = grid.nodestrings
    assert [segment.tolist() for segment in original] == [[0, 1], [2, 3]]
    assert grid.cellstrings == [[0], [1]]
    _ = grid.coastline_nodes
    assert 'coastline_nodes' in grid.__dict__

    grid.obc_nodes = [0, 1]
    for name in ('_nodestrings', 'cellstrings', 'coastline_nodes'):
        assert name not in grid.__dict__

    assert grid.n_obc == 1
    assert [segment.tolist() for segment in grid.nodestrings] == [[0, 1]]
    assert grid.cellstrings == [[0]]
    assert grid.nodestrings is not original
    assert len(grid.coastline_nodes) == 4


def test_explicit_nodestring_assignment_rebuilds_downstream_caches(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    assert grid.cellstrings == [[0], [1]]
    _ = grid.coastline_nodes
    grid.nodestrings = [np.array([2, 3], dtype=int)]
    assert 'cellstrings' not in grid.__dict__
    assert 'coastline_nodes' not in grid.__dict__
    assert grid.cellstrings == [[1]]
    np.testing.assert_array_equal(grid.nodestrings[0], [2, 3])


def test_tri_reassignment_invalidates_cached_nodestrings(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    grid.obc_nodes = [0, 1]
    cached = grid.nodestrings
    assert [s.tolist() for s in cached] == [[0, 1]]
    _ = grid.cellstrings
    _ = grid.coastline_nodes

    grid.tri = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int64)
    for name in ('_nodestrings', 'cellstrings', 'coastline_nodes', 'boundary_nodes'):
        assert name not in grid.__dict__
    assert grid.nodestrings is not cached
    np.testing.assert_array_equal(grid.nodestrings[0], [0, 1])
    assert grid.cellstrings == [[0]]


def test_xy_update_preserves_saved_nodestring_order_but_clears_geometry_caches(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    stored = grid.nodestrings
    _ = grid.coastline_nodes
    _ = grid.boundary_nodes
    grid.x = grid.x + 5.0
    assert grid.nodestrings is stored
    assert 'coastline_nodes' not in grid.__dict__
    assert 'boundary_nodes' not in grid.__dict__
    assert [s.tolist() for s in grid.nodestrings] == [[0, 1], [2, 3]]


def test_resetting_obc_to_empty_removes_stale_strings(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    _ = grid.nodestrings
    _ = grid.cellstrings
    grid.obc_nodes = []
    assert grid.nodestrings == []
    assert grid.cellstrings == []
    assert grid.n_obc == 0


def test_obc_reassignment_rejects_disconnected_node_instead_of_using_stale_cache(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    _ = grid.nodestrings
    grid.obc_nodes = [0, 2]  # This is the interior diagonal, not a mesh boundary.
    with pytest.raises(ValueError, match='Cannot uniquely order OBC nodes away from the mesh boundary'):
        _ = grid.nodestrings
    # A failed reconstruction must not preserve or cache the earlier nodestring.
    assert '_nodestrings' not in grid.__dict__
