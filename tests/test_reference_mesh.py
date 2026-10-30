"""Optional regression tests based on the paired FVCOM mesh and M.npy."""

import numpy as np

from fvtools.grid.fvcom_grd import FVCOM_grid


def test_reference_2dm_matches_saved_mesh(reference_paths):
    dm_path, npy_path = reference_paths
    # M.npy is a NumPy object array and may use pickle: only run on trusted inputs.
    grid = FVCOM_grid(str(npy_path))
    dm_grid = FVCOM_grid(str(dm_path))
    tri, x, y, strings = dm_grid.tri, dm_grid.x, dm_grid.y, dm_grid.nodestrings
    # Original SMS IDs are not stored on FVCOM_grid; check ND records directly.
    with open(dm_path, encoding='utf-8') as stream:
        node_ids = [int(fields[1]) for line in stream
                    if (fields := line.split()) and fields[0] == 'ND']

    assert len(x) == 48803
    assert len(tri) == 93524
    assert len(strings) == 3
    assert [len(ns) for ns in strings] == [25, 57, 17]
    np.testing.assert_array_equal(node_ids, np.arange(1, 48804))
    np.testing.assert_array_equal(grid.tri, tri)
    np.testing.assert_allclose(grid.x, x, rtol=0, atol=0)
    np.testing.assert_allclose(grid.y, y, rtol=0, atol=0)
    assert len(grid.nodestrings) == len(strings)
    for saved, original in zip(grid.nodestrings, strings):
        np.testing.assert_array_equal(saved, original)
    np.testing.assert_array_equal(grid.obc_nodes, np.concatenate(strings))


def test_reference_saved_grid_sigma_coordinates(reference_paths):
    _, npy_path = reference_paths
    grid = FVCOM_grid(str(npy_path))

    assert grid.h.shape == (48803,)
    assert grid.siglay.shape == (48803, 35)
    assert grid.siglev.shape == (48803, 36)
    assert np.all(np.isfinite(grid.h))
    assert np.all(grid.h > 0)
    np.testing.assert_allclose(grid.siglev[:, 0], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(grid.siglev[:, -1], -1.0, rtol=0, atol=1e-12)
    assert np.all(np.diff(grid.siglev, axis=1) < 0)
    assert np.all(grid.siglev[:, 1:] < grid.siglay)
    assert np.all(grid.siglay < grid.siglev[:, :-1])
