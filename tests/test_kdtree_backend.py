"""Cross-version tests for optional pykdtree / scipy KDTree backend."""

import numpy as np

from fvtools.util.kdtree import BACKEND, KDTree


def test_backend_exists_and_is_jit_independent():
    assert BACKEND in ("pykdtree", "scipy")
    assert callable(KDTree)


def test_nearest_neighbour_distances_and_indices_agree_with_geometry():
    sources = np.array([[0., 0.], [3., 0.], [0., 4.], [5., 5.]])
    target = np.array([[0.2, 0.1], [2.8, 0.2], [0.3, 3.5], [4.8, 4.8]])
    expected_indices = np.array([0, 1, 2, 3])
    distances, indices = KDTree(sources).query(target)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(distances, np.linalg.norm(target - sources[indices], axis=1), rtol=1e-12, atol=1e-12)


def test_multiple_neighbours_sorted_and_distances_match():
    sources = np.array([[0., 0.], [1., 0.], [0., 2.], [4., 4.]])
    targets = np.array([[0.1, 0.01], [3.9, 4.1]])
    dist, indices = KDTree(sources).query(targets, k=3)
    assert dist.shape == (2, 3)
    assert indices.shape == (2, 3)
    np.testing.assert_array_equal(np.argsort(dist, axis=1), np.tile(np.arange(3), (2,1)))
    for i in range(len(targets)):
        np.testing.assert_allclose(dist[i], np.linalg.norm(sources[indices[i]] - targets[i], axis=1), rtol=1e-12, atol=1e-12)
