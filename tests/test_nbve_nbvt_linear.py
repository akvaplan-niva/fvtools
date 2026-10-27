"""Exact-regression tests for tge.get_NBVE_NBVT's O(NT) implementation.

This is deliberately not merely a comparison of neighbor *sets*:
cell IDs, corner IDs, ordering, matrix shape, dtype, and -1 padding
must match the old routine because get_NTSN_NBSN consumes the arrays.
"""

import numpy as np
import pytest

from fvtools.grid import tge


def quadratic_reference(mt, nv, cap):
    """Independent, readable version of the legacy node-first search."""
    nbve = np.full((mt, cap + 1), -1, dtype=np.int32)
    nbvt = np.full((mt, cap + 2), -1, dtype=np.int32)
    ntve = np.zeros(mt, dtype=np.int32)
    for node in range(mt):
        for cell, corners in enumerate(nv):
            matches = np.flatnonzero(corners == node)
            if matches.size:
                slot = ntve[node]
                nbve[node, slot] = cell
                nbvt[node, slot] = matches[-1]
                ntve[node] += 1
    return nbve, nbvt, ntve


def assert_same_result(actual, expected):
    assert len(actual) == len(expected) == 3
    for a, e in zip(actual, expected):
        assert a.dtype == e.dtype == np.dtype('int32')
        assert a.shape == e.shape
        np.testing.assert_array_equal(a, e)


def strip_mesh(n):
    """A simple valid strip (three consecutive vertices per triangle)."""
    return np.array([[i, i + 1, i + 2] for i in range(n)], dtype=np.int64)


def fan_mesh(n):
    """An interior node surrounded by n consecutive triangles."""
    return np.array([[0, i + 1, (i + 1) % n + 1] for i in range(n)], dtype=np.int64)


@pytest.mark.parametrize('mt,nv,cap', [
    (4, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64), 20),
    (6, np.array([[2, 1, 0], [2, 3, 0]], dtype=np.int64), 3),
    (8, np.array([[0, 1, 2]], dtype=np.int64), 2),
    (5, np.empty((0, 3), dtype=np.int64), 20),
    (0, np.empty((0, 3), dtype=np.int64), 20),
    (3, np.array([[0, 1, 2]], dtype=np.int64), 0),
    (22, fan_mesh(21), 20),  # capacity is MXNBR_ELEMS + 1
])
def test_exact_match_against_node_first_reference(mt, nv, cap):
    expected = quadratic_reference(mt, nv, cap)
    got = tge.get_NBVE_NBVT(mt, len(nv), nv, cap)
    assert_same_result(got, expected)


def test_each_corner_is_stored_with_its_original_cell_index():
    nv = np.array([[2, 0, 1], [3, 2, 1], [0, 3, 2]], dtype=np.int64)
    nbve, nbvt, ntve = tge.get_NBVE_NBVT(4, 3, nv, 5)
    np.testing.assert_array_equal(ntve, [2, 2, 3, 2])
    for node in range(4):
        for j, k in zip(nbve[node, :ntve[node]], nbvt[node, :ntve[node]]):
            assert nv[j, k] == node
        assert np.all(np.diff(nbve[node, :ntve[node]]) > 0)
        assert np.all(nbve[node, ntve[node]:] == -1)
        assert np.all(nbvt[node, ntve[node]:] == -1)


@pytest.mark.parametrize('seed', [1, 19, 42, 1001])
def test_exact_match_on_shuffled_triangles_and_cyclic_vertex_rotations(seed):
    rng = np.random.default_rng(seed)
    nv = np.vstack((strip_mesh(250), fan_mesh(8) + np.array([260, 260, 260])))
    nv = nv[rng.permutation(len(nv))].copy()
    rotations = rng.integers(0, 3, size=len(nv))
    for i, rotate in enumerate(rotations):
        nv[i] = np.roll(nv[i], rotate)
    expected = quadratic_reference(269, nv, 20)
    actual = tge.get_NBVE_NBVT(269, len(nv), nv, 20)
    assert_same_result(actual, expected)


@pytest.mark.parametrize('nv,mt,cap,error', [
    (np.array([[0, 1]], dtype=np.int64), 3, 20, 'shape'),
    (np.array([[0, 1, 3]], dtype=np.int64), 3, 20, 'outside'),
    (np.array([[-1, 1, 2]], dtype=np.int64), 3, 20, 'outside'),
    (np.array([[0, 0, 1]], dtype=np.int64), 3, 20, 'repeated'),
    (np.array([[0, 1, 2]], dtype=np.int64), 3, -1, 'nonnegative'),
    (np.array([[0, 1, 2]], dtype=np.int64), -1, 20, 'nonnegative'),
    (fan_mesh(22), 23, 20, 'capacity'),
])
def test_invalid_inputs_fail_explicitly(nv, mt, cap, error):
    with pytest.raises(ValueError, match=error):
        tge.get_NBVE_NBVT(mt, len(nv), nv, cap)


def test_mismatching_cell_count_is_rejected():
    nv = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match='shape'):
        tge.get_NBVE_NBVT(3, 2, nv, 20)


def test_input_connectivity_is_not_modified():
    nv = fan_mesh(7)
    original = nv.copy()
    tge.get_NBVE_NBVT(8, len(nv), nv, 20)
    np.testing.assert_array_equal(nv, original)


def test_downstream_topological_reordering_is_unchanged():
    """get_NTSN_NBSN must give identical output with either input builder."""
    nv = fan_mesh(4)
    mt, nt, cap = 5, len(nv), 20
    nbe = tge.get_NBE(nt, mt, nv)
    _, isonb = tge.get_BOUNDARY(nt, mt, nbe, nv)
    legacy = quadratic_reference(mt, nv, cap)
    updated = tge.get_NBVE_NBVT(mt, nt, nv, cap)
    assert_same_result(updated, legacy)

    outputs = []
    for nbve, nbvt, ntve in (legacy, updated):
        # This routine mutates NBVE/NBVT in-place, so compare independent copies.
        outputs.append(tge.get_NTSN_NBSN(
            nbve.copy(), ntve.copy(), nbvt.copy(),
            nbe.copy(), nv.copy(), isonb.copy(), cap, mt,
        ))
    for a, b in zip(outputs[0], outputs[1]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(outputs[0][-1], np.zeros(mt, dtype=int))


def test_real_fvcom_reference_mesh_matches_independent_vectorized_algorithm(reference_paths):
    """Checks all ~280,000 incidences, not sampled rows or neighbor sets."""
    _, path = reference_paths
    # NumPy object-array loading uses pickle: only use this trusted fixture.
    mesh = np.load(path, allow_pickle=True).item()
    nv = np.asarray(mesh['nv'], dtype=np.int64)
    mt, nt, cap = len(mesh['x']), len(nv), 20
    nbve, nbvt, ntve = tge.get_NBVE_NBVT(mt, nt, nv, cap)

    # Independent grouping algorithm: stable sort flattened corner records by
    # node ID. Stable ordering preserves the original increasing cell index.
    flat_nodes = nv.ravel()
    sort_index = np.argsort(flat_nodes, kind='stable')
    nodes = flat_nodes[sort_index]
    expected_counts = np.bincount(flat_nodes, minlength=mt)
    assert expected_counts.max() <= cap + 1
    starts = np.cumsum(expected_counts, dtype=np.int64) - expected_counts
    slots = np.arange(len(flat_nodes)) - np.repeat(starts, expected_counts)
    cells = np.repeat(np.arange(nt, dtype=np.int32), 3)
    corners = np.tile(np.arange(3, dtype=np.int32), nt)

    expected_bve = np.full(nbve.shape, -1, dtype=np.int32)
    expected_bvt = np.full(nbvt.shape, -1, dtype=np.int32)
    expected_bve[nodes, slots] = cells[sort_index]
    expected_bvt[nodes, slots] = corners[sort_index]

    assert_same_result((nbve, nbvt, ntve),
                       (expected_bve, expected_bvt,
                        expected_counts.astype(np.int32)))
    assert mt == 48803 and nt == 93524
