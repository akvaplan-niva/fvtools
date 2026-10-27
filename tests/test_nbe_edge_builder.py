"""Exact regression tests for the edge-based tge.get_NBE replacement.

All valid-mesh assertions compare entire NBE matrices, including column
ordering and -1 boundary sentinels. Legacy implementation is preserved here
as an independent oracle and not imported from the production code.
"""

import numpy as np
import pytest
from numba import jit

from fvtools.grid import tge


@jit(nopython=True)
def legacy_nbe(nt, mt, nv):
    """Faithful reference copy of original cell-via-shared-nodes algorithm.

    Restricted to valid test meshes with <=50 triangles per node, matching
    its original fixed-width temporary array.
    """
    nbe = -np.ones((nt, 3), dtype=np.int64)
    cells = -np.ones((nt, 50), dtype=np.int64)
    counts = np.zeros(mt, dtype=np.int64)
    for i in range(nt):
        for k in range(3):
            node = nv[i, k]
            cells[node, counts[node]] = i
            counts[node] += 1
    for i in range(nt):
        n0, n1, n2 = nv[i]
        for a in range(counts[n0]):
            for b in range(counts[n1]):
                if cells[n0, a] == cells[n1, b] and cells[n0, a] != i:
                    nbe[i, 2] = cells[n0, a]
        for b in range(counts[n1]):
            for c in range(counts[n2]):
                if cells[n1, b] == cells[n2, c] and cells[n1, b] != i:
                    nbe[i, 0] = cells[n1, b]
        for a in range(counts[n0]):
            for c in range(counts[n2]):
                if cells[n2, c] == cells[n0, a] and cells[n2, c] != i:
                    nbe[i, 1] = cells[n2, c]
    return nbe


def fan_mesh(spokes):
    return np.asarray([[0, i + 1, (i + 1) % spokes + 1]
                       for i in range(spokes)], dtype=np.int64)


def hole_mesh():
    """Ring of eight triangles surrounding an unmeshed square (island)."""
    return np.asarray([
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)


def rectangular_mesh(nx=11, ny=8):
    def node(i, j):
        return j * (nx + 1) + i
    triangles = []
    for j in range(ny):
        for i in range(nx):
            a, b = node(i, j), node(i + 1, j)
            c, d = node(i, j + 1), node(i + 1, j + 1)
            triangles.extend(((a, b, d), (a, d, c)))
    return np.asarray(triangles, dtype=np.int64), (nx + 1) * (ny + 1)


def check_exact(nv, mt):
    nt = len(nv)
    old_input = nv.copy()
    expected = legacy_nbe(nt, mt, nv)
    got = tge.get_NBE(nt, mt, nv)
    assert got.shape == expected.shape == (nt, 3)
    assert got.dtype == expected.dtype == np.int64
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(nv, old_input)
    for i in range(nt):
        for corner in range(3):
            other = got[i, corner]
            if other < 0:
                continue
            assert other != i
            edge = set(nv[i, [(corner + 1) % 3, (corner + 2) % 3]])
            matching = [j for j in range(3) if set(nv[other, [(j + 1) % 3, (j + 2) % 3]]) == edge]
            assert len(matching) == 1
            assert got[other, matching[0]] == i
    return got


def test_empty_mesh_and_isolated_nodes():
    nv = np.empty((0, 3), dtype=np.int64)
    check_exact(nv, mt=6)


def test_single_cell_has_three_exterior_edges():
    got = check_exact(np.array([[0, 1, 2]], dtype=np.int64), 5)
    np.testing.assert_array_equal(got, [[-1, -1, -1]])


def test_square_exact_opposite_corner_convention():
    nv = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    got = check_exact(nv, 4)
    np.testing.assert_array_equal(got, [[-1, 1, -1], [-1, -1, 0]])


@pytest.mark.parametrize('spokes', [3, 4, 5, 8, 12, 20, 30])
def test_fan_interior_node_and_boundary(spokes):
    got = check_exact(fan_mesh(spokes), spokes + 1)
    assert np.sum(got == -1) == spokes


def test_ring_with_island_has_two_boundary_components():
    got = check_exact(hole_mesh(), 8)
    assert np.sum(got == -1) == 8  # four exterior and four island edges


@pytest.mark.parametrize('seed', [0, 1, 17, 42, 1996, 2026])
def test_exact_under_cell_shuffle_and_corner_rotations(seed):
    rng = np.random.default_rng(seed)
    nv, mt = rectangular_mesh(17, 11)
    nv = nv[rng.permutation(len(nv))].copy()
    rolls = rng.integers(0, 3, size=len(nv))
    for i, shift in enumerate(rolls):
        nv[i] = np.roll(nv[i], int(shift))
    if seed % 2:
        nv[:, [1, 2]] = nv[:, [2, 1]]
    check_exact(nv, mt)


@pytest.mark.parametrize('nv,mt,error', [
    (np.array([[0, 1]], dtype=np.int64), 3, 'shape'),
    (np.array([[0, 1, 3]], dtype=np.int64), 3, 'outside'),
    (np.array([[-1, 1, 2]], dtype=np.int64), 3, 'outside'),
    (np.array([[0, 0, 1]], dtype=np.int64), 3, 'repeated'),
    (np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64), 5, '[Nn]on-manifold'),
    (np.array([[0, 1, 2]], dtype=np.int64), -1, 'nonnegative'),
])
def test_invalid_mesh_is_rejected(nv, mt, error):
    with pytest.raises(ValueError, match=error):
        tge.get_NBE(len(nv), mt, nv)


def test_inconsistent_nt_is_rejected():
    with pytest.raises(ValueError, match='shape'):
        tge.get_NBE(2, 3, np.asarray([[0, 1, 2]], dtype=np.int64))


def test_zero_nodes_and_zero_cells():
    nv = np.empty((0, 3), dtype=np.int64)
    got = tge.get_NBE(0, 0, nv)
    assert got.shape == (0, 3) and got.dtype == np.int64


def test_no_fixed_fifty_cell_limit_for_high_valence():
    nv = fan_mesh(60)
    got = tge.get_NBE(len(nv), 61, nv)
    assert got.shape == (60, 3)
    # All inner edges are paired, and each fan cell has an exterior edge.
    assert np.sum(got == -1) == 60


@pytest.mark.parametrize('nv,mt,obc', [
    (np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64), 4, np.array([0, 1], dtype=np.int64)),
    (hole_mesh(), 8, np.array([0, 1, 2, 3], dtype=np.int64)),
    (fan_mesh(8), 9, np.array([1, 2, 3], dtype=np.int64)),
])
def test_boundary_and_open_boundary_flags_are_identical(nv, mt, obc):
    nt = len(nv)
    old_nbe, new_nbe = legacy_nbe(nt, mt, nv), tge.get_NBE(nt, mt, nv)
    expected_cell, expected_node = tge.get_BOUNDARY(nt, mt, old_nbe, nv)
    got_cell, got_node = tge.get_BOUNDARY(nt, mt, new_nbe, nv)
    np.testing.assert_array_equal(got_cell, expected_cell)
    np.testing.assert_array_equal(got_node, expected_node)
    expected_o_node, expected_o_cell = tge.set_boundary(
        expected_node.copy(), expected_cell.copy(), nv, old_nbe, obc, nt)
    got_o_node, got_o_cell = tge.set_boundary(
        got_node.copy(), got_cell.copy(), nv, new_nbe, obc, nt)
    np.testing.assert_array_equal(got_o_node, expected_o_node)
    np.testing.assert_array_equal(got_o_cell, expected_o_cell)


def test_downstream_node_connectivity_reordering_is_identical():
    nv = fan_mesh(4)
    nt, mt, cap = len(nv), 5, 20
    nbve, nbvt, ntve = tge.get_NBVE_NBVT(mt, nt, nv, cap)
    nbe_old = legacy_nbe(nt, mt, nv)
    nbe_new = tge.get_NBE(nt, mt, nv)
    _, isonb = tge.get_BOUNDARY(nt, mt, nbe_old, nv)
    out_old = tge.get_NTSN_NBSN(
        nbve.copy(), ntve.copy(), nbvt.copy(), nbe_old, nv, isonb, cap, mt)
    out_new = tge.get_NTSN_NBSN(
        nbve.copy(), ntve.copy(), nbvt.copy(), nbe_new, nv, isonb, cap, mt)
    for old, new in zip(out_old, out_new):
        np.testing.assert_array_equal(new, old)


def test_small_fvcom_grid_tge_boundary_integration(small_grid_file):
    from fvtools.grid.fvcom_grd import FVCOM_grid
    mesh = FVCOM_grid(str(small_grid_file))
    old = legacy_nbe(mesh.cell_number, mesh.node_number, mesh.tri)
    new = tge.get_NBE(mesh.cell_number, mesh.node_number, mesh.tri)
    np.testing.assert_array_equal(new, old)
    boundary = tge.calculate_nbe_and_boundaries(mesh)
    # TGE may reorient NV before deriving its edge-column convention.
    np.testing.assert_array_equal(
        boundary.NBE, legacy_nbe(len(boundary.NV), mesh.node_number, boundary.NV))


def test_full_reference_grid_exact_nbe_and_boundary_flags(reference_paths):
    """Check all 280572 entries, not a sample of cells or neighbour sets."""
    _, path = reference_paths
    mesh = np.load(path, allow_pickle=True).item()  # trusted optional fixture
    nv = np.asarray(mesh['nv'], dtype=np.int64)
    mt, nt = len(mesh['x']), len(nv)
    assert mt == 48803 and nt == 93524
    new = check_exact(nv, mt)
    legacy = legacy_nbe(nt, mt, nv)
    a_cell, a_node = tge.get_BOUNDARY(nt, mt, legacy, nv)
    b_cell, b_node = tge.get_BOUNDARY(nt, mt, new, nv)
    np.testing.assert_array_equal(b_cell, a_cell)
    np.testing.assert_array_equal(b_node, a_node)
    np.testing.assert_array_equal(new == -1, legacy == -1)
