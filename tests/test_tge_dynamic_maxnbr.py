"""Regression tests for dynamic TGE adjacency widths.

Compare valid connectivity, topology, and ART1, rather than arrays' padded
shapes: choosing the true maximum valence is meant to shrink the latter.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.grid.fvcom import tge


def _mesh(n, *, closed):
    """Clockwise triangle fans: closed centre is interior, open centre is boundary."""
    if closed:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xy = np.vstack(([0.0, 0.0], np.column_stack((np.cos(angles), np.sin(angles)))))
        nv = np.array([[0, (j + 1) % n + 1, j + 1] for j in range(n)], dtype=np.int64)
    else:
        angles = np.linspace(0, np.pi / 2, n + 1)
        xy = np.vstack(([0.0, 0.0], np.column_stack((np.cos(angles), np.sin(angles)))))
        nv = np.array([[0, j + 2, j + 1] for j in range(n)], dtype=np.int64)
    return xy, nv


def _maxnbr(nv, mt):
    counts = np.bincount(nv.ravel(), minlength=mt)
    return int(counts.max(initial=0))


def _pipeline(xy, nv, capacity, obc_nodes=None):
    """Execute the actual production topology kernels at a chosen capacity."""
    mt, nt = len(xy), len(nv)
    nbe = tge.get_NBE(nt, mt, nv)
    cell_flags, node_flags = tge.get_BOUNDARY(nt, mt, nbe, nv)
    if obc_nodes is not None:
        node_flags, cell_flags = tge.set_boundary(
            node_flags.copy(), cell_flags.copy(), nv, nbe,
            np.asarray(obc_nodes, dtype=np.int64), nt,
        )
    nbve, nbvt, ntve = tge.get_NBVE_NBVT(mt, nt, nv, capacity)
    ntsn, nbsn, bve, bvt, invalid = tge.get_NTSN_NBSN(
        nbve.copy(), ntve.copy(), nbvt.copy(),
        nbe, nv, node_flags, capacity, mt,
    )
    assert not invalid.any(), f'Unexpected invalid mesh nodes: {np.flatnonzero(invalid).tolist()}'
    xc = xy[nv, 0].mean(axis=1)
    yc = xy[nv, 1].mean(axis=1)
    art1 = tge.get_art1(nt, mt, xy[:, 0], xy[:, 1], xc, yc,
                        nv, node_flags, ntve, bve, bvt, nbsn)
    return dict(nbe=nbe, cell_flags=cell_flags, node_flags=node_flags,
                nbve=bve, nbvt=bvt, ntve=ntve, ntsn=ntsn,
                nbsn=nbsn, art1=art1, capacity=capacity)


def _compare(old, new, mt):
    """Exactly equal valid records, including order; no requirement on padding."""
    for key in ('nbe', 'cell_flags', 'node_flags', 'ntve', 'ntsn', 'art1'):
        np.testing.assert_array_equal(new[key], old[key], err_msg=key)
    assert new['nbve'].dtype == old['nbve'].dtype == np.int32
    assert new['nbvt'].dtype == old['nbvt'].dtype == np.int32
    assert new['nbsn'].dtype == old['nbsn'].dtype == np.int64
    for node in range(mt):
        # NTVE entries refer to real neighboring triangles (and vertices).
        count = int(old['ntve'][node])
        np.testing.assert_array_equal(new['nbve'][node, :count + 1], old['nbve'][node, :count + 1])
        np.testing.assert_array_equal(new['nbvt'][node, :count + 1], old['nbvt'][node, :count + 1])
        # NTSN includes TGE's boundary sentinels, not just adjacent nodes.
        nn = int(old['ntsn'][node])
        np.testing.assert_array_equal(new['nbsn'][node, :nn], old['nbsn'][node, :nn])
        assert np.all(new['nbve'][node, count + 1:] == -1)
        assert np.all(new['nbvt'][node, count + 1:] == -1)
        assert np.all(new['nbsn'][node, nn:] == -1)


@pytest.mark.parametrize('spokes', [3, 5, 8, 19, 20, 21, 22, 31, 51, 60])
@pytest.mark.parametrize('closed', [False, True])
def test_fan_topology_and_control_volume_area_match_legacy_capacity(spokes, closed):
    xy, nv = _mesh(spokes, closed=closed)
    actual = _maxnbr(nv, len(xy))
    assert actual == spokes
    old = _pipeline(xy, nv, max(20, actual))
    new = _pipeline(xy, nv, actual)
    assert new['nbve'].shape == (len(xy), actual + 1)
    assert new['nbvt'].shape == (len(xy), actual + 2)
    assert new['nbsn'].shape == (len(xy), actual + 3)
    _compare(old, new, len(xy))


def test_true_maximum_valence_is_not_forced_to_twenty():
    xy, nv = _mesh(4, closed=True)
    out = SimpleNamespace(MT=len(xy), NT=len(nv), NV=nv,
                          NBE=tge.get_NBE(len(nv), len(xy), nv))
    _, out.ISONB = tge.get_BOUNDARY(out.NT, out.MT, out.NBE, out.NV)
    tge._find_node_connectivity(out, SimpleNamespace(plot_grid=lambda: None), False)
    assert _maxnbr(nv, len(xy)) == 4
    assert out.NBVE.shape == (len(xy), 5)
    assert out.NBVT.shape == (len(xy), 6)
    assert out.NBSN.shape == (len(xy), 7)
    np.testing.assert_array_equal(out.NTVE, [4, 2, 2, 2, 2])
    np.testing.assert_array_equal(out.NTSN, [5, 5, 5, 5, 5])


@pytest.mark.parametrize('spokes,closed', [(3, True), (8, True), (21, True),
                                          (32, False), (60, False)])
def test_find_node_connectivity_calls_dynamic_capacity(spokes, closed):
    xy, nv = _mesh(spokes, closed=closed)
    mt = len(xy)
    out = SimpleNamespace(MT=mt, NT=len(nv), NV=nv,
                          NBE=tge.get_NBE(len(nv), mt, nv))
    _, out.ISONB = tge.get_BOUNDARY(out.NT, out.MT, out.NBE, out.NV)
    tge._find_node_connectivity(out, SimpleNamespace(plot_grid=lambda: None), False)
    k = _maxnbr(nv, mt)
    assert out.NBVE.shape == (mt, k + 1)
    assert out.NBVT.shape == (mt, k + 2)
    assert out.NBSN.shape == (mt, k + 3)
    assert out.NTVE.max() == k
    assert not np.any(out.NTVE > k)


def test_real_reference_grid_uses_eight_and_keeps_all_topology_and_art1(reference_paths):
    _, filename = reference_paths
    data = np.load(filename, allow_pickle=True).item()  # trusted reference file
    nv = np.asarray(data['nv'], dtype=np.int64)
    xy = np.column_stack((data['x'], data['y'])).astype(float)
    assert xy.shape == (48803, 2) and nv.shape == (93524, 3)
    cap = _maxnbr(nv, len(xy))
    assert cap == 8
    obc = data['obc_nodes']
    old = _pipeline(xy, nv, 20, obc_nodes=obc)
    new = _pipeline(xy, nv, cap, obc_nodes=obc)
    _compare(old, new, len(xy))
    assert old['nbve'].shape[1] == 21
    assert new['nbve'].shape[1] == 9


def test_real_reference_grid_integration_through_find_node_connectivity(reference_paths):
    _, filename = reference_paths
    data = np.load(filename, allow_pickle=True).item()  # trusted reference file
    nv = np.asarray(data['nv'], dtype=np.int64)
    mt, nt = len(data['x']), len(nv)
    out = SimpleNamespace(MT=mt, NT=nt, NV=nv, NBE=tge.get_NBE(nt, mt, nv))
    _, out.ISONB = tge.get_BOUNDARY(nt, mt, out.NBE, nv)
    tge._find_node_connectivity(out, SimpleNamespace(plot_grid=lambda: None), False)
    assert out.NBVE.shape == (mt, 9)
    assert out.NBVT.shape == (mt, 10)
    assert out.NBSN.shape == (mt, 11)
    np.testing.assert_array_equal(out.NTVE, np.bincount(nv.ravel(), minlength=mt))


def test_empty_mesh_maxnbr_is_zero_and_direct_connectivity_arrays_are_well_formed():
    nv = np.empty((0, 3), dtype=np.int64)
    assert _maxnbr(nv, 0) == 0
    for mt in (0, 5):
        cap = _maxnbr(nv, mt)
        nbve, nbvt, ntve = tge.get_NBVE_NBVT(mt, 0, nv, cap)
        assert nbve.shape == (mt, 1)
        assert nbvt.shape == (mt, 2)
        np.testing.assert_array_equal(ntve, np.zeros(mt, dtype=np.int32))
        assert np.all(nbve == -1) and np.all(nbvt == -1)
    # Full TGE has separate assumptions about a valid, nonempty mesh.
    # Do not call _find_node_connectivity for isolated nodes here.
