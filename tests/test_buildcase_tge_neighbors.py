"""Regression checks for TGE-backed smoothing adjacency in BuildCase.

The TGE rings include closure sentinels; BuildCase must NOT include these
when averaging bathymetry or choosing the deepest neighbor in MEO smoothing.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.tri import Triangulation

from fvtools.prepro._buildcase.depth import DepthHandler


def fan(n, closed):
    angles = np.linspace(0., 2.*np.pi if closed else np.pi, n if closed else n+1, endpoint=not closed)
    x = np.r_[0., np.cos(angles)]
    y = np.r_[0., np.sin(angles)]
    ntri = n if closed else n
    tri = np.array([[0, j+1, (j+1) % n+1] for j in range(n)] if closed
                   else [[0, j+1, j+2] for j in range(n)], dtype=np.int64)
    return SimpleNamespace(x=x, y=y, tri=tri)


def legacy(obj):
    """Independent Matplotlib edge-derived reference, no retired BuildCase API."""
    edges = Triangulation(obj.x, obj.y, obj.tri).edges
    counts = np.bincount(edges.ravel(), minlength=len(obj.x)).astype(np.int64)
    neighbors = np.full((len(obj.x), int(counts.max(initial=0))), -1, dtype=np.int64)
    offsets = np.zeros(len(obj.x), dtype=np.int64)
    for left, right in edges:
        neighbors[left, offsets[left]] = right
        offsets[left] += 1
        neighbors[right, offsets[right]] = left
        offsets[right] += 1
    np.testing.assert_array_equal(offsets, counts)
    return (neighbors, counts), edges


@pytest.mark.parametrize('closed', [True, False])
@pytest.mark.parametrize('n', [3, 5, 8, 16, 25])
def test_tge_smoothing_neighbors_match_legacy_connectivity(closed, n):
    obj = fan(n, closed)
    (old_neighbors, old_counts), edges = legacy(obj)
    DepthHandler.setup_metrics(obj)
    np.testing.assert_array_equal(obj.ntsn, old_counts)
    np.testing.assert_array_equal(obj.edges, edges)
    assert obj.nbsn.shape[1] == int(old_counts.max())
    for node, count in enumerate(old_counts):
        old = old_neighbors[node, :count]
        new = obj.nbsn[node, :count]
        assert set(new.tolist()) == set(old.tolist())
        assert len(set(new.tolist())) == count
        assert node not in new
        assert np.all(obj.nbsn[node, count:] == -1)
    # Numerical differences can occur at ~1 ulp when the neighbor order changes.
    depth = np.linspace(5., 120., len(obj.x))
    h_old = DepthHandler.laplacian_filter(obj.x, obj.y, depth, old_neighbors, old_counts, .2)
    h_new = DepthHandler.laplacian_filter(obj.x, obj.y, depth, obj.nbsn, obj.ntsn, .2)
    np.testing.assert_allclose(h_new, h_old, rtol=1e-14, atol=1e-12)


def test_mellor_ezer_oey_still_receives_no_self_or_duplicate_nodes():
    obj = fan(16, True)
    DepthHandler.setup_metrics(obj)
    assert obj.ntsn[0] == 16
    assert np.all(obj.ntsn[1:] == 3)
    h = np.linspace(5., 100., len(obj.x))
    (old_nbsn, old_ntsn), _ = legacy(obj)
    got, _, _ = DepthHandler.mellor_ezer_oey(h, obj.ntsn, obj.nbsn, .18)
    expected, _, _ = DepthHandler.mellor_ezer_oey(h, old_ntsn, old_nbsn, .18)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_setup_metrics_does_not_invoke_full_tge_or_write_files():
    obj = fan(5, True)
    assert not hasattr(obj, 'T')
    DepthHandler.setup_metrics(obj)
    assert not hasattr(obj, 'T')
    assert not hasattr(obj, '_T')


def test_empty_mesh_rejected_early():
    obj = SimpleNamespace(x=np.array([]), y=np.array([]), tri=np.empty((0,3), dtype=np.int64))
    with pytest.raises(ValueError, match='empty mesh'):
        DepthHandler.setup_metrics(obj)


def test_real_mesh_neighbor_sets_and_smoothed_depths(reference_paths):
    """Trusted reference: verifies TGE conventions on a realistic coastal mesh."""
    _, filename = reference_paths
    grid = np.load(filename, allow_pickle=True).item()
    obj = SimpleNamespace(x=grid['x'], y=grid['y'], tri=grid['nv'])
    (old_nbsn, old_ntsn), edges = legacy(obj)
    DepthHandler.setup_metrics(obj)
    np.testing.assert_array_equal(obj.ntsn, old_ntsn)
    np.testing.assert_array_equal(obj.edges, edges)
    for i in range(len(obj.x)):
        count = int(old_ntsn[i])
        assert set(obj.nbsn[i, :count]) == set(old_nbsn[i, :count])
    bath = np.asarray(grid['h'], dtype=float)
    h_old = DepthHandler.laplacian_filter(obj.x, obj.y, bath, old_nbsn, old_ntsn, .2)
    h_new = DepthHandler.laplacian_filter(obj.x, obj.y, bath, obj.nbsn, obj.ntsn, .2)
    np.testing.assert_allclose(h_new, h_old, rtol=0, atol=1e-12)
