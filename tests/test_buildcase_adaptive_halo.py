"""Adaptive source-window regression: dense, nonuniform unstructured bathymetry."""
import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator
try:
    from fvtools.prepro._buildcase import tiled_bathymetry as tiled
except ImportError:  # stand-alone isolated testing
    import tiled_bathymetry as tiled

TilingOptions = tiled.TilingOptions
interpolate_unstructured_npy = tiled.interpolate_unstructured_npy


@pytest.mark.parametrize('origin', [(0., 0.), (500_000., 7_700_000.)])
@pytest.mark.parametrize('workers', [1, 2])
def test_dense_halo_avoids_single_node_fallback(tmp_path, monkeypatch, origin, workers):
    rng = np.random.default_rng(63)
    offset = np.asarray(origin, dtype=np.float64)
    inner = rng.uniform(-0.025, 0.025, (8000,2))
    outer = np.array([[-10.,-10.],[10.,-10.],[10.,10.],[-10.,10.]])
    positions = np.concatenate([inner,outer])
    depth = np.cos(positions[:,0]*20.) + np.sin(positions[:,1]*10.)
    src = np.column_stack((positions+offset,depth))
    targets = rng.uniform(-0.008,0.008,(16,2))
    path=tmp_path / f'adaptive_{workers}.npy'
    np.save(path,src)
    fallback_calls=[]
    real=tiled._interpolate_dense_single
    def count_fallback(*args,**kwargs):
        fallback_calls.append(1)
        return real(*args,**kwargs)
    monkeypatch.setattr(tiled,'_interpolate_dense_single',count_fallback)
    options=TilingOptions(workers=workers,max_triangulation_points=150,
                          nodes_per_tile=16,bins_per_axis=9,source_chunk_rows=501)
    got=interpolate_unstructured_npy(path,targets+offset,options=options)
    want=LinearNDInterpolator(src[:,:2]-offset,depth)((targets+offset)-offset)
    np.testing.assert_allclose(got,want,rtol=1e-9,atol=1e-9,equal_nan=True)
    if workers == 1:
        assert not fallback_calls, 'Dense tiles should shrink the source halo, not stream every node'


def test_truly_unavoidable_dense_halo_retains_safe_fallback(tmp_path):
    # The new strategy must not silently drop the old streaming fallback when
    # a local region is geometrically insufficient to certify a simplex.
    rng = np.random.default_rng(4)
    clustered = rng.uniform(-.001,.001,(1200,2))
    outlying = np.array([[-30.,-30.],[30.,-30.],[30.,30.],[-30.,30.]])
    coords=np.vstack([clustered,outlying])
    z=coords[:,0]*0.5+coords[:,1]**2*0.05
    path=tmp_path/'gap.npy'
    np.save(path,np.column_stack([coords,z]))
    queries=np.array([[0.,0.],[5.,6.],[15.,18.]])
    got=interpolate_unstructured_npy(path,queries,options=TilingOptions(
        workers=1,nodes_per_tile=3,max_triangulation_points=90,bins_per_axis=9))
    expected=LinearNDInterpolator(coords,z)(queries)
    np.testing.assert_allclose(got,expected,rtol=1e-9,atol=1e-9,equal_nan=True)
