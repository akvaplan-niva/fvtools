"""Regression tests for memory-budgeted global kNN / local exact Delaunay mode."""
import os
import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator
from fvtools.prepro._buildcase import tiled_bathymetry as tiled
from fvtools.prepro._buildcase import fast_kdtree as fast

@pytest.fixture(autouse=True)
def fast_mode(monkeypatch):
    monkeypatch.setenv('FVTOOLS_BATHY_FAST_INDEX_MIN_POINTS','0')
    monkeypatch.setenv('FVTOOLS_BATHY_FAST_INDEX_MB','1024')


def run(tmp_path, xy, q, *, workers=1, cap=3000, suffix='a', bins=32):
    filename=tmp_path/f'fast_{suffix}.npy'
    z = np.sin((xy[:,0]-xy[:,0].min())/12.) + 0.05*(xy[:,1]-xy[:,1].min())
    src=np.column_stack((xy,z))
    np.save(filename,src)
    res=tiled.interpolate_unstructured_npy(filename,q,options=tiled.TilingOptions(
        workers=workers,max_triangulation_points=cap,nodes_per_tile=80,bins_per_axis=bins,source_chunk_rows=1234))
    # Normalize source/query to a local frame to keep reference robust at UTM northings.
    origin=src[0,:2]
    expected=LinearNDInterpolator(src[:,:2]-origin,src[:,2])(q-origin)
    np.testing.assert_allclose(res,expected,atol=1e-8,rtol=1e-8,equal_nan=True)
    np.testing.assert_array_equal(np.load(filename),src)
    return res

@pytest.mark.parametrize('shift',[(0.,0.),(500_000.,7_700_000.)])
def test_uniform_fast_tree_matches_global(tmp_path,shift,capsys):
    rng=np.random.default_rng(12)
    xy = rng.uniform(0,100,(9000,2))+np.array(shift)
    q=rng.uniform(5,95,(230,2))+np.array(shift)
    got=run(tmp_path,xy,q,cap=3000)
    out=capsys.readouterr().out
    assert 'Fast global kNN / local Delaunay' in out
    assert 'Interpolation of grid nodes: 100.0%' in out
    assert np.isfinite(got).all()

@pytest.mark.parametrize('shift',[(0.,0.),(500_000.,7_700_000.)])
def test_highly_nonuniform_gap_has_no_disk_single_node_fallback(tmp_path,shift,capsys):
    rng=np.random.default_rng(45)
    cloud=np.vstack((rng.normal(-40,1,(7000,2)),
                     rng.normal(40,1,(7000,2)),
                     [[-200,-200],[200,-200],[-200,200],[200,200]]))
    queries=rng.uniform(-30,30,(50,2))
    run(tmp_path, cloud+shift, queries+shift,cap=3000)
    out=capsys.readouterr().out
    assert 'Fast kNN: 0 hard nodes needed disk fallback' in out


def test_serial_parallel_same_and_no_global_delaunay(tmp_path,monkeypatch):
    rng=np.random.default_rng(17)
    xy=rng.uniform(0,50,(12000,2))
    q=rng.uniform(5,45,(150,2))
    monkeypatch.setattr(tiled,'_cpu_slots',lambda:2)
    orig=fast.Delaunay
    sizes=[]
    def restricted(x):
        sizes.append(len(x))
        assert len(x)<=2000
        return orig(x)
    monkeypatch.setattr(fast,'Delaunay',restricted)
    a=run(tmp_path,xy,q,workers=1,cap=2000,suffix='serial')
    b=run(tmp_path,xy,q,workers=2,cap=2000,suffix='parallel')
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-10,equal_nan=True)
    assert sizes


def test_fast_index_respects_budget_and_falls_back_to_disk(tmp_path,monkeypatch,capsys):
    monkeypatch.setenv('FVTOOLS_BATHY_FAST_INDEX_MB','1')
    rng=np.random.default_rng(55)
    xy=rng.uniform(0,100,(12000,2))
    q=rng.uniform(5,95,(35,2))
    run(tmp_path,xy,q,workers=1,cap=3000)
    output=capsys.readouterr().out
    assert 'Fast index unavailable' in output
    assert 'using disk-only mode' in output


def test_float32_source_keeps_depth_precision(tmp_path):
    rng=np.random.default_rng(63)
    xy=rng.uniform(0,100,(4000,2)).astype('f4')
    z=(np.cos(xy[:,0]/5.)+xy[:,1]/10.).astype('f4')
    src=np.column_stack((xy,z));q=rng.uniform(5,95,(65,2))
    fn=tmp_path/'source_f32.npy'
    np.save(fn,src)
    got=tiled.interpolate_unstructured_npy(fn,q,options=tiled.TilingOptions(
        workers=1,bins_per_axis=16,max_triangulation_points=1000,nodes_per_tile=32))
    exp=LinearNDInterpolator(src[:,:2],src[:,2])(q)
    np.testing.assert_allclose(got,exp,rtol=1e-5,atol=1e-5)
