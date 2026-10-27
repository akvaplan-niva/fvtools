"""Regression tests for process-parallel tiled bathymetry interpolation.

Serial and spawned Windows/Linux workers use the exact same numerical kernel.
"""

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator
from fvtools.prepro._buildcase import tiled_bathymetry as mod


def _scene(seed=17, count=5000, nq=320):
    rng=np.random.default_rng(seed)
    points=rng.uniform(-200,200,(count,2))
    z=(np.sin(points[:,0]/21)+np.cos(points[:,1]/18)+.005*points[:,0]*points[:,1])
    # Add clustered data and sparse edge queries to stress adaptive splitting.
    queries=np.vstack((rng.uniform(-150,150,(nq-6,2)),
                       [[-190.,-190.],[-160.,180.],[10.,22.],[130.,-140.],[195.,195.],[150.,150.]]))
    return np.column_stack((points,z)), queries


@pytest.mark.parametrize('workers',[1,2,3])
def test_processes_match_serial_and_global_scipy(tmp_path, workers):
    data,xy=_scene(count=7000,nq=205)
    np.save(tmp_path/'bath.npy',data)
    opts=mod.TilingOptions(workers=workers,source_chunk_rows=401,
                           nodes_per_tile=35,bins_per_axis=11,
                           max_triangulation_points=4500)
    got=mod.interpolate_unstructured_npy(tmp_path/'bath.npy',xy,options=opts)
    expected=LinearNDInterpolator(data[:,:2],data[:,2])(xy)
    np.testing.assert_allclose(got,expected,rtol=2e-10,atol=2e-10,equal_nan=True)


def test_workers_read_same_disk_index_and_are_deterministic(tmp_path):
    data,xy=_scene(seed=28,count=15500,nq=360)
    np.save(tmp_path/'bath.npy',data.astype('f4'))
    kw=dict(source_chunk_rows=1500,nodes_per_tile=31,bins_per_axis=15,
            max_triangulation_points=1200)
    a=mod.interpolate_unstructured_npy(tmp_path/'bath.npy',xy,options=mod.TilingOptions(workers=1,**kw))
    b=mod.interpolate_unstructured_npy(tmp_path/'bath.npy',xy,options=mod.TilingOptions(workers=2,**kw))
    np.testing.assert_array_equal(a,b) # Every task has the identical decomposition and kernel.
    expected=LinearNDInterpolator(data.astype('f4')[:,:2],data.astype('f4')[:,2])(xy)
    np.testing.assert_allclose(a,expected,rtol=2e-5,atol=2e-5,equal_nan=True)


def test_worker_count_environment_and_limits(monkeypatch):
    opts=mod.TilingOptions()
    monkeypatch.setenv('FVTOOLS_BATHY_WORKERS','1')
    assert mod._num_workers(opts,11)==1
    monkeypatch.setenv('FVTOOLS_BATHY_WORKERS','4')
    assert mod._num_workers(opts,11)==min(4,mod._cpu_slots())
    assert mod._num_workers(mod.TilingOptions(workers=1),11)==1
    with pytest.raises(ValueError,match='at least 1'):
        mod._num_workers(mod.TilingOptions(workers=0),3)
    with pytest.raises(ValueError):
        mod._num_workers(mod.TilingOptions(workers=-1),3)


def test_partitions_disjoint_and_cover_all_nodes():
    rng=np.random.default_rng(31)
    xy=rng.uniform(size=(303,2))
    nodes=np.arange(len(xy),dtype=np.int64)
    parts=list(mod._partition_nodes(xy,nodes,41))
    assert len(parts)>2
    assert all(0<len(p)<=41 for p in parts)
    assert sorted(np.concatenate(parts).tolist())==nodes.tolist()


def test_workers_use_memmap_and_small_index_metadata(tmp_path):
    data,xy=_scene(count=900,nq=45)
    np.save(tmp_path/'bathy.npy',data)
    indexdir=tmp_path/'temp';indexdir.mkdir()
    idx=mod._DiskBins(np.load(tmp_path/'bathy.npy',mmap_mode='r'),(-5250.,-5250.,5250.,5250.),indexdir,mod.TilingOptions())
    path=tmp_path/'nodes.npy';np.save(path,xy)
    config=dict(path=str(idx.path), offsets=idx.offsets, bounds=idx.bounds, count=idx.count,
                record_bytes=idx.record_bytes, dtype=str(idx.dtype))
    mod._init_worker(config,str(path),mod.TilingOptions(workers=1),idx.bounds,
                     (xy[:,0].min(),xy[:,1].min(),xy[:,0].max(),xy[:,1].max()))
    shared = mod._WORKER_STATE[0]
    assert isinstance(mod._WORKER_STATE[1],np.memmap)
    assert not hasattr(shared,'data')
    assert shared.path==idx.path
    assert 'data' not in config
