"""Regression tests for bounded-memory fallback when a single query overflows a halo."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator

try:
    from fvtools.prepro._buildcase import tiled_bathymetry as mod
except ImportError:  # standalone copy used by the isolated packager smoke tests
    import tiled_bathymetry as mod


def _data(seed=123, n=5500):
    rng = np.random.default_rng(seed)
    # All these points fit into the initial >=0.1-m halo of a single node.
    dense = rng.uniform(-0.009, 0.009, (n, 2))
    remote = np.array([[-50., -30.], [40., -40.], [80., 40.],
                       [-60., 50.], [55., 85.]])
    xy = np.vstack((dense, remote))
    z = np.sin(xy[:, 0]*9.) + np.cos(xy[:,1]*7.) + .03*xy[:,0]
    return np.c_[xy,z]


def _run(tmp_path, *, workers, max_points, queries, seed=123):
    src = _data(seed)
    # Windows does not allow overwriting a .npy file while a read-only
    # NumPy memmap still holds it open (possibly via worker processes).
    # Each invocation must therefore create a distinct source file.
    path = tmp_path / f'dense_workers_{workers}_cap_{max_points}_seed_{seed}.npy'
    np.save(path,src)
    kw = dict(max_triangulation_points=max_points, nodes_per_tile=2,
              bins_per_axis=9, source_chunk_rows=501)
    if 'workers' in mod.TilingOptions.__dataclass_fields__:
        kw['workers'] = workers
    settings = mod.TilingOptions(**kw)
    actual = mod.interpolate_unstructured_npy(path, queries, options=settings)
    reference = LinearNDInterpolator(src[:,:2],src[:,2])(queries)
    np.testing.assert_allclose(actual,reference,rtol=1e-9,atol=1e-9,equal_nan=True)
    return actual


@pytest.mark.parametrize('max_points',[64,256])
def test_one_dense_node_exceeds_cap_but_does_not_raise(tmp_path,max_points,capsys):
    result=_run(tmp_path,workers=1,max_points=max_points,queries=np.array([[0.,0.]]))
    assert np.isfinite(result).all()
    assert 'disk-streamed single-node interpolation' not in capsys.readouterr().out


def test_multiple_dense_nodes_are_exact_and_deterministic(tmp_path):
    queries = np.array([[0.,0.],[.001,.002],[-.003,.005],[.005,-.005]])
    one=_run(tmp_path,workers=1,max_points=150,queries=queries)
    two=_run(tmp_path,workers=2,max_points=150,queries=queries)
    np.testing.assert_array_equal(one,two)


def test_disk_nearest_is_exact_and_streaming(tmp_path):
    src=_data(44)
    directory=tmp_path/'idx';directory.mkdir()
    idx=mod._DiskBins(src,(-100.,-100.,100.,100.),directory,
                      mod.TilingOptions(source_chunk_rows=717,bins_per_axis=15))
    target=np.array([.0,.0])
    near=mod._nearest_on_disk(idx,target,25)
    expected_dist=np.sort(np.sum((src[:,:2]-target)**2,axis=1))[:25]
    np.testing.assert_allclose(np.sort(np.sum((near[:,:2]-target)**2,axis=1)),expected_dist)
    assert len(near)==25


def test_disk_interior_point_certification_checks_global_cloud(tmp_path):
    src=_data(83)
    directory=tmp_path/'idx';directory.mkdir()
    idx=mod._DiskBins(src,(-100.,-100.,100.,100.),directory,
                      mod.TilingOptions(source_chunk_rows=717,bins_per_axis=15))
    # Circle around (0,0) radius .01 includes much of the dense cloud.
    intruders=mod._intruders_on_disk(idx,np.array([0.,0.]),.01,max_points=5)
    assert len(intruders)==5
    assert np.all(np.sum(intruders[:,:2]**2,axis=1)<.01**2)
    empty=mod._intruders_on_disk(idx,np.array([25.,25.]),.001)
    assert len(empty)==0


def test_large_sparse_gap_with_many_dense_points_on_one_side(tmp_path):
    rng=np.random.default_rng(79)
    clustered=rng.uniform(-.04,.04,(5500,2))
    outer=np.array([[-80.,-80.],[-80.,80.],[80.,-80.],[80.,80.]])
    xy=np.r_[clustered,outer]
    z=xy[:,0]*.25+xy[:,1]*.75+np.cos(xy[:,0]/13.)
    queries=np.array([[10.,10.],[40.,10.],[-15.,30.],[.0,.0]])
    src=np.c_[xy,z]
    path=tmp_path/'gap.npy';np.save(path,src)
    opts = dict(max_triangulation_points=90, nodes_per_tile=1,bins_per_axis=11)
    if 'workers' in mod.TilingOptions.__dataclass_fields__:
        opts['workers']=1
    actual=mod.interpolate_unstructured_npy(path,queries,
        options=mod.TilingOptions(**opts))
    reference=LinearNDInterpolator(xy,z)(queries)
    np.testing.assert_allclose(actual,reference,rtol=1e-9,atol=1e-9,equal_nan=True)
