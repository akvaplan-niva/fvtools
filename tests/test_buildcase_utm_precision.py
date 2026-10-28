"""UTM-coordinate regression tests for memory-capped dense Delaunay fallback.

Baseline defect: the same tiny point cloud works at (0, 0), but the
uncentered circumcircle check fails with large easting/northing values.
"""
import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator
from fvtools.prepro._buildcase.tiled_bathymetry import (
    TilingOptions, _DiskBins, _interpolate_dense_single, interpolate_unstructured_npy
)


def _make_data(offset, seed=128, count=2_000):
    rng=np.random.default_rng(seed)
    offset=np.asarray(offset, dtype=np.float64)
    dense=rng.uniform(-0.009,0.009,(count,2)) + offset
    remote=np.array([[-50.,-30.],[40.,-40.],[80.,40.],[-60.,50.],[55.,85.]])+offset
    xy=np.vstack((dense,remote))
    centered=xy-offset
    z=3.+np.sin(centered[:,0]*17.)+np.cos(centered[:,1]*29.)+0.035*centered[:,0]
    return np.column_stack((xy,z))


@pytest.mark.parametrize('offset',[(0.,0.), (10_000.,10_000.),(500_000.,7_700_000.)])
@pytest.mark.parametrize('max_points',[64, 150])
def test_dense_single_normalizes_utm_before_delaunay(tmp_path, offset,max_points):
    xyz=_make_data(offset)
    xy=np.asarray(offset)
    bounds=(xy[0]-100,xy[1]-100,xy[0]+100,xy[1]+100)
    options=TilingOptions(max_triangulation_points=max_points,bins_per_axis=9,source_chunk_rows=1000,workers=1)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=tmp_path) as d:
        index=_DiskBins(xyz,bounds,d,options)
        result, refinements=_interpolate_dense_single(xy,index,options)
    expected=float(LinearNDInterpolator(xyz[:,:2]-xy,xyz[:,2])(np.zeros((1,2)))[0])
    np.testing.assert_allclose(result,expected,rtol=1e-10,atol=1e-10)
    assert refinements < max_points


@pytest.mark.parametrize('workers',[1,2])
def test_full_interpolator_matches_translation_invariant_reference(tmp_path,workers):
    offset=np.array([500_000.,7_700_000.])
    xyz=_make_data(offset,seed=112,count=5_500)
    queries=offset + np.array([[0.,0.],[.001,.002],[-.003,.005],[.005,-.005]])
    path=tmp_path/f'utm_dense_workers_{workers}.npy'
    np.save(path,xyz)
    options=TilingOptions(max_triangulation_points=150, source_chunk_rows=1001,
                          nodes_per_tile=2,bins_per_axis=9,workers=workers)
    got=interpolate_unstructured_npy(path,queries,options=options)
    reference=LinearNDInterpolator(xyz[:,:2]-offset,xyz[:,2])(queries-offset)
    np.testing.assert_allclose(got,reference,rtol=1e-9,atol=1e-9,equal_nan=True)
