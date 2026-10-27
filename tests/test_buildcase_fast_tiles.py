"""Fast, certified sparse Delaunay regression suite for BuildCase bathymetry.

Pure synthetic fixtures; no real FVCOM model input needed. Both the
triangulations and final depth arrays are checked against SciPy global Delaunay.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator

from fvtools.prepro._buildcase import tiled_bathymetry as tiled


def _compare(tmp_path, data, queries, *, config=None, rtol=1e-10, atol=1e-10):
    path = tmp_path / 'bathymetry.npy'
    np.save(path, data)
    opts = config or tiled.TilingOptions(
        source_chunk_rows=1000, max_triangulation_points=30_000,
        nodes_per_tile=350, bins_per_axis=17)
    actual = tiled.interpolate_unstructured_npy(path, queries, options=opts)
    lo = queries.min(axis=0)-5000
    hi = queries.max(axis=0)+5000
    inside = np.all((data[:, :2] >= lo) & (data[:, :2] <= hi), axis=1)
    reference = LinearNDInterpolator(data[inside, :2], data[inside, 2])(queries)
    np.testing.assert_allclose(actual, reference, rtol=rtol, atol=atol, equal_nan=True)
    np.testing.assert_array_equal(np.load(path), data)
    return actual


def test_certified_sparse_interpolation_random_points(tmp_path):
    rng = np.random.default_rng(123)
    xy = rng.uniform(0, 500, (11_000,2))
    z = np.sin(xy[:,0]/17) + np.cos(xy[:,1]/41) + xy[:,0]**2/10000
    q = rng.uniform(20,480,(780,2))
    _compare(tmp_path,np.column_stack((xy,z)),q)


def test_high_source_density_matches_global_delaunay_with_fewer_triangles(tmp_path,monkeypatch):
    rng = np.random.default_rng(78)
    xy = rng.uniform(0,400,(35_000,2)).astype('f4')
    z = (np.sin(xy[:,0]/40)+np.cos(xy[:,1]/80)).astype('f4')
    q = rng.uniform(25,375,(200,2))
    # This case exercises k-nearest expansion beyond the initial k=16.
    opts=tiled.TilingOptions(source_chunk_rows=800,bins_per_axis=29,
                            nodes_per_tile=25,max_triangulation_points=2400,workers=1)
    orig=tiled.Delaunay
    calls=[]
    def counted(x):
        calls.append(len(x))
        assert len(x)<=opts.max_triangulation_points
        return orig(x)
    monkeypatch.setattr(tiled,'Delaunay',counted)
    got=_compare(tmp_path,np.c_[xy,z],q,config=opts,rtol=1e-5,atol=1e-6)
    assert np.isfinite(got).all()
    assert calls and max(calls)<=2400


def test_skips_points_outside_global_convex_hull_without_expanding_to_full_domain(tmp_path):
    rng = np.random.default_rng(222)
    # A triangular source domain with a substantial part of bounding box empty.
    w = rng.dirichlet([1,1,1], size=24_000)
    points = w @ np.array([[0.,0.],[100.,0.],[0.,100.]])
    depth = np.sin(points[:,0]/10)+np.cos(points[:,1]/10)
    query = np.r_[rng.uniform(10,60,(180,2)),
                  [[90.,90.],[80.,90.],[90.,80.],[75.,75.]]]
    opts=tiled.TilingOptions(source_chunk_rows=2500,bins_per_axis=16,
                            max_triangulation_points=1200,nodes_per_tile=150)
    got=_compare(tmp_path,np.c_[points,depth],query,config=opts)
    assert np.isnan(got[-4:]).all()


def test_sparse_interpolation_f32_and_precise_mesh_vertices(tmp_path):
    rng=np.random.default_rng(9)
    xy=rng.uniform(100000, 101000, (3000,2)).astype('f4')
    z=(np.sin(xy[:,0]/17)+np.cos(xy[:,1]/23)).astype('f4')
    q=np.r_[xy[[0,12,120]].astype('f8'),rng.uniform(100100,100900,(150,2))]
    got=_compare(tmp_path,np.c_[xy,z].astype('f4'),q,rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(got[:3],z[[0,12,120]],atol=1e-9,rtol=1e-9)


def test_large_target_group_reuses_local_source_triangulations(tmp_path,monkeypatch):
    rng=np.random.default_rng(2026)
    xy=rng.uniform(0,300,(40_000,2))
    z=np.sin(xy[:,0]*0.02)+np.cos(xy[:,1]*0.04)
    q=rng.uniform(12,288,(2500,2))
    opts=tiled.TilingOptions(nodes_per_tile=4096,bins_per_axis=32,
                            max_triangulation_points=50_000,source_chunk_rows=10_000,workers=1)
    orig=tiled.Delaunay
    counts=[]
    def counted(x):
        counts.append(len(x))
        return orig(x)
    monkeypatch.setattr(tiled,'Delaunay',counted)
    _compare(tmp_path,np.c_[xy,z],q,config=opts)
    assert len(counts)<70, f'Too many redundant Delaunay triangulations: {len(counts)}'
    assert max(counts)<=opts.max_triangulation_points


def test_progress_shows_selected_sources_and_completion(tmp_path,capsys):
    rng=np.random.default_rng(23)
    xy=rng.uniform(0,100,(3000,2));q=rng.uniform(10,90,(400,2))
    _compare(tmp_path,np.c_[xy,xy[:,0]*.3+xy[:,1]*.6],q)
    out=capsys.readouterr().out
    assert 'triangulating' in out and 'selected points' in out
    assert 'Interpolation of grid nodes: 100.0%' in out
    assert 'Finished:' in out


def test_default_group_size_is_larger_than_prior_128():
    assert tiled.TilingOptions().nodes_per_tile == 4096
