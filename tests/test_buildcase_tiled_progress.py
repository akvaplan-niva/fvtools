"""Progress-only regressions for the tiled unstructured bathymetry reader."""

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from fvtools.prepro._buildcase import tiled_bathymetry as tiled


def test_progress_for_disk_index_and_grid_nodes(tmp_path, capsys):
    rng = np.random.default_rng(139)
    points = rng.uniform(-100., 100., (2200, 2))
    depths = 20. + .01 * points[:, 0]**2 + points[:, 1] * .1
    queries = rng.uniform(-80., 80., (230, 2))
    filename = tmp_path / 'source.npy'
    np.save(filename, np.column_stack((points, depths)))
    opts = tiled.TilingOptions(source_chunk_rows=127, bins_per_axis=17,
                              nodes_per_tile=17, max_triangulation_points=6000)
    result = tiled.interpolate_unstructured_npy(filename, queries, options=opts)
    reference = LinearNDInterpolator(points, depths)(queries)
    np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10,
                               equal_nan=True)
    out = capsys.readouterr().out
    assert 'Index pass 1/2' in out and 'Index pass 2/2' in out
    assert 'Interpolating 230 grid nodes' in out
    assert 'Interpolation of grid nodes: 0.0% (0/230)' in out
    assert 'Interpolation of grid nodes: 100.0% (230/230)' in out
    assert 'Finished:' in out
    assert 'local Delaunay triangulations' in out
    assert 'elapsed' in out


def test_small_mesh_and_large_chunks_reach_100_percent(tmp_path, capsys):
    xy = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.], [5., 5.]])
    xyz = np.column_stack((xy, np.array([0., 10., 20., 30., 15.])))
    filename = tmp_path / 'source.npy'
    np.save(filename, xyz)
    got = tiled.interpolate_unstructured_npy(filename, xy)
    np.testing.assert_array_equal(got, xyz[:, 2])
    out = capsys.readouterr().out
    assert 'Index pass 1/2' in out
    assert 'Index pass 1/2 – counting source points: 100.0% (5/5)' in out
    assert 'Index pass 2/2 – writing disk index: 100.0% (5/5)' in out
    assert 'Interpolation of grid nodes: 100.0% (5/5)' in out


def test_progress_works_when_halo_expands_or_batches_split(tmp_path, capsys):
    rng = np.random.default_rng(22)
    source_xy = rng.uniform(0, 500, (10_000, 2))
    source = np.c_[source_xy, np.sin(source_xy[:,0]/20) + source_xy[:,1]/50]
    q = rng.uniform(50, 450, (140, 2))
    filename = tmp_path / 'source.npy'
    np.save(filename, source)
    options = tiled.TilingOptions(source_chunk_rows=900, bins_per_axis=22,
                                  nodes_per_tile=18, max_triangulation_points=2000)
    got = tiled.interpolate_unstructured_npy(filename, q, options=options)
    expected = LinearNDInterpolator(source_xy, source[:,2])(q)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)
    out = capsys.readouterr().out
    # The percent must be based on completed nodes, not on the number of
    # recursively split batches (which cannot be known beforehand).
    assert 'Interpolation of grid nodes: 100.0% (140/140)' in out
    assert 'Batch 1:' in out
