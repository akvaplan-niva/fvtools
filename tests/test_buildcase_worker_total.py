"""Regression: disk-index worker state must retain total source-point count."""

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from fvtools.setup._buildcase import tiled_bathymetry as mod


def test_spawned_worker_index_has_total_and_can_interpolate_dense_node(tmp_path, monkeypatch):
    rng = np.random.default_rng(211)
    coords = rng.uniform(-50, 50, (150, 2))
    coords = np.concatenate((coords, [[-60, -60], [-60, 60], [60, -60], [60, 60]]))
    z = 4.0 + 0.2 * coords[:, 0] - 0.4 * coords[:, 1]
    source = np.column_stack((coords, z))
    options = mod.TilingOptions(bins_per_axis=8, max_triangulation_points=100,
                                source_chunk_rows=50, workers=1)
    idx = mod._DiskBins(source, (-100., -100., 100., 100.), tmp_path, options)
    mesh_file = tmp_path / 'worker_mesh.npy'
    np.save(mesh_file, np.array([[0., 0.]], dtype=np.float64))
    index_info = {
        'path': str(idx.path), 'offsets': idx.offsets, 'bounds': idx.bounds,
        'count': idx.count, 'record_bytes': idx.record_bytes,
        'dtype': str(idx.dtype), 'hull_vertices': idx.hull_vertices,
    }

    monkeypatch.setattr(mod, '_WORKER_STATE', None)
    mod._init_worker(index_info, str(mesh_file), options,
                     (-100., -100., 100., 100.), (-60., -60., 60., 60.))
    worker_idx, worker_mesh, worker_options, _, _ = mod._WORKER_STATE
    assert worker_idx.total == idx.total == int(idx.offsets[-1]) == len(source)
    np.testing.assert_array_equal(worker_mesh, [[0., 0.]])

    value, attempts = mod._interpolate_dense_single(
        np.array([0., 0.]), worker_idx, worker_options)
    expected = float(LinearNDInterpolator(coords, z)(np.array([[0., 0.]]))[0])
    np.testing.assert_allclose(value, expected, rtol=0, atol=1e-9)
    assert attempts >= 1
