"""Verified-simplex reuse for difficult dense unstructured bathymetry queries."""
import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator
try:
    from fvtools.prepro._buildcase import tiled_bathymetry as tiled
except ImportError:
    import tiled_bathymetry as tiled


def _gap_cloud(shift):
    rng = np.random.default_rng(45)
    left = rng.normal(-40, 1, (20_000, 2))
    right = rng.normal(40, 1, (20_000, 2))
    outside = np.array([[-200, -200], [200, -200], [-200, 200], [200, 200]])
    xy = np.vstack((left, right, outside))
    z = np.sin(xy[:, 0] / 10) + 0.2 * xy[:, 1]
    queries = rng.uniform(-30, 30, (50, 2))
    shift = np.asarray(shift, dtype=float)
    return np.column_stack((xy + shift, z)), queries + shift


@pytest.mark.parametrize('shift', [(0., 0.), (500_000., 7_700_000.)])
def test_cached_global_delaunay_triangles_reduce_disk_fallback(tmp_path, monkeypatch, shift):
    source, queries = _gap_cloud(shift)
    path = tmp_path / 'gap.npy'
    np.save(path, source)
    calls = []
    original = tiled._interpolate_dense_single

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(tiled, '_interpolate_dense_single', counted)
    result = tiled.interpolate_unstructured_npy(
        path, queries,
        options=tiled.TilingOptions(workers=1, nodes_per_tile=128,
                                    bins_per_axis=32,
                                    max_triangulation_points=3000))
    # Keep geometry local for the reference Delaunay, too.
    reference = LinearNDInterpolator(source[:, :2] - shift, source[:, 2])(
        queries - shift)
    np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-9)
    assert len(calls) < 25, f'Only {len(queries)-len(calls)} queries reused certified triangles'


def test_reuse_is_only_allowed_inside_a_validated_triangle():
    triangle = np.array([[500_000., 7_700_000., 3.],
                         [500_010., 7_700_000., 5.],
                         [500_000., 7_700_010., 7.]])
    targets = np.array([[500_001., 7_700_001.],
                        [500_003., 7_700_003.],
                        [500_011., 7_700_001.],
                        [500_001., 7_700_011.]])
    mask, values = tiled._cached_simplex_values(triangle, targets)
    np.testing.assert_array_equal(mask, [True, True, False, False])
    np.testing.assert_allclose(values[:2], [3.6, 4.8], rtol=0, atol=1e-12)
    assert np.isnan(values[2:]).all()


def test_parallel_and_serial_same_on_gap_queries(tmp_path):
    source, queries = _gap_cloud((500_000., 7_700_000.))
    path = tmp_path / 'gap_parallel.npy'
    np.save(path, source)
    kwargs = dict(nodes_per_tile=15, max_triangulation_points=3000, bins_per_axis=32)
    single = tiled.interpolate_unstructured_npy(
        path, queries, options=tiled.TilingOptions(workers=1, **kwargs))
    multi = tiled.interpolate_unstructured_npy(
        path, queries, options=tiled.TilingOptions(workers=2, **kwargs))
    np.testing.assert_allclose(multi, single, rtol=1e-10, atol=1e-10,
                               equal_nan=True)
