"""Regression tests for flux-conservative river merging and splitting.

Run using: uv run python -W error::RuntimeWarning -m pytest tests -v -rs
These tests deliberately avoid HBV files, Jupyter and network map services.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from fvtools.prepro._buildrivers import forcing


def make_rivers(positions, flux, temperature, mesh_locations=None,
                threshold=0.5, iloc='node', land_count=12):
    obj = forcing.FVCOM_rivers.__new__(forcing.FVCOM_rivers)
    coords = np.asarray(positions, dtype=float)
    obj.info = {
        'iloc': iloc, 'minrcoef': threshold, 'tideamp': 0.,
        'min_depth': 1., 'Isplit': 1.,
        'river_projection': 'epsg:32633', 'grid_projection': 'epsg:32633',
    }
    obj.xriv, obj.yriv = coords.copy(), np.zeros(len(coords))
    obj.RiverTransport = np.array(flux, dtype=float)
    obj.RiverTemp = np.array(temperature, dtype=float)
    obj.mesh_location = (np.arange(len(coords)) if mesh_locations is None
                         else np.array(mesh_locations))
    obj.river_names = [f'R{i}' for i in range(len(coords))]
    obj.short_names = list(np.arange(100, 100 + len(coords)))
    xy = np.arange(land_count, dtype=float)
    obj.M = SimpleNamespace(
        x=xy, y=np.zeros(land_count), xc=xy, yc=np.zeros(land_count),
        h=np.ones(land_count), hc=np.ones(land_count),
        ts=np.array([1.]), tri_area=np.ones(land_count),
        find_nearest=lambda x, y, loc: np.asarray(x, dtype=int),
    )
    return obj


def set_land_trees(monkeypatch, obj, mapping=None):
    xy = np.column_stack((obj.M.x, obj.M.y))
    land = cKDTree(xy)
    if mapping is None:
        mesh = cKDTree(xy)
    else:
        class MappedTree:
            def query(self, points, **kwargs):
                # Every land point ends up in the same destination.
                arr = np.atleast_2d(points)
                return np.zeros(len(arr)), np.full(len(arr), mapping)
        mesh = MappedTree()
    monkeypatch.setattr(type(obj), 'land_tree', property(lambda self: land))
    monkeypatch.setattr(type(obj), 'mesh_tree', property(lambda self: mesh))
    monkeypatch.setattr(type(obj), 'x_land', property(lambda self: obj.M.x))
    monkeypatch.setattr(type(obj), 'y_land', property(lambda self: obj.M.y))


def test_merge_conserves_each_time_step_and_weights_temperature():
    r = make_rivers([0, 0, 2],
                    [[1, 3, 0], [0, 0, 2]],
                    [[10, 20, 30], [11, 21, 31]],
                    mesh_locations=[0, 0, 2], land_count=3)
    q0 = r.RiverTransport.copy()
    q, t, names, ids = r.merge_rivers()
    np.testing.assert_array_equal(q, [[4, 0], [0, 2]])
    np.testing.assert_allclose(t, [[17.5, 30], [16., 31.]])
    np.testing.assert_allclose(q.sum(axis=1), q0.sum(axis=1))
    assert names == ['R0, R1', 'R2']
    assert ids == ['100, 101', '102']  # Numeric river IDs are supported.
    np.testing.assert_array_equal(r.xriv, [0, 2])


def test_merge_one_river_preserves_all_fields():
    r = make_rivers([2], [[2.], [0.]], [[4.], [9.]], land_count=4)
    q, t, names, ids = r.merge_rivers()
    np.testing.assert_array_equal(q, [[2.], [0.]])
    np.testing.assert_array_equal(t, [[4.], [9.]])
    assert names == ['R0']
    assert ids == ['100']


@pytest.mark.parametrize('bad_q', [np.nan, np.inf, -0.25])
def test_merge_rejects_invalid_flux(bad_q):
    r = make_rivers([0], [[bad_q]], [[12.]])
    with pytest.raises(ValueError, match='finite and nonnegative'):
        r.merge_rivers()


def test_merge_rejects_non_finite_temperature():
    r = make_rivers([0], [[1]], [[np.nan]])
    with pytest.raises(ValueError, match='finite temperatures'):
        r.merge_rivers()


def test_split_multiple_bad_indices_preserves_totals_and_temperature(monkeypatch):
    r = make_rivers([0, 6, 11], [[1.4, .1, 1.3], [2., .2, 1.8]],
                    [[5, 9, 13], [6, 10, 14]])
    set_land_trees(monkeypatch, r)
    before = r.RiverTransport.sum(axis=1)
    r.split_problematic_river(np.array([0, 2]), np.array([1.2, .1, 1.1]))
    np.testing.assert_allclose(r.RiverTransport.sum(axis=1), before, rtol=1e-14)
    assert len(r.xriv) == len(r.mesh_location) == r.RiverTemp.shape[1] == r.RiverTransport.shape[1] == 9
    assert len(r.river_names) == len(r.short_names) == 9
    assert r.river_names[4] == 'R1'  # Middle unsplit column retained.
    np.testing.assert_array_equal(r.RiverTemp[0], [5]*4 + [9] + [13]*4)
    assert len(set(r.xriv[:4])) == len(set(r.mesh_location[:4])) == 4
    assert len(set(r.xriv[5:])) == len(set(r.mesh_location[5:])) == 4
    assert r.river_names[0] == 'R0-p0'
    assert r.river_names[-1] == 'R2-p3'


def test_split_bad_index_zero_is_not_silently_ignored(monkeypatch):
    r = make_rivers([0, 11], [[1.2, .1]], [[8., 16.]])
    set_land_trees(monkeypatch, r)
    r.split_problematic_river(np.array([0]), np.array([1.2, .1]))
    assert r.RiverTransport.shape[1] == 5
    np.testing.assert_allclose(r.RiverTransport.sum(axis=1), [1.3])
    assert r.river_names[-1] == 'R1'


def test_split_is_atomic_when_later_river_has_too_few_land_points(monkeypatch):
    r = make_rivers([0, 3], [[.9, 2.]], [[10, 20]], land_count=4)
    set_land_trees(monkeypatch, r)
    original = (r.RiverTransport.copy(), r.RiverTemp.copy(), r.xriv.copy(),
                r.yriv.copy(), list(r.river_names), list(r.short_names))
    with pytest.raises(ValueError, match='only 4 land locations exist'):
        r.split_problematic_river([0, 1], [0.9, 2.0])
    for a, b in zip((r.RiverTransport, r.RiverTemp, r.xriv, r.yriv), original[:4]):
        np.testing.assert_array_equal(a, b)
    assert r.river_names == original[4]
    assert r.short_names == original[5]


def test_split_rejects_duplicates_that_map_to_one_mesh_location(monkeypatch):
    r = make_rivers([0], [[1.]], [[8.]], land_count=8)
    set_land_trees(monkeypatch, r, mapping=0)
    with pytest.raises(ValueError, match='distinct mesh outlets required'):
        r.split_problematic_river([0], [1.])


@pytest.mark.parametrize('iloc', ['node', 'edge'])
def test_stability_is_finite_and_uses_correct_depth_grid(iloc):
    r = make_rivers([0, 2], [[.2, .4], [.5, .6]],
                    [[10, 11], [12, 13]], mesh_locations=[0, 2], iloc=iloc,
                    land_count=3)
    r.unique_mesh = np.array([0, 2])
    r.M.h = np.array([2., 1., 2.])
    r.M.hc = np.array([4., 4., 4.])
    r.M.tri_area = np.array([2., 2., 2.])
    r.M.ts = np.array([2.])
    r.info['Isplit'] = 2.
    expected_depth = 2. if iloc == 'node' else 4.
    np.testing.assert_allclose(r.river_stability(),
                               [4*.5/(expected_depth*2), 4*.6/(expected_depth*2)])


def test_stability_depth_floor_and_non_positive_area_error():
    r = make_rivers([0], [[1.]], [[7.]])
    r.unique_mesh = np.array([0])
    r.info['tideamp'] = 2.
    r.info['min_depth'] = 3.
    np.testing.assert_allclose(r.river_stability(), [1/3])
    r.M.tri_area[0] = 0.
    with pytest.raises(ValueError, match='Invalid cell area'):
        r.river_stability()


def test_connect_to_mesh_handles_only_bad_index_zero_and_terminates(monkeypatch):
    r = make_rivers([0, 11], [[1.2, .1], [1.2, .1]],
                    [[8., 12.], [8., 12.]])
    set_land_trees(monkeypatch, r)
    monkeypatch.setattr(type(r), 'river_connection', lambda *args: None)
    monkeypatch.setattr(forcing.plt, 'scatter', lambda *a, **kw: None)
    monkeypatch.setattr(forcing.plt, 'legend', lambda *a, **kw: None)
    monkeypatch.setattr(forcing.plt, 'show', lambda *a, **kw: None)
    # Force a projection-independent mocked plotting call.
    monkeypatch.setattr(forcing.ccrs, 'epsg', lambda epsg: None)
    r.connect_to_mesh(r.M)
    np.testing.assert_allclose(r.RiverTransport.sum(axis=1), [1.3, 1.3])
    assert r.RiverTransport.shape[1] >= 4
    assert np.all(r.river_stability() <= r.info['minrcoef'])
    assert len(r.mesh_location) == len(r.river_names)


def test_connect_to_mesh_merges_before_checking_stability(monkeypatch):
    r = make_rivers([0, 0], [[.2, .2]], [[0., 20.]],
                    mesh_locations=[0, 0])
    set_land_trees(monkeypatch, r)
    monkeypatch.setattr(type(r), 'river_connection', lambda *args: None)
    monkeypatch.setattr(forcing.plt, 'scatter', lambda *a, **kw: None)
    monkeypatch.setattr(forcing.plt, 'legend', lambda *a, **kw: None)
    monkeypatch.setattr(forcing.plt, 'show', lambda *a, **kw: None)
    monkeypatch.setattr(forcing.ccrs, 'epsg', lambda epsg: None)
    r.connect_to_mesh(r.M)
    np.testing.assert_allclose(r.RiverTransport, [[.4]])
    np.testing.assert_allclose(r.RiverTemp, [[10.]])
    np.testing.assert_array_equal(r.mesh_location, [0])


def test_connect_to_mesh_reports_impossible_distribution(monkeypatch):
    r = make_rivers([0], [[3.]], [[8.]], land_count=3)
    set_land_trees(monkeypatch, r)
    monkeypatch.setattr(type(r), 'river_connection', lambda *args: None)
    with pytest.raises(ValueError, match='only 3 land locations'):
        r.connect_to_mesh(r.M)


def test_connect_to_mesh_splits_two_bad_rivers_without_index_shift(monkeypatch):
    r = make_rivers([0, 11], [[1.1, 1.2], [1.8, 1.6]],
                    [[5., 15.], [6., 16.]])
    set_land_trees(monkeypatch, r)
    monkeypatch.setattr(type(r), 'river_connection', lambda *args: None)
    monkeypatch.setattr(forcing.plt, 'scatter', lambda *a, **kw: None)
    monkeypatch.setattr(forcing.plt, 'legend', lambda *a, **kw: None)
    monkeypatch.setattr(forcing.plt, 'show', lambda *a, **kw: None)
    monkeypatch.setattr(forcing.ccrs, 'epsg', lambda epsg: None)
    r.connect_to_mesh(r.M)
    np.testing.assert_allclose(r.RiverTransport.sum(axis=1), [2.3, 3.4])
    assert len(r.mesh_location) == len(r.river_names) == r.RiverTemp.shape[1]
    assert np.all(r.river_stability() <= .5)
