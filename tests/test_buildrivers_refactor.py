"""BuildRivers regression tests: public API, numerical outputs and notebook editing."""
import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fvtools.prepro import BuildRivers as br
from fvtools.prepro._buildrivers import sources, interaction, forcing, constants


def test_classes_still_import_from_buildrivers():
    assert br.HBVRunoff is sources.HBVRunoff
    assert br.RiverPositions is sources.RiverPositions
    assert br.RiverTemperatures is sources.RiverTemperatures
    assert br.DraggablePoints is interaction.DraggablePoints
    assert br.CropRivers is interaction.CropRivers
    assert br.FVCOM_rivers is forcing.FVCOM_rivers
    assert issubclass(br.FVCOM_rivers, br.CropRivers)
    assert br.version == constants.version == forcing.version == 2.0
    assert asyncio.iscoroutinefunction(br.main)


def test_default_configuration_is_independent_between_calls():
    first = br.get_input('/tmp/rivers')
    second = br.get_input('/tmp/rivers')
    assert first is not second
    assert first['iloc'] == 'edge'
    assert first['river_projection'] == 'epsg:32633'
    assert first['runoff'] == '/tmp/rivers/Niva_2025_2018v20.05/'
    first['iloc'] = 'node'
    assert second['iloc'] == 'edge'


def test_riverpositions_retains_original_coordinates(tmp_path):
    csv = tmp_path / 'river_positions.csv'
    pd.DataFrame({
        'rivers': [10, 20], 'vassdragsomraade': [91, 92],
        'x_outlet': [500010., 500020.], 'y_outlet': [7700010., 7700020.],
        'areal': [10., 20.], 'elvenavn': ['Testelv', None],
    }).to_csv(csv, index=False)
    rivers = br.RiverPositions(csv, [91, 92], 'epsg:32633')
    np.testing.assert_array_equal(rivers.rivers.x_original, [500010., 500020.])
    rivers.x = np.array([500050., 500100.])
    np.testing.assert_array_equal(rivers.rivers.x_original, [500010., 500020.])
    rivers.crop([1])
    assert rivers.rivers.index.tolist() == [20]
    np.testing.assert_array_equal(rivers.x, [500100.])


def test_case_specific_csv_must_have_original_outlet_coordinates(tmp_path):
    csv = tmp_path / 'river_positions_case.csv'
    pd.DataFrame({'rivers': [10], 'vassdragsomraade': [91],
                  'x_outlet': [500010.], 'y_outlet': [7700010.]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match='x_original'):
        br.RiverPositions(csv, [91], 'epsg:32633')


def test_draggable_widget_keeps_originals_and_waits_for_done(monkeypatch):
    monkeypatch.setattr(interaction, 'display', lambda obj: None)
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    fig.canvas.draw()
    editor = br.DraggablePoints(ax, [3.], [4.], [1.], [2.])
    try:
        xy = ax.transData.transform([[3., 4.]])[0]
        editor.on_press(SimpleNamespace(inaxes=ax, x=float(xy[0]), y=float(xy[1])))
        editor.on_motion(SimpleNamespace(inaxes=ax, xdata=5., ydata=6.))
        editor.on_release(SimpleNamespace(inaxes=ax))
        assert editor.was_modified
        np.testing.assert_allclose(editor.get_data(), [[5.], [6.]])
        np.testing.assert_array_equal(editor.original_x, [1.])
        np.testing.assert_array_equal(editor.original_y, [2.])
        np.testing.assert_array_equal(editor._line_x[:3], [1., 5., np.nan])

        async def wait_and_confirm():
            task = asyncio.create_task(editor.wait())
            await asyncio.sleep(0)
            assert not task.done()
            editor.on_done(None)
            return await task
        result = asyncio.run(wait_and_confirm())
        np.testing.assert_allclose(result, [[5.], [6.]])
        assert editor.done_button.disabled
    finally:
        editor.disconnect()
        plt.close(fig)


def test_merge_rivers_preserves_flux_sum_and_weighted_temperature():
    river = br.FVCOM_rivers.__new__(br.FVCOM_rivers)
    river.info = {'iloc': 'node'}
    river.M = SimpleNamespace(x=np.array([10., 20., 30.]),
                              y=np.array([40., 50., 60.]))
    river.mesh_location = np.array([0, 0, 2])
    river.RiverTransport = np.array([[1., 2., 3.], [4., 5., 6.]])
    river.RiverTemp = np.array([[10., 12., 20.], [11., 13., 21.]])
    river.river_names = ['A', 'B', 'C']
    river.short_names = ['a', 'b', 'c']
    flux, temperature, names, ids = river.merge_rivers()
    np.testing.assert_array_equal(flux, [[3., 3.], [9., 6.]])
    np.testing.assert_allclose(temperature, [[34. / 3., 20.], [109. / 9., 21.]])
    assert names == ['A, B', 'C']
    assert ids == ['a, b', 'c']
    np.testing.assert_array_equal(river.xriv, [10., 30.])


def test_namelist_keeps_one_based_locations_and_vertical_weights(tmp_path):
    obj = br.FVCOM_rivers.__new__(br.FVCOM_rivers)
    obj.M = SimpleNamespace(siglev_c=np.array([[0., -.4, -1.], [0., -.2, -1.]]))
    obj.mesh_location = np.array([1])
    obj.river_names = ['Elv ø']
    output = tmp_path / 'RiverNamelist.nml'
    obj.write_namelist(str(output))
    contents = output.read_text()
    assert "RIVER_NAME = 'Elv o'" in contents
    assert 'RIVER_GRID_LOCATION = 2' in contents
    assert 'RIVER_VERTICAL_DISTRIBUTION = 0.20000 0.80000' in contents


def test_main_workflow_and_projection_order_are_preserved(monkeypatch):
    events = []
    class Grid:
        reference = 'epsg:32632'
        casename = 'synthetic'
        def __init__(self, path):
            events.append('grid')
        def re_project(self, projection):
            events.append(('project', projection))
        def get_cfl(self, verbose):
            assert verbose is False
            events.append('cfl')

    class Forcing:
        def __init__(self, info, mesh, vassdrag, start, stop):
            assert mesh.reference == 'epsg:32632'
            events.append('forcing')
        async def crop_rivers_far_from_land_and_close_to_OBC(self, positions):
            events.append('crop')
            return positions
        def redistribute_runoff(self, *args): events.append('runoff')
        def interpolate_forcing_to_model_time(self, *args): events.append('interpolate')
        def connect_to_mesh(self, mesh): events.append('connect')
        def dump(self): events.append('dump')
        def write_namelist(self): events.append('namelist')

    class HBV:
        def __init__(self, *args, **kwargs): events.append('hbv')

    class Positions:
        def __init__(self, *args): events.append('positions')

    class Temperature:
        def __init__(self, *args, **kwargs): events.append('temperature')
        def make_individual_river_climatology(self, positions):
            events.append('climatology')
            return object()
        def insert_observed_temperature(self, values, vassdrag):
            events.append('observed')
            return values

    monkeypatch.setattr(br, 'FVCOM_grid', Grid)
    monkeypatch.setattr(br, 'FVCOM_rivers', Forcing)
    monkeypatch.setattr(br, 'HBVRunoff', HBV)
    monkeypatch.setattr(br, 'RiverPositions', Positions)
    monkeypatch.setattr(br, 'RiverTemperatures', Temperature)
    monkeypatch.setattr(br, 'show_forcing', lambda *args: events.append('show'))
    conf = {'river_projection': 'epsg:32633', 'runoff': 'runoff',
            'riverpositions': 'positions', 'rivertemp': 'temperatures'}
    asyncio.run(br.main('2024-01-01', '2024-01-02', [91], 'M.npy', info=conf))
    assert conf['grid_projection'] == 'epsg:32632'
    assert events == [
        'grid', ('project', 'epsg:32633'), 'cfl', 'forcing',
        'hbv', 'positions', 'temperature', 'crop', 'runoff',
        'climatology', 'observed', 'interpolate', 'connect', 'dump',
        'namelist', ('project', 'epsg:32632'), 'show',
        ('project', 'epsg:32633'),
    ]


def test_interactive_map_keeps_georeference_and_compact_legend(monkeypatch):
    """The map must exist BEFORE georeference(), including with ipympl logic."""
    from shapely.geometry import box
    seen = []
    class Mesh:
        filepath = 'M.npy'
        casename = 'case'
        def __init__(self):
            self.Proj = lambda x, y, inverse=False: (np.asarray(x), np.asarray(y))
        def re_project(self, reference):
            seen.append(('projection', reference))
        def georeference(self):
            seen.append(('geo_axes', plt.gca()))
            plt.gca().plot([0, 10], [0, 10], color='blue', label='basemap reference')

    class Positions:
        reference = 'epsg:32633'
        def __init__(self):
            self.rivers = pd.DataFrame({
                'x_outlet': [5., 50.], 'y_outlet': [5., 50.],
                'x_original': [5., 50.], 'y_original': [5., 50.],
                'areal': [100000., 100000.],
            })
            self.Proj = lambda x, y, inverse=False: (np.asarray(x), np.asarray(y))
        @property
        def lon(self): return self.rivers.x_outlet.to_numpy()
        @property
        def lat(self): return self.rivers.y_outlet.to_numpy()
        @property
        def x(self): return self.rivers.x_outlet.to_numpy()
        @property
        def y(self): return self.rivers.y_outlet.to_numpy()
        def crop(self, indices):
            self.rivers = self.rivers.iloc[indices].copy()

    class Model(br.CropRivers):
        @property
        def buffered_domain(self): return box(0, 0, 10, 10)
        @property
        def x_land(self): return np.array([1., 9.])
        @property
        def y_land(self): return np.array([1., 9.])
        @property
        def obc_tree(self):
            return SimpleNamespace(query=lambda xy: (
                np.full(len(xy), 1000.), np.zeros(len(xy), dtype=int)))

    class FakeEditor:
        def __init__(self, ax, x, y, xo, yo):
            assert ax is seen[-1][1]  # georeferenced axes, not a replacement
            self.was_modified = False
            self.xy = (x.copy(), y.copy())
            self.legend = ax.legend_
            seen.append(('editor', ax))
        async def wait(self):
            return self.xy
        def disconnect(self):
            seen.append(('disconnect',))

    monkeypatch.setattr(interaction, 'DraggablePoints', FakeEditor)
    monkeypatch.setattr(interaction, 'plot_polygon', lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    forcing_obj = Model()
    forcing_obj.M = Mesh()
    forcing_obj.info = {'grid_projection': 'epsg:32632', 'obc buffer': 10}
    positions = Positions()
    result = asyncio.run(forcing_obj.crop_rivers_far_from_land_and_close_to_OBC(positions))
    assert result is positions
    assert len(result.rivers) == 1
    assert result.rivers.x_outlet.iloc[0] == 5.
    assert seen[0] == ('projection', 'epsg:32632')
    assert seen[1][0] == 'geo_axes'
    assert seen[2] == ('editor', seen[1][1])
    handles = seen[2][1].get_legend().legend_handles
    for handle in handles:
        if hasattr(handle, 'get_sizes'):
            np.testing.assert_array_equal(handle.get_sizes(), [24.])
    assert ('disconnect',) in seen
    assert seen[-1] == ('projection', 'epsg:32633')


def test_buildrivers_does_not_globally_suppress_warnings():
    import inspect
    assert 'warnings.filterwarnings("ignore")' not in inspect.getsource(br)
