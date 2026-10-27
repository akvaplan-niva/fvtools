"""Regression tests for BuildRivers output organization.

Default FVCOM schema, paths, public names and plotting remain supported;
configurable output destinations are opt-in.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.prepro import BuildRivers as br
from fvtools.prepro._buildrivers import forcing, output


def make_forcing():
    rivers = br.FVCOM_rivers.__new__(br.FVCOM_rivers)
    rivers.xriv = np.array([1.0, 2.0])
    rivers.yriv = np.array([2.0, 3.0])
    rivers.model_time = np.array([59000.0, 59000.125], dtype=float)
    rivers.RiverTransport = np.array([[1.25, 2.5], [3.25, 4.5]])
    rivers.RiverTemp = np.array([[5.0, 7.5], [6.0, 8.5]])
    rivers.river_names = ['Elv ø', 'Elv æ']
    rivers.mesh_location = np.array([0, 1])
    rivers.M = SimpleNamespace(siglev_c=np.array([[0., -.4, -1.],
                                                 [0., -.2, -1.]]))
    return rivers


def test_public_apis_remain_available():
    assert br.FVCOM_rivers is forcing.FVCOM_rivers
    assert br.show_forcing is output.show_forcing
    assert issubclass(br.FVCOM_rivers, output.RiverOutputMixin)
    assert issubclass(br.FVCOM_rivers, br.CropRivers)
    obj = make_forcing()
    assert callable(obj.dump)
    assert callable(obj.write_namelist)
    assert obj.fix_nordic('Æ Ø Å æ ø å') == 'E O A e o a'


def test_default_writer_uses_backward_compatible_no_argument_calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    obj = make_forcing()
    calls = []
    obj.dump = lambda: calls.append(('dump',))
    obj.write_namelist = lambda: calls.append(('namelist',))
    files = output.write_forcing_files(obj)
    assert calls == [('dump',), ('namelist',)]
    assert files == (Path('input/riverdata.nc'), Path('input/RiverNamelist.nml'))
    assert (tmp_path / 'input').is_dir()


def test_custom_directory_passes_matching_paths_and_basename(tmp_path):
    obj = make_forcing()
    calls = []
    obj.dump = lambda path: calls.append(('dump', Path(path)))
    obj.write_namelist = lambda path: calls.append(('namelist', Path(path)))
    folder = tmp_path / 'nested' / 'forcing'
    files = output.write_forcing_files(obj, folder)
    assert folder.is_dir()
    assert calls == [('dump', folder / 'riverdata.nc'),
                     ('namelist', folder / 'RiverNamelist.nml')]
    assert files == (folder / 'riverdata.nc', folder / 'RiverNamelist.nml')


def test_custom_directory_failure_does_not_call_writers(tmp_path):
    blocker = tmp_path / 'occupied'
    blocker.write_text('file, not a directory')
    obj = make_forcing()
    obj.dump = lambda *_: pytest.fail('Should never reach writer')
    with pytest.raises((OSError, FileExistsError, NotADirectoryError)):
        output.write_forcing_files(obj, blocker)


def test_namelist_output_is_identical_for_default_and_custom_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    obj = make_forcing()
    (tmp_path / 'input').mkdir()
    obj.write_namelist()
    default_bytes = (tmp_path / 'input' / 'RiverNamelist.nml').read_bytes()
    custom = tmp_path / 'custom' / 'RiverNamelist.nml'
    custom.parent.mkdir()
    obj.write_namelist(custom)
    assert custom.read_bytes() == default_bytes
    content = default_bytes.decode('utf-8')
    assert content.count('&NML_RIVER') == 2
    assert "RIVER_FILE = 'riverdata.nc'" in content
    assert "RIVER_NAME = 'Elv o'" in content
    assert "RIVER_NAME = 'Elv e'" in content
    assert 'RIVER_GRID_LOCATION = 1' in content
    assert 'RIVER_GRID_LOCATION = 2' in content
    assert 'RIVER_VERTICAL_DISTRIBUTION = 0.40000 0.60000' in content


def test_netcdf_schema_and_values_remain_unchanged(tmp_path):
    netcdf4 = pytest.importorskip('netCDF4')
    obj = make_forcing()
    path = tmp_path / 'riverdata.nc'
    obj.dump(path)
    with netcdf4.Dataset(path) as ds:
        assert set(ds.dimensions) == {'time', 'rivers', 'DateStrLen', 'namelen'}
        assert ds.dimensions['time'].isunlimited()
        assert len(ds.dimensions['rivers']) == 2
        assert len(ds.dimensions['namelen']) == 80
        assert set(ds.variables) == {'time', 'Itime', 'Itime2', 'river_flux',
                                     'river_temp', 'river_salt', 'river_names'}
        np.testing.assert_allclose(ds['river_flux'][:], obj.RiverTransport)
        np.testing.assert_allclose(ds['river_temp'][:], obj.RiverTemp)
        np.testing.assert_allclose(ds['river_salt'][:], 0)
        np.testing.assert_allclose(ds['time'][:], obj.model_time)
        np.testing.assert_array_equal(ds['Itime'][:], [59000, 59000])
        np.testing.assert_array_equal(ds['Itime2'][:], [0, 10800000])
        assert ds['river_flux'].units == 'm^3s^-1'
        assert ds['river_temp'].units == 'Celsius'
        assert ds['river_salt'].units == 'PSU'
        assert ds.source.startswith('Akvaplan-niva BuildRiver, version ')


def test_writer_produces_both_files_in_custom_directory(tmp_path):
    pytest.importorskip('netCDF4')
    obj = make_forcing()
    paths = output.write_forcing_files(obj, tmp_path / 'river_output')
    assert all(path.is_file() for path in paths)
    assert "RIVER_FILE = 'riverdata.nc'" in paths[1].read_text()


def test_config_has_default_output_directory():
    info = br.get_input('/tmp/hbv')
    assert info.get('output_dir', 'input') == 'input'
    assert info.get('qc_plot_dir') is None


def test_qc_plotting_remains_interactive_and_can_save_pngs(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    obj = make_forcing()
    # x_land/y_land are read-only views of the xy_land property.
    # Replace that dependency for plotting tests without changing production code.
    monkeypatch.setattr(type(obj), 'xy_land',
                        property(lambda self: np.array([[0., 0.],
                                                        [5., 5.]])))
    grid = SimpleNamespace(georeference=lambda: None)
    rivers = SimpleNamespace(crs='unused')
    original_scatter = output.plt.scatter

    def scatter(*args, **kwargs):
        kwargs.pop('transform', None)  # Plain test Axes have no Cartopy CRS.
        return original_scatter(*args, **kwargs)

    shown = []
    monkeypatch.setattr(output.plt, 'scatter', scatter)
    monkeypatch.setattr(output.plt, 'show', lambda **kw: shown.append(kw))
    try:
        directory = tmp_path / 'figures'
        assert output.show_forcing(obj, grid, rivers, save_dir=directory) is None
        assert (directory / 'river_mean_transport.png').stat().st_size > 100
        assert (directory / 'river_max_temperature.png').stat().st_size > 100
        assert shown == [{'block': False}, {'block': False}]
    finally:
        plt.close('all')


def test_qc_plotting_does_not_write_files_by_default(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    obj = make_forcing()
    # x_land/y_land are read-only views of the xy_land property.
    # Replace that dependency for plotting tests without changing production code.
    monkeypatch.setattr(type(obj), 'xy_land',
                        property(lambda self: np.array([[0., 0.],
                                                        [5., 5.]])))
    original_scatter = output.plt.scatter

    def scatter(*args, **kwargs):
        kwargs.pop('transform', None)
        return original_scatter(*args, **kwargs)

    monkeypatch.setattr(output.plt, 'scatter', scatter)
    monkeypatch.setattr(output.plt, 'show', lambda **kw: None)
    try:
        assert output.show_forcing(obj, SimpleNamespace(georeference=lambda: None),
                                   SimpleNamespace(crs='unused')) is None
        assert list(tmp_path.iterdir()) == []
    finally:
        plt.close('all')
