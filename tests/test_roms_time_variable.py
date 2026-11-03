"""ROMS time-axis compatibility for operational and hindcast NetCDF files.

Synthetic sources: no THREDDS connection or downloads are used.
"""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.grid import roms_grid
from fvtools.setup.nesting import roms as nesting_roms


class FakeTimeVariable:
    def __init__(self, values, units='hours since 2025-01-01 00:00:00'):
        self.values = np.asarray(values)
        self.units = units
        self.reads = 0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        self.reads += 1
        return self.values[key]


@pytest.mark.parametrize('name', ['ocean_time', 'time'])
def test_time_variable_selection(name):
    source = FakeTimeVariable(np.arange(24))
    nc = SimpleNamespace(variables={name: source})
    assert roms_grid.get_roms_time_variable(nc) is source
    assert source.reads == 0


def test_ocean_time_keeps_precedence_and_missing_axis_is_explicit():
    operational = FakeTimeVariable([0, 1])
    hindcast = FakeTimeVariable([10, 11])
    nc = SimpleNamespace(variables={'time': hindcast, 'ocean_time': operational})
    assert roms_grid.get_roms_time_variable(nc) is operational
    with pytest.raises(ValueError, match="neither 'ocean_time' nor 'time'"):
        roms_grid.get_roms_time_variable(SimpleNamespace(variables={}))


@pytest.mark.parametrize('name', ['ocean_time', 'time'])
@pytest.mark.parametrize('grid_class', [
    roms_grid.METNorKystV2, roms_grid.METNorKystV3, roms_grid.IMRNorFjords,
])
def test_norkyst_file_probe_uses_dimension_length_only(monkeypatch, name, grid_class):
    clock = FakeTimeVariable(np.arange(24))

    class Dataset:
        variables = {name: clock}
        def __init__(self, path, mode='r'):
            assert mode == 'r'
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(roms_grid, 'Dataset', Dataset)
    grid_class().test_ncfile('source.nc')
    assert clock.reads == 0  # len(variable) avoids a remote time[:] request


@pytest.mark.parametrize('name', ['ocean_time', 'time'])
def test_norshelf_day_probe_accepts_both_time_names(monkeypatch, name):
    clock = FakeTimeVariable(np.arange(24))

    class Dataset:
        variables = {name: clock}
        def __init__(self, path, mode='r'):
            assert mode == 'r'
        def close(self):
            pass

    monkeypatch.setattr(roms_grid, 'Dataset', Dataset)
    model = roms_grid.NorShelf(avg=False)
    chosen = model.test_day(datetime(2025, 1, 1))
    assert chosen.endswith('20250101T00Z.nc')
    assert clock.reads == 0


@pytest.mark.parametrize('names', [
    ('ocean_time', 'time'),
    ('time', 'ocean_time'),
    ('time', 'time'),
])
def test_nesting_filelist_with_mixed_time_names_preserves_indices(monkeypatch, names):
    sources = {
        'old.nc': {names[0]: FakeTimeVariable([0., 6., 12.])},
        'hindcast.nc': {names[1]: FakeTimeVariable([6., 18., 24.])},
    }
    closed = []

    class Dataset:
        def __init__(self, path):
            self.path = path
            self.variables = sources[path]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(self.path)
            return False

    class Source:
        def test_day(self, day):
            return 'old.nc' if day.day == 1 else 'hindcast.nc'

    monkeypatch.setattr(nesting_roms.netCDF4, 'Dataset', Dataset)
    monkeypatch.setattr(nesting_roms.netCDF4, 'num2date', lambda data, **kwargs: data)
    monkeypatch.setattr(nesting_roms.netCDF4, 'date2num', lambda data, **kwargs: data)
    monkeypatch.setattr(nesting_roms.time_module, 'sleep', lambda seconds: None)

    times, paths, indices = nesting_roms.make_fileList(
        datetime(2025, 1, 1), datetime(2025, 1, 2), Source())
    np.testing.assert_array_equal(times, [0., 6., 12., 18., 24.])
    assert paths == ['old.nc', 'hindcast.nc', 'old.nc', 'hindcast.nc', 'hindcast.nc']
    assert indices == [0, 0, 2, 1, 2]
    assert closed == ['old.nc', 'hindcast.nc']
    assert all(var.reads == 1 for fields in sources.values() for var in fields.values())


@pytest.mark.parametrize('name', ['ocean_time', 'time'])
def test_real_netcdf_time_units_are_preserved(tmp_path, name):
    import netCDF4

    path = tmp_path / 'roms.nc'
    with netCDF4.Dataset(path, 'w') as nc:
        nc.createDimension('record', 2)
        clock = nc.createVariable(name, 'f8', ('record',))
        clock.units = 'hours since 2024-01-01 00:00:00'
        clock[:] = [0., 6.]

    with netCDF4.Dataset(path) as nc:
        clock = roms_grid.get_roms_time_variable(nc)
        datetimes = netCDF4.num2date(
            clock[:], units=clock.units, only_use_cftime_datetimes=False)
    assert datetimes[0] == datetime(2024, 1, 1, 0, 0)
    assert datetimes[1] == datetime(2024, 1, 1, 6, 0)
