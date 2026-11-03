"""Offline regression tests for robust, low-request MetCoOp file listing."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.grid import arome_grid
from fvtools.setup.atm import read_metCoop as met


class TimeVariable:
    units = 'hours since 2024-04-18 00:00:00'

    def __init__(self, *, fail=False):
        self.fail = fail

    def __getitem__(self, key):
        if self.fail:
            raise OSError('NetCDF: Access failure')
        return np.arange(8.)[key]


class Dataset:
    def __init__(self, variable, opens, path):
        self.variables = {'time': variable}
        self.opens = opens
        self.opens.append(path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_modern_day_reuses_time_from_successful_probe(monkeypatch):
    opened = []
    grid = arome_grid.NewAromeGrid(lambda *args, **kwargs: (args[0], args[1]))
    monkeypatch.setattr(arome_grid, 'Dataset',
                        lambda path, mode='r': Dataset(TimeVariable(), opened, path))
    monkeypatch.setattr(met.netCDF4, 'num2date',
                        lambda vals, **kw: np.asarray(vals))
    monkeypatch.setattr(met.netCDF4, 'date2num',
                        lambda vals, **kw: np.asarray(vals))
    timestamps, paths, acc, rad, indices = met.metcoop_make_fileList(
        datetime(2024, 4, 18), datetime(2024, 4, 18), NewAROME=grid)
    assert len(opened) == 1  # One source open, not two.
    np.testing.assert_array_equal(timestamps, [2., 3., 4., 5.])
    assert indices == [2, 3, 4, 5]
    assert paths == acc == rad == [opened[0]] * 4


def test_confirmed_missing_url_tries_second_archive_without_retry(monkeypatch):
    opened = []
    monkeypatch.setattr(arome_grid, '_opendap_http_status',
                        lambda path: 404 if path == 'missing.nc' else 200)

    def opener(path, mode='r'):
        opened.append(path)
        if path == 'missing.nc':
            raise OSError('NetCDF: Access failure')
        return Dataset(TimeVariable(), [], path)

    monkeypatch.setattr(arome_grid, 'Dataset', opener)
    grid = arome_grid.NewAromeGrid(None)
    assert grid._test_paths(['missing.nc', 'valid.nc']) == 'valid.nc'
    assert opened == ['missing.nc', 'valid.nc']


def test_http_403_is_access_error_not_a_missing_day(monkeypatch):
    monkeypatch.setattr(arome_grid, 'Dataset',
                        lambda path, mode='r': (_ for _ in ()).throw(OSError('NetCDF: Access failure')))
    monkeypatch.setattr(arome_grid, '_opendap_http_status', lambda path: 403)
    grid = arome_grid.NewAromeGrid(None)
    with pytest.raises(PermissionError, match='403'):
        grid._test_paths(['forbidden.nc', 'fallback.nc'])


def test_http_429_is_retried_and_cannot_silently_skip(monkeypatch):
    calls = []
    monkeypatch.setattr(arome_grid, '_opendap_http_status', lambda path: 429)
    monkeypatch.setattr(arome_grid.time_mod, 'sleep', lambda seconds: calls.append(('sleep', seconds)))

    def opener(path, mode='r'):
        calls.append('open')
        if calls.count('open') == 1:
            raise OSError('NetCDF: Access failure')
        return Dataset(TimeVariable(), [], path)

    monkeypatch.setattr(arome_grid, 'Dataset', opener)
    assert arome_grid.NewAromeGrid(None)._test_paths(['limited.nc']) == 'limited.nc'
    assert calls == ['open', ('sleep', 10), 'open']


def test_http_200_but_netcdf_read_fails_is_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(arome_grid, '_opendap_http_status', lambda path: 200)
    monkeypatch.setattr(arome_grid.time_mod, 'sleep', lambda _: calls.append('sleep'))

    def opener(path, mode='r'):
        calls.append('open')
        return Dataset(TimeVariable(fail=calls.count('open') == 1), [], path)

    monkeypatch.setattr(arome_grid, 'Dataset', opener)
    grid = arome_grid.NewAromeGrid(None)
    assert grid._test_paths(['unsteady.nc'], capture_time=True) == 'unsteady.nc'
    assert calls == ['open', 'sleep', 'open']
    assert grid._verified_time[0] == 'unsteady.nc'


def test_unknown_access_failure_is_not_reported_as_missing(monkeypatch):
    monkeypatch.setattr(arome_grid, 'Dataset',
                        lambda path, mode='r': (_ for _ in ()).throw(OSError('NetCDF: Access failure')))
    monkeypatch.setattr(arome_grid, '_opendap_http_status', lambda path: None)
    monkeypatch.setattr(arome_grid.time_mod, 'sleep', lambda _: None)
    grid = arome_grid.NewAromeGrid(None)
    with pytest.raises(RuntimeError, match='after 4 attempts') as exc:
        grid._test_paths(['unavailable.nc'])
    assert isinstance(exc.value.__cause__, OSError)
