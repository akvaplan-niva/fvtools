"""Offline tests for parent-time tendency adapters in restart initializers."""
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

from fvtools.setup.initial import fvcom, cmems


def test_fvcom_parent_tendency_uses_nearest_three_records(tmp_path, monkeypatch):
    times = np.array([100.0, 100.0 + 1.0 / 24.0, 100.0 + 2.0 / 24.0])
    zeta = np.stack([np.full(4, value) for value in (1.0, 2.0, 5.0)])
    path = tmp_path / 'mother.nc'
    with Dataset(path, 'w') as nc:
        nc.createDimension('time', 3)
        nc.createDimension('node', 4)
        nc.createVariable('time', 'f8', ('time',))[:] = times
        nc.createVariable('zeta', 'f8', ('time', 'node'))[:] = zeta

    monkeypatch.setattr(
        fvcom, 'Filelist',
        lambda _: SimpleNamespace(
            time=times, path=[str(path)] * 3, index=[0, 1, 2]),
    )
    parent_grid = SimpleNamespace(
        cropped_nodes=np.arange(4),
        find_nearest=lambda x, y, grid: np.arange(len(x)),
    )
    child_grid = SimpleNamespace(x=np.arange(4), y=np.arange(4))
    with Dataset(path) as mother:
        tendency, selected_times = fvcom._fvcom_parent_eta_tendency(
            mother, str(path), parent_grid, child_grid, times[1],
            filelist='unused-filelist.txt',
        )

    np.testing.assert_allclose(tendency, np.full(4, 2.0 / 3600.0))
    np.testing.assert_array_equal(selected_times, times)


def test_glorys_parent_tendency_uses_three_daily_samples(monkeypatch):
    times = np.array(
        ['2026-01-01T00', '2026-01-02T00', '2026-01-03T00'],
        dtype='datetime64[ms]',
    )
    fields = [np.full((2, 2), value) for value in (1.0, 2.0, 5.0)]
    dataset = SimpleNamespace(time=SimpleNamespace(values=times))
    monkeypatch.setattr(cmems, '_read_field', lambda ds, name, index: fields[index])
    mask = np.ones((2, 2), dtype=bool)
    ids = np.array([[0, 1, 2, 3]])
    weights = np.full((1, 4), 0.25)

    tendency, selected_times = cmems._glorys_eta_tendency(
        dataset, fields[1], 1, times[1], mask, ids, weights)
    np.testing.assert_allclose(tendency, [2.0 / 86400.0])
    np.testing.assert_array_equal(selected_times, times)
