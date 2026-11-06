"""Offline tests for a single combined GLORYS NetCDF."""

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import xarray as xr

from fvtools.grid.glorys_grid import GLORYS
from fvtools.setup.download.cmems import download_cmems


def dynamic_dataset():
    xyz = dict(time=np.array(['2023-07-01', '2023-07-02'], dtype='datetime64[D]'),
               depth=[0.5, 12.], latitude=[55.5, 55.6, 55.7],
               longitude=[4., 4.1, 4.2])
    array = np.full((2, 2, 3, 3), 35., dtype='f4')
    array[:, :, 0, 0] = np.nan
    fields = {name: (('time', 'depth', 'latitude', 'longitude'), array.copy())
              for name in ('thetao', 'so', 'uo', 'vo')}
    fields['zos'] = (('time', 'latitude', 'longitude'),
                     np.ones((2, 3, 3), dtype='f4'))
    return xr.Dataset(fields, coords=xyz)


def static_dataset(*, shifted=False):
    lon = np.array([4.3, 4.2, 4.1, 4.0, 3.9]) + (0.02 if shifted else 0.)
    lat = np.array([55.8, 55.7, 55.6, 55.5, 55.4])
    arr = (100 + np.arange(5)[:, None] * 10 + np.arange(5)[None, :]).astype('f4')
    return xr.Dataset({'deptho': (('latitude', 'longitude'), arr, {'units': 'm'})},
                      coords={'longitude': lon, 'latitude': lat})


def mock_copernicus(monkeypatch, *, shifted=False, fail_dynamic=False):
    calls = []
    def subset(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs['output_directory']) / kwargs['output_filename']
        if kwargs.get('dataset_part') == 'bathy':
            static_dataset(shifted=shifted).to_netcdf(path)
        else:
            if fail_dynamic:
                path.write_bytes(b'partial')
                raise OSError('synthetic download error')
            dynamic_dataset().to_netcdf(path)
    monkeypatch.setitem(sys.modules, 'copernicusmarine', SimpleNamespace(subset=subset))
    return calls


def model():
    return SimpleNamespace(lon=np.array([4., 4.2]), lat=np.array([55.5, 55.7]))


def test_combined_file_uses_deptho_and_keeps_dynamic_fields(tmp_path, monkeypatch):
    calls = mock_copernicus(monkeypatch)
    output = tmp_path / 'glorys.nc'
    result = download_cmems(model(), datetime(2023, 7, 1), datetime(2023, 7, 3),
                            outfile=output)
    assert result == output
    assert sorted(p.name for p in tmp_path.iterdir()) == ['glorys.nc']
    assert len(calls) == 2
    static, dynamic = calls
    assert static['dataset_part'] == 'bathy'
    assert static['variables'] == ['deptho']
    assert 'start_datetime' not in static
    assert dynamic['end_datetime'] == datetime(2023, 7, 6)
    assert set(dynamic['variables']) == {'thetao', 'so', 'uo', 'vo', 'zos'}
    assert dynamic['minimum_longitude'] == pytest.approx(3.7)
    with xr.open_dataset(result) as ds:
        assert {'thetao', 'so', 'uo', 'vo', 'zos', 'deptho'} <= set(ds.data_vars)
        assert ds.deptho.dims == ('latitude', 'longitude')
        assert ds.sizes['time'] == 2
        assert ds.so.dims == ('time', 'depth', 'latitude', 'longitude')
        assert np.isnan(ds.deptho.values[0, 0])
        assert ds.deptho.values[1, 1] == pytest.approx(122.)
    grid = GLORYS(result)
    assert grid.bathymetry_source == 'deptho'
    assert grid.h_rho[1, 1] == pytest.approx(122.)


def test_misaligned_static_grid_never_publishes(tmp_path, monkeypatch):
    mock_copernicus(monkeypatch, shifted=True)
    target = tmp_path / 'glorys.nc'
    with pytest.raises(ValueError, match='coordinates do not match'):
        download_cmems(model(), '2023-07-01', '2023-07-03', outfile=target)
    assert not list(tmp_path.iterdir())


def test_failed_download_preserves_old_output(tmp_path, monkeypatch):
    mock_copernicus(monkeypatch, fail_dynamic=True)
    target = tmp_path / 'glorys.nc'
    target.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        download_cmems(model(), '2023-07-01', '2023-07-03', outfile=target)
    with pytest.raises(OSError, match='synthetic download error'):
        download_cmems(model(), '2023-07-01', '2023-07-03',
                       outfile=target, overwrite=True)
    assert target.read_bytes() == b'original'
    assert [p.name for p in tmp_path.iterdir()] == ['glorys.nc']
