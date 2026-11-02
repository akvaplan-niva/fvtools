"""Offline regression tests for CMEMS downloader and static GLORYS deptho."""

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import xarray as xr

from fvtools.grid.glorys_grid import GLORYS
from fvtools.setup.nesting.cmems_download import download_cmems


def dynamic_dataset():
    salinity = np.full((2, 3, 3), 35., dtype='f4')
    salinity[:, 0, 0] = np.nan  # dynamic land point
    ds = xr.Dataset(
        {'so': (('time', 'depth', 'latitude', 'longitude'), salinity[None])},
        coords={
            'time': np.array(['2023-07-01'], dtype='datetime64[D]'),
            'depth': [0.5, 12.],
            'latitude': [55.5, 55.6, 55.7],
            'longitude': [4.0, 4.1, 4.2],
        },
    )
    return ds


def static_dataset():
    # Larger, reversed static domain; coordinate selection must be by label.
    lat = np.array([55.8, 55.7, 55.6, 55.5, 55.4])
    lon = np.array([4.3, 4.2, 4.1, 4.0, 3.9])
    depth = 100 + 10 * np.arange(5)[:, None] + np.arange(5)[None, :]
    return xr.Dataset(
        {'deptho': (('latitude', 'longitude'), depth.astype('f4'), {'units': 'm'})},
        coords={'latitude': lat, 'longitude': lon},
    )


def test_deptho_is_actual_model_bathymetry_and_mask_unchanged():
    dynamic, static = dynamic_dataset(), static_dataset()
    grid = GLORYS(dynamic, bathymetry=static)
    assert grid.bathymetry_source == 'deptho'
    expected = static.deptho.sel(
        latitude=dynamic.latitude, longitude=dynamic.longitude
    ).values
    expected[0, 0] = np.nan
    np.testing.assert_allclose(grid.h_rho, expected, equal_nan=True)
    assert grid.land_mask[0, 0]
    assert not grid.land_mask[1, 1]
    assert grid.land_3d.shape == (2, 3, 3)
    assert grid.h_rho[1, 1] > grid.depth[-1]  # Not deepest wet z-level.


def test_old_glorys_constructor_remains_compatible(tmp_path):
    dataset = dynamic_dataset()
    filename = tmp_path / 'dynamic.nc'
    dataset.to_netcdf(filename)
    old = GLORYS(filename)
    assert old.bathymetry_source == 'deepest_wet_z_level'
    assert old.h_rho[1, 1] == pytest.approx(12.0)
    assert np.isnan(old.h_rho[0, 0])


def test_incompatible_static_grid_fails_instead_of_nearest_other_grid():
    wrong = static_dataset().assign_coords(longitude=static_dataset().longitude + .01)
    with pytest.raises(ValueError, match='coordinates do not match'):
        GLORYS(dynamic_dataset(), bathymetry=wrong)


def test_invalid_deptho_at_dynamic_wet_point_fails():
    static = static_dataset().copy(deep=True)
    static['deptho'].loc[dict(latitude=55.6, longitude=4.1)] = np.nan
    with pytest.raises(ValueError, match='invalid'):
        GLORYS(dynamic_dataset(), bathymetry=static)


def test_download_requests_static_once_and_dynamic_with_three_extra_days(
        tmp_path, monkeypatch):
    calls = []

    def fake_subset(**kwargs):
        calls.append(kwargs)
        filename = Path(kwargs['output_directory']) / kwargs['output_filename']
        if kwargs.get('dataset_part') == 'bathy':
            static_dataset().to_netcdf(filename)
        else:
            dynamic = dynamic_dataset()
            for var in ('thetao', 'uo', 'vo'):
                dynamic[var] = dynamic.so.copy()
            dynamic['zos'] = dynamic.so.isel(depth=0, drop=True)
            dynamic.to_netcdf(filename)

    monkeypatch.setitem(sys.modules, 'copernicusmarine', SimpleNamespace(subset=fake_subset))
    model = SimpleNamespace(
        lon=np.array([4.0, 4.2, 4.1]),
        lat=np.array([55.5, 55.7, 55.6]),
    )
    start = datetime(2023, 7, 1)
    stop = datetime(2023, 7, 3)
    files = download_cmems(model, start, stop, output_directory=tmp_path)
    assert set(files) == {'dynamic', 'bathymetry'}
    assert all(path.exists() for path in files.values())
    assert len(calls) == 2
    static_req, dynamic_req = calls
    assert static_req['dataset_id'] == 'cmems_mod_glo_phy_my_0.083deg_static'
    assert static_req['dataset_part'] == 'bathy'
    assert static_req['variables'] == ['deptho']
    assert 'start_datetime' not in static_req
    assert dynamic_req['dataset_id'] == 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
    assert set(dynamic_req['variables']) == {'so', 'thetao', 'uo', 'vo', 'zos'}
    assert dynamic_req['start_datetime'] == start
    assert dynamic_req['end_datetime'] == stop + timedelta(days=3)
    assert dynamic_req['minimum_longitude'] == pytest.approx(3.7)
    assert dynamic_req['maximum_latitude'] == pytest.approx(56.0)
    assert dynamic_req['coordinates_selection_method'] == 'outside'

    # Fully cached: no Copernicus calls at all on the second invocation.
    again = download_cmems(model, start, stop, output_directory=tmp_path)
    assert again == files
    assert len(calls) == 2
    assert sorted(p.suffix for p in tmp_path.iterdir()) == ['.nc', '.nc']

    grid = GLORYS(files['dynamic'], bathymetry=files['bathymetry'])
    assert grid.bathymetry_source == 'deptho'


def test_failed_redownload_cannot_replace_existing_dynamic_file(tmp_path, monkeypatch):
    model = SimpleNamespace(lon=np.array([4., 4.2]), lat=np.array([55.5, 55.7]))

    def good_subset(**kwargs):
        dest = Path(kwargs['output_directory']) / kwargs['output_filename']
        if kwargs.get('dataset_part') == 'bathy':
            static_dataset().to_netcdf(dest)
        else:
            ds = dynamic_dataset()
            for name in ('thetao', 'uo', 'vo'):
                ds[name] = ds.so.copy()
            ds['zos'] = ds.so.isel(depth=0, drop=True)
            ds.to_netcdf(dest)

    monkeypatch.setitem(sys.modules, 'copernicusmarine', SimpleNamespace(subset=good_subset))
    files = download_cmems(model, '2023-07-01', '2023-07-02', output_directory=tmp_path)
    before = files['dynamic'].read_bytes()

    def broken_subset(**kwargs):
        if kwargs.get('dataset_part') == 'bathy':
            return good_subset(**kwargs)
        dest = Path(kwargs['output_directory']) / kwargs['output_filename']
        dest.write_bytes(b'partial data')
        raise OSError('synthetic interrupted download')

    monkeypatch.setitem(sys.modules, 'copernicusmarine', SimpleNamespace(subset=broken_subset))
    with pytest.raises(OSError, match='interrupted'):
        download_cmems(model, '2023-07-01', '2023-07-02', output_directory=tmp_path,
                       force=True)
    assert files['dynamic'].read_bytes() == before
    assert not list(tmp_path.glob('*.partial.nc'))
