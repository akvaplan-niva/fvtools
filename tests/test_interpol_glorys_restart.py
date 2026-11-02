"""Offline regression tests for GLORYS -> FVCOM restart initialization."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset, date2num

from fvtools.setup import interpol_glorys_restart as restart
from fvtools.grid.glorys_grid import GLORYS


class FakeFVCOM:
    """Small planar FVCOM mesh with a constant 20 m water depth."""
    def __init__(self, path='M.npy'):
        self.x = np.array([2500., 7500., 2500., 7500.])
        self.y = np.array([2500., 2500., 7500., 7500.])
        self.tri = np.array([[0, 1, 2], [1, 3, 2]])
        self.xc = self.x[self.tri].mean(axis=1)
        self.yc = self.y[self.tri].mean(axis=1)
        self.h = np.full(4, 20.)
        self.siglay = np.tile([-.25, -.75], (4, 1))
        self.siglay_center = np.tile([-.15, -.65], (2, 1))
        self.siglev_center = np.tile([0., -.3, -1.], (2, 1))
        self.cell_utm_angle = np.zeros(2)
        self.casename = 'tiny'

    @staticmethod
    def Proj(lon, lat):
        return np.asarray(lon)*10_000, np.asarray(lat)*10_000

    @staticmethod
    def rotate_vector_to_utm(u, v, angle):
        return u*np.cos(angle)+v*np.sin(angle), -u*np.sin(angle)+v*np.cos(angle)


def glorys_data():
    depth = np.array([.5, 10., 20.])
    shape = (2, 3, 3, 3)
    temp = np.broadcast_to(2 + depth[None, :, None, None], shape).copy()
    salt = np.full(shape, 35.)
    east = np.full(shape, 1.)
    north = np.full(shape, 2.)
    return xr.Dataset(
        data_vars={
            'thetao': (('time', 'depth', 'latitude', 'longitude'), temp),
            'so': (('time', 'depth', 'latitude', 'longitude'), salt),
            'uo': (('time', 'depth', 'latitude', 'longitude'), east),
            'vo': (('time', 'depth', 'latitude', 'longitude'), north),
            'zos': (('time', 'latitude', 'longitude'), np.full((2, 3, 3), 2.)),
            'deptho': (('latitude', 'longitude'), np.full((3, 3), 20.), {'units': 'm'}),
        },
        coords={'time': np.array(['2023-07-01T12', '2023-07-02T12'], dtype='datetime64[m]'),
                'depth': depth, 'latitude': [0., .5, 1.], 'longitude': [0., .5, 1.]},
    )


def write_restart(path, when=datetime(2023, 7, 1, 12), *, uv=True):
    model = FakeFVCOM()
    with Dataset(path, 'w') as nc:
        nc.createDimension('time', 1)
        nc.createDimension('node', 4)
        nc.createDimension('nele', 2)
        nc.createDimension('siglay', 2)
        nc.createVariable('h', 'f4', ('node',))[:] = model.h
        t = nc.createVariable('time', 'f4', ('time',))
        t.units = 'days since 1858-11-17 00:00:00'
        mjd = float(date2num(when, units=t.units))
        t[:] = mjd
        nc.createVariable('Itime', 'i4', ('time',))[:] = int(mjd)
        nc.createVariable('Itime2', 'i4', ('time',))[:] = int(round((mjd-int(mjd))*86_400_000))
        for name, dims in {
            'zeta': ('time', 'node'), 'temp': ('time', 'siglay', 'node'),
            'salinity': ('time', 'siglay', 'node'),
            'u': ('time', 'siglay', 'nele'), 'v': ('time', 'siglay', 'nele'),
            'ua': ('time', 'nele'), 'va': ('time', 'nele'),
        }.items():
            nc.createVariable(name, 'f4', dims)[:] = -99.
        nc.keep_me = 'original metadata'


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(restart, 'FVCOM_grid', FakeFVCOM)
    src = tmp_path/'glorys.nc'
    glorys_data().to_netcdf(src)
    output = tmp_path/'restart.nc'
    write_restart(output)
    return src, output


def test_exact_time_hydrography_and_zeta_shift(inputs):
    src, output = inputs
    result = restart.main(restartfile=output, glorys_file=src, latlon=True)
    assert result == str(output)
    with Dataset(output) as nc:
        np.testing.assert_allclose(nc['zeta'][0], 2.)
        # z(depth) = 2 + sigma*(20+2), so GLORYS depths = 3.5,14.5m.
        np.testing.assert_allclose(nc['temp'][0], [[5.5]*4, [16.5]*4])
        np.testing.assert_allclose(nc['salinity'][0], 35.)
        np.testing.assert_allclose(nc['u'][0], -99.)  # uv=False preserves currents
        assert nc.keep_me == 'original metadata'
        assert nc.glorys_source_time == '2023-07-01T12:00:00'


def test_cell_currents_and_sigma_weighted_averages(inputs):
    src, output = inputs
    restart.main(restartfile=output, glorys_file=src, uv=True, latlon=False)
    with Dataset(output) as nc:
        for name, expected in [('u', 1.), ('v', 2.), ('ua', 1.), ('va', 2.)]:
            np.testing.assert_allclose(nc[name][0], expected)
        np.testing.assert_allclose(nc['temp'][0], [[5.5]*4, [16.5]*4])


def test_exact_time_rejects_nearest_unless_requested(inputs):
    src, output = inputs
    write_restart(output, datetime(2023, 7, 1, 0))
    before = output.read_bytes()
    with pytest.raises(ValueError, match='No exact GLORYS'):
        restart.main(restartfile=output, glorys_file=src)
    assert output.read_bytes() == before
    restart.main(restartfile=output, glorys_file=src,
                 time_method='nearest', max_time_difference_hours=12)
    with Dataset(output) as nc:
        assert nc.glorys_source_time == '2023-07-01T12:00:00'
        assert nc.glorys_restart_time == '2023-07-01T00:00:00'


def test_invalid_interior_source_value_aborts_without_touching_restart(inputs):
    src, output = inputs
    with xr.open_dataset(src) as ds:
        modified = ds.load()
    modified['thetao'].values[0, 1, 0, 0] = np.nan
    modified.to_netcdf(src)
    before = output.read_bytes()
    with pytest.raises(ValueError, match='interior missing'):
        restart.main(restartfile=output, glorys_file=src, uv=True)
    assert output.read_bytes() == before
    assert not list(output.parent.glob('.glorys_restart_*'))


def test_restart_bathymetry_must_match_model(inputs):
    src, output = inputs
    with Dataset(output, 'r+') as nc:
        nc['h'][:] = 50.
    before = output.read_bytes()
    with pytest.raises(ValueError, match='bathymetry differs'):
        restart.main(restartfile=output, glorys_file=src)
    assert output.read_bytes() == before


def test_nearest_wet_extrapolation_is_bounded():
    ds = glorys_data()
    ds['so'].values[:, :, 0, 0] = np.nan
    grid = GLORYS(ds)
    model = FakeFVCOM()
    # Exactly on a masked GLORYS corner: no valid weighted stencil remains.
    with pytest.raises(ValueError, match='farther'):
        restart._coastal_weights(grid, model, np.array([[0., 0.]]),
                                 max_distance_m=100.)
    wet, ids, weights = restart._coastal_weights(
        grid, model, np.array([[0., 0.]]), max_distance_m=6_000.)
    assert wet.sum() == 8
    np.testing.assert_allclose(weights.sum(axis=1), 1.)
    assert np.all(ids >= 0)


def test_time_index_is_explicit_for_daily_means():
    t = np.array(['2023-07-01T12', '2023-07-02T12'], dtype='datetime64[m]')
    with pytest.raises(ValueError, match='exact'):
        restart._source_time_index(t, '2023-07-01T00')
    assert restart._source_time_index(
        t, '2023-07-01T00', method='nearest', max_hours=12) == 0
    with pytest.raises(ValueError, match='too far'):
        restart._source_time_index(
            t, '2023-07-01T00', method='nearest', max_hours=6)


def test_new_restart_is_created_in_staging_directory(inputs, monkeypatch, tmp_path):
    src, existing = inputs
    destination = tmp_path/'created.nc'
    calls = []

    def fake_make_initial_file(model, date, obc_type):
        calls.append((date, obc_type, Path.cwd()))
        write_restart(Path.cwd()/'tiny_initial.nc')
        return 'tiny_initial.nc'

    monkeypatch.setattr(restart, 'make_initial_file', fake_make_initial_file)
    restart.main(restartfile=destination, restart_date='2023-07-01-12',
                 fvcom_grid='M.npy', glorys_file=src)
    assert destination.exists()
    assert calls[0][0:2] == ('2023-07-01-12', 3)
    assert calls[0][2] != tmp_path  # make_initial_file did not run in user cwd
    with Dataset(destination) as nc:
        np.testing.assert_allclose(nc['zeta'][0], 2.)
    with pytest.raises(FileExistsError):
        restart.main(restartfile=destination, restart_date='2023-07-01-12',
                     glorys_file=src)


def test_restart_barotropic_uses_glorys_depth_not_fvcom_sigma(inputs):
    src, output = inputs
    with xr.open_dataset(src) as ds:
        edited = ds.load()
    # A strongly curved GLORYS profile. Interpolation changes the
    # raw sigma-layer mean, so the barotropic correction is required.
    edited['uo'].values[0] = np.array([1., 10., 1.])[:, None, None]
    edited['vo'].values[0] = np.array([2., 4., 0.])[:, None, None]
    edited.to_netcdf(src)

    restart.main(restartfile=output, glorys_file=src, uv=True, latlon=True)
    from fvtools.interpolators.glorys_barotropic import depth_average_glorys
    with xr.open_dataset(src) as ds:
        native_u, native_v = depth_average_glorys(
            ds.uo.isel(time=0).values, ds.vo.isel(time=0).values,
            ds.depth.values, ds.deptho.values, ds.zos.isel(time=0).values,
        )
    with Dataset(output) as nc:
        np.testing.assert_allclose(nc['ua'][0], native_u.mean(), rtol=1e-5)
        np.testing.assert_allclose(nc['va'][0], native_v.mean(), rtol=1e-5)
        sigma_weights = np.array([.3, .7])
        for component, mean_name in (('u', 'ua'), ('v', 'va')):
            layered = np.asarray(nc[component][0])
            target_mean = np.asarray(nc[mean_name][0])
            np.testing.assert_allclose(
                np.sum(layered * sigma_weights[:, None], axis=0),
                target_mean, rtol=1e-6, atol=1e-6,
            )
        # Applying a column-uniform correction preserves source vertical shear.
        z = np.array([0.5, 10., 20.])
        # Actual depths: -(zeta + sigma*(h+zeta))
        target_depths = -2 + np.abs(np.array([-.15, -.65])) * 22.
        raw_u = np.interp(target_depths, z, [1., 10., 1.])
        np.testing.assert_allclose(np.diff(nc['u'][0, :, 0]), np.diff(raw_u), atol=1e-5)
        assert 'piecewise-linear' in nc.glorys_barotropic_method
        assert 'sigma u/v corrected' in nc.glorys_barotropic_method
