"""Offline TPXO->FVCOM restart tests, with a mocked astronomical predictor."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import os

import numpy as np
import pytest
from netCDF4 import Dataset

from fvtools.setup import interpol_tpxo_restart as restart


def _make_restart(path, *, coordinate_system='geographic east/north'):
    with Dataset(path, 'w') as nc:
        for dim, count in [('time', 1), ('node', 3), ('nele', 2), ('siglay', 2)]:
            nc.createDimension(dim, count)
        coords = {
            'lon': ('node', [5., 5.5, 6.]),
            'lat': ('node', [60., 60., 60.]),
            'lonc': ('nele', [5.25, 5.75]),
            'latc': ('nele', [60., 60.]),
        }
        for name, (dim, data) in coords.items():
            nc.createVariable(name, 'f8', (dim,))[:] = data
        nc.createVariable('Itime', 'i4', ('time',))[:] = [60000]
        nc.createVariable('Itime2', 'i4', ('time',))[:] = [3600000]
        nc.createVariable('zeta', 'f4', ('time', 'node'))[:] = 2.
        nc.createVariable('temp', 'f4', ('time', 'siglay', 'node'))[:] = 12.
        nc.createVariable('salinity', 'f4', ('time', 'siglay', 'node'))[:] = 35.
        for name in ('u', 'v'):
            nc.createVariable(name, 'f4', ('time', 'siglay', 'nele'))[:] = [[1., 2.], [3., 4.]]
        for name in ('ua', 'va'):
            nc.createVariable(name, 'f4', ('time', 'nele'))[:] = [2., 3.]
        nc.glorys_velocity_coordinates = coordinate_system
        nc.keep_me = 'safe'


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    outfile = tmp_path/'restart.nc'
    _make_restart(outfile)
    calls = []

    def fake_load(path, component, x, y, cons):
        calls.append(('load', component, len(x)))
        # Return 2 m elevation or 2 m2/s transports, converted by hu/hv.
        return np.full((len(cons), len(x)), 2.+0.j)

    def fake_predict(coef, cons, time, lat):
        calls.append(('predict', len(time), len(lat)))
        return np.mean(coef.real, axis=0)[None, :]

    monkeypatch.setattr(restart.tpxo, '_load_harmonic_file', fake_load)
    monkeypatch.setattr(restart, '_predict_one_time', fake_predict)
    monkeypatch.setattr(restart, '_tpxo_depth_at_elements',
                        lambda fname, component, lon, lat: np.array([2., 4.]))
    return outfile, calls


def test_only_zeta_preserves_other_fields_and_marks_once(inputs):
    outfile, calls = inputs
    result = restart.main(outfile, harmonics_files={'zeta': 'h.nc'},
                          constituents=('M2',))
    assert result == str(outfile)
    with Dataset(outfile) as nc:
        np.testing.assert_allclose(nc['zeta'][0], 4.)
        np.testing.assert_allclose(nc['u'][0], [[1., 2.], [3., 4.]])
        np.testing.assert_allclose(nc['ua'][0], [2., 3.])
        np.testing.assert_allclose(nc['temp'][0], 12.)
        assert nc.keep_me == 'safe'
        assert nc.tpxo_applied == 1
        assert nc.tidal_source == 'TPXO'
        assert nc.tidal_constituents == 'M2'
    original = outfile.read_bytes()
    with pytest.raises(ValueError, match='already been added'):
        restart.main(outfile, harmonics_files={'zeta': 'h.nc'})
    assert outfile.read_bytes() == original
    assert calls == [('load', 'zeta', 3), ('predict', 1, 3)]


def test_current_tides_are_added_uniformly_to_all_layers_and_barotropic(inputs):
    outfile, _ = inputs
    restart.main(outfile, harmonics_files={k: k+'.nc' for k in ('zeta', 'u', 'v')},
                 constituents=('M2',), tpxo_bathymetry='bathy.nc')
    with Dataset(outfile) as nc:
        np.testing.assert_allclose(nc['zeta'][0], 4.)
        np.testing.assert_allclose(nc['u'][0], [[2., 2.5], [4., 4.5]])
        np.testing.assert_allclose(nc['v'][0], [[2., 2.5], [4., 4.5]])
        np.testing.assert_allclose(nc['ua'][0], [3., 3.5])
        np.testing.assert_allclose(nc['va'][0], [3., 3.5])
        # Vertical shear unchanged and the layered arithmetic mean remains
        # exactly the depth-averaged barotropic velocity (equal thicknesses).
        for name, bar in (('u', 'ua'), ('v', 'va')):
            np.testing.assert_allclose(np.mean(nc[name][0], axis=0), nc[bar][0])
        assert nc.tpxo_current_coordinates == 'geographic east/north'


def test_projected_currents_use_fvcom_rotation(inputs, monkeypatch):
    outfile, _ = inputs
    with Dataset(outfile, 'r+') as nc:
        nc.glorys_velocity_coordinates = 'FVCOM projected x/y'

    class FakeGrid:
        def __init__(self, path):
            self.xc = np.array([1., 2.])
            self.lonc = np.array([5.25, 5.75])
            self.latc = np.array([60., 60.])
            self.cell_utm_angle = np.array([np.pi/2, np.pi/2])

        @staticmethod
        def rotate_vector_to_utm(u, v, angle):
            return u*np.cos(angle)+v*np.sin(angle), -u*np.sin(angle)+v*np.cos(angle)

    import fvtools.grid.fvcom_grd as grids
    monkeypatch.setattr(grids, 'FVCOM_grid', FakeGrid)
    restart.main(outfile, harmonics_files={k: k+'.nc' for k in ('zeta', 'u', 'v')},
                 constituents=('M2',), tpxo_bathymetry='bathy.nc', fvcom_grid='M.npy')
    with Dataset(outfile) as nc:
        # Eastward+Northward tide [1,.5] rotates to [.5,-1] at pi/2.
        np.testing.assert_allclose(nc['ua'][0], [3., 3.5], atol=1e-6)
        np.testing.assert_allclose(nc['va'][0], [1., 2.5], atol=1e-6)
        assert nc.tpxo_current_coordinates == 'FVCOM projected x/y'


def test_rejects_missing_components_and_mismatched_velocity_frame(inputs):
    outfile, calls = inputs
    sources = {'zeta':'h.nc', 'u':'u.nc'}
    with pytest.raises(ValueError, match='BOTH'):
        restart.main(outfile, harmonics_files=sources)
    with pytest.raises(ValueError, match='tpxo_bathymetry'):
        restart.main(outfile, harmonics_files={'zeta':'h.nc','u':'u.nc','v':'v.nc'})
    with pytest.raises(ValueError, match='disagrees'):
        restart.main(outfile, harmonics_files={'zeta':'h.nc','u':'u.nc','v':'v.nc'},
                     tpxo_bathymetry='bathy.nc', latlon=False)
    assert calls == []


def test_failed_prediction_leaves_restart_and_lock_untouched(inputs, monkeypatch):
    outfile, _ = inputs
    before = outfile.read_bytes()
    def fail(*args, **kwargs):
        raise RuntimeError('invalid coastline')
    monkeypatch.setattr(restart, '_predict_one_time', fail)
    with pytest.raises(RuntimeError, match='invalid coastline'):
        restart.main(outfile, harmonics_files={'zeta':'h.nc'})
    assert outfile.read_bytes() == before
    assert not list(outfile.parent.glob('*.tpxo.lock'))


def test_write_failure_does_not_mutate_original(inputs, monkeypatch):
    outfile, _ = inputs
    original = outfile.read_bytes()
    def broken_writer(nc, predictions, constituents):
        nc['zeta'][0, :] = 999.
        raise OSError('out of space')
    monkeypatch.setattr(restart.tpxo, '_write_tpxo_predictions', broken_writer)
    with pytest.raises(OSError, match='out of space'):
        restart.main(outfile, harmonics_files={'zeta':'h.nc'})
    assert outfile.read_bytes() == original
    assert sorted(p.name for p in outfile.parent.iterdir()) == ['restart.nc']


def test_vectorized_predictor_matches_exact_utide_for_standard_constituents():
    names = ('M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1')
    rng = np.random.default_rng(42)
    coefficients = rng.normal(size=(len(names), 17)) + 1j * rng.normal(size=(len(names), 17))
    latitudes = np.linspace(58., 70., 17)
    times = np.array([60234.137], dtype=float)
    actual = restart._predict_one_time(coefficients, names, times, latitudes, chunk_size=3)
    expected = restart.tpxo._predict_utide(coefficients, names, times, latitudes)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_depth_interpolation_respects_square_grid_axis_names(tmp_path):
    import xarray as xr
    grid = tmp_path / 'tpxo_h.nc'
    ds = xr.Dataset(
        data_vars={
            'hu': (('lat_u', 'lon_u'), np.array([[10., 20.], [30., 40.]]), {'units':'m'}),
            'hv': (('lon_v', 'lat_v'), np.array([[11., 31.], [21., 41.]]), {'units':'m'}),
        },
        coords={'lon_u': [5., 6.], 'lat_u': [60., 61.],
                'lon_v': [5., 6.], 'lat_v': [60., 61.]},
    )
    ds.to_netcdf(grid)
    np.testing.assert_allclose(restart._tpxo_depth_at_elements(grid, 'u',
                               np.array([5.]), np.array([61.])), [30.])
    np.testing.assert_allclose(restart._tpxo_depth_at_elements(grid, 'v',
                               np.array([5.]), np.array([61.])), [31.])
