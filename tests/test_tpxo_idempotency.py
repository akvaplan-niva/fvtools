"""TPXO metadata guard and atomic-write regressions; no real TPXO data needed."""

from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pytest

from fvtools.setup.nesting import tpxo


def _make_nest(filename):
    with Dataset(filename, 'w') as nc:
        nc.createDimension('time', 2)
        nc.createDimension('node', 2)
        nc.createDimension('nele', 1)
        nc.createDimension('siglay', 3)
        for name, dimension, vals in (
            ('lon', 'node', [5., 6.]), ('lat', 'node', [60., 60.]),
            ('lonc', 'nele', [5.5]), ('latc', 'nele', [60.]),
            ('xc', 'nele', [0.]), ('yc', 'nele', [0.]),
            ('Itime', 'time', [60000, 60000]),
            ('Itime2', 'time', [0, 3600000]),
        ):
            nc.createVariable(name, 'f8', (dimension,))[:] = vals
        nc.createVariable('zeta', 'f8', ('time', 'node'))[:] = 2.
        for key in ('u', 'v'):
            nc.createVariable(key, 'f8', ('time', 'siglay', 'nele'))[:] = 3.
        for key in ('ua', 'va'):
            nc.createVariable(key, 'f8', ('time', 'nele'))[:] = 4.
        nc.interpolation_projection = 'degrees'


def _fake_harmonics(monkeypatch):
    calls = []

    def load(path, key, x, y, names):
        calls.append(key)
        return np.ones((len(names), len(x)), dtype=complex)

    def predict(coeff, names, times, lat):
        return np.full((len(times), len(lat)), np.mean(np.abs(coeff)))

    monkeypatch.setattr(tpxo, '_load_harmonic_file', load)
    monkeypatch.setattr(tpxo, '_predict_utide', predict)
    return calls


def _make_bathy(filename):
    with Dataset(filename, 'w') as nc:
        nc.createDimension('lon_u', 2)
        nc.createDimension('lat_u', 2)
        nc.createDimension('lon_v', 2)
        nc.createDimension('lat_v', 2)
        for key in ('u', 'v'):
            nc.createVariable(f'lon_{key}', 'f8', (f'lon_{key}',))[:] = [5., 6.]
            nc.createVariable(f'lat_{key}', 'f8', (f'lat_{key}',))[:] = [59.5, 60.5]
            dep = nc.createVariable(f'h{key}', 'f8', (f'lat_{key}', f'lon_{key}'))
            dep[:] = 2.
            dep.units = 'm'


def test_tpxo_is_applied_only_once_even_with_currents(tmp_path, monkeypatch):
    nest = tmp_path / 'nest.nc'
    bathy = tmp_path / 'bathy.nc'
    _make_nest(nest)
    _make_bathy(bathy)
    calls = _fake_harmonics(monkeypatch)
    sources = {'zeta': 'z.nc', 'u': 'u.nc', 'v': 'v.nc'}
    tpxo.main(str(nest), harmonics_files=sources, tpxo_bathymetry=str(bathy))
    with Dataset(nest) as nc:
        assert nc.getncattr('tpxo_applied') == 1
        assert nc.getncattr('tidal_source') == 'TPXO'
        assert 'M2' in nc.getncattr('tidal_constituents')
        np.testing.assert_array_equal(nc['zeta'][:], 3.)
        for field in ('u', 'v'):
            np.testing.assert_array_equal(nc[field][:], 3.5)
        for field in ('ua', 'va'):
            np.testing.assert_array_equal(nc[field][:], 4.5)
    assert calls == ['zeta', 'u', 'v']

    content = nest.read_bytes()
    with pytest.raises(ValueError, match='already been added'):
        tpxo.main(str(nest), harmonics_files=sources, tpxo_bathymetry=str(bathy))
    assert calls == ['zeta', 'u', 'v']  # Reject before computing TPXO
    assert nest.read_bytes() == content


def test_preexisting_legacy_tidal_source_blocks_double_add(tmp_path, monkeypatch):
    nest = tmp_path / 'nest.nc'
    _make_nest(nest)
    with Dataset(nest, 'r+') as nc:
        nc.tidal_source = 'TPXO'  # The old implementation already wrote this.
    calls = _fake_harmonics(monkeypatch)
    before = nest.read_bytes()
    with pytest.raises(ValueError, match='already been added'):
        tpxo.main(str(nest), harmonics_files={'zeta': 'z.nc'})
    assert not calls
    assert nest.read_bytes() == before


def test_uncertain_tpxo_flag_blocks_instead_of_double_adding(tmp_path, monkeypatch):
    nest = tmp_path / 'nest.nc'
    _make_nest(nest)
    with Dataset(nest, 'r+') as nc:
        nc.tpxo_applied = 0  # Ambiguous/unfinished marker: fail closed.
    calls = _fake_harmonics(monkeypatch)
    with pytest.raises(ValueError, match='already been added'):
        tpxo.main(str(nest), harmonics_files={'zeta': 'z.nc'})
    assert calls == []


def test_error_during_write_keeps_original_pristine(tmp_path, monkeypatch):
    nest = tmp_path / 'nest.nc'
    _make_nest(nest)
    _fake_harmonics(monkeypatch)
    original = nest.read_bytes()

    def fail_after_partial_write(nc, predictions, constituents):
        nc['zeta'][:] = 999.
        raise RuntimeError('synthetic disk write failure')

    monkeypatch.setattr(tpxo, '_write_tpxo_predictions', fail_after_partial_write)
    with pytest.raises(RuntimeError, match='synthetic disk write failure'):
        tpxo.main(str(nest), harmonics_files={'zeta': 'z.nc'})
    assert nest.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ['nest.nc']
