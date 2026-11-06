"""Synthetic NetCDF regression checks for TPXO double-application protection."""
import os

from netCDF4 import Dataset
import numpy as np
import pytest

from fvtools.setup.nesting import tpxo


def _make_nest(filename):
    with Dataset(filename, 'w') as nc:
        nc.createDimension('time', 2)
        nc.createDimension('node', 2)
        nc.createDimension('nele', 1)
        nc.createDimension('siglay', 2)
        for name, dim, values in (
            ('lon', 'node', [5, 6]), ('lat', 'node', [60, 60]),
            ('lonc', 'nele', [5.5]), ('latc', 'nele', [60]),
            ('xc', 'nele', [0]), ('yc', 'nele', [0]),
            ('Itime', 'time', [61000, 61000]),
            ('Itime2', 'time', [0, 3_600_000]),
        ):
            nc.createVariable(name, 'f8', (dim,))[:] = values
        nc.createVariable('zeta', 'f8', ('time', 'node'))[:] = 2.
        nc.interpolation_projection = 'degrees'


def _fake_prediction(monkeypatch):
    monkeypatch.setattr(tpxo, '_load_harmonic_file',
                        lambda filename, key, lon, lat, constituents:
                        np.ones((len(constituents), len(lon)), dtype=complex))
    monkeypatch.setattr(tpxo, '_predict_utide',
                        lambda c, names, t, lat: np.full((len(t), len(lat)), .25))


def test_tpxo_tag_blocks_second_addition(tmp_path, monkeypatch):
    fn = tmp_path / 'nest.nc'
    _make_nest(fn)
    _fake_prediction(monkeypatch)
    tpxo.main(fn, harmonics_files={'zeta': 'h.nc'})
    with Dataset(fn) as nc:
        assert nc.tpxo_applied == 1
        assert nc.tidal_source == 'TPXO'
        np.testing.assert_array_equal(nc['zeta'][:], 2.25)
    before = fn.read_bytes()
    with pytest.raises(ValueError, match='already been added'):
        tpxo.main(fn, harmonics_files={'zeta': 'h.nc'})
    assert fn.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['nest.nc']


def test_old_tpxo_metadata_blocks_readdition(tmp_path, monkeypatch):
    fn = tmp_path / 'nest.nc'
    _make_nest(fn)
    with Dataset(fn, 'r+') as nc:
        nc.tidal_source = 'TPXO'
    _fake_prediction(monkeypatch)
    before = fn.read_bytes()
    with pytest.raises(ValueError, match='already been added'):
        tpxo.main(fn, harmonics_files={'zeta': 'h.nc'})
    assert fn.read_bytes() == before


def test_commit_error_keeps_original_and_cleans_up(tmp_path, monkeypatch):
    fn = tmp_path / 'nest.nc'
    _make_nest(fn)
    _fake_prediction(monkeypatch)
    before = fn.read_bytes()
    def error(*args):
        raise OSError('simulated interrupted commit')
    monkeypatch.setattr(os, 'replace', error)
    with pytest.raises(OSError, match='simulated interrupted commit'):
        tpxo.main(fn, harmonics_files={'zeta': 'h.nc'})
    assert fn.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['nest.nc']


def test_locked_file_is_untouched(tmp_path, monkeypatch):
    fn = tmp_path / 'nest.nc'
    _make_nest(fn)
    _fake_prediction(monkeypatch)
    lock = tmp_path / 'nest.nc.tpxo.lock'
    lock.touch()
    before = fn.read_bytes()
    with pytest.raises(RuntimeError, match='lock'):
        tpxo.main(fn, harmonics_files={'zeta': 'h.nc'})
    assert fn.read_bytes() == before
    assert lock.exists()
