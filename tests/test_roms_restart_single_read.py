"""Offline regressions for single-read ROMS -> FVCOM restart preparation.

All THREDDS, NetCDF, and geometric computations are mocked where irrelevant.
The numerical ROMS/FVCOM interpolation is covered by the existing nesting tests.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.prepro import interpol_roms_restart as restart


@pytest.mark.parametrize('uv', [False, True])
def test_main_reads_one_roms_timestep_and_reuses_vertical_weights(monkeypatch, uv):
    events = []
    zeta = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_zeta = np.array([1.5, 3.5])

    class Grid:
        def __init__(self, filename):
            events.append(('grid', filename))
            self.x = np.array([0.0, 1.0])
            self.y = np.array([0.0, 1.0])
            self.xc = self.yc = np.array([0.5])
            self.h = np.array([10.0, 20.0])
            self.tri = np.array([[0, 1, 1]], dtype=int)
            self.Proj = object()
            self.zeta = np.zeros(2)

        @property
        def d(self):
            return self.h + self.zeta

    class Roms:
        cropped_x_rho = cropped_y_rho = np.arange(4)
        cropped_x_u = cropped_y_u = np.arange(4)
        cropped_x_v = cropped_y_v = np.arange(4)
        Land_rho = Land_u = Land_v = np.zeros(4, dtype=bool)

        def __init__(self):
            self.zeta = None
            self.z = None

        def __str__(self):
            return 'fake ROMS'

        def load_grid(self, x, y, *, date):
            events.append('load_grid')

    roms = Roms()

    class N4:
        def __init__(self, source, **kwargs):
            self.ROMS = source
            self.rho_index = np.array([[0, 1], [2, 3]])
            self.rho_coef = np.full((2, 2), 0.5)
            if uv:
                self.u_index = self.v_index = np.array([[0, 1]])
                self.u_coef = self.v_coef = np.array([[0.5, 0.5]])
            events.append('n4_init')

        def nearest4(self):
            events.append('nearest4')

        def correct_land(self, index, coef, *args, **kwargs):
            return index, coef

        def dump(self):
            events.append('n4_dump')
            return SimpleNamespace(vertical_ready=hasattr(self, 'vertical_ready'))

    constructed = []

    class Downloader:
        def __init__(self, outfile, n4, tri, variables, latlon, verbose=False):
            constructed.append(self)
            self.N4 = n4.dump()
            events.append('downloader_init')

        @contextmanager
        def reuse_roms_datasets(self):
            events.append('scope_open')
            try:
                yield
            finally:
                events.append('scope_close')

        @property
        def z(self):
            events.append('zeta_read')
            return zeta

        def download(self, *, variables, full_zeta):
            events.append(('download', tuple(variables)))
            assert full_zeta is zeta
            assert roms.zeta is zeta
            return SimpleNamespace(zeta=zeta.ravel().copy())

        def dump(self, timestep, *, variables):
            events.append('write')
            assert self.N4.vertical_ready
            assert np.array_equal(timestep.zeta, zeta.ravel())
            assert tuple(variables) == (
                ('salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va') if uv
                else ('salt', 'temp', 'zeta')
            )

    class DummyDataset:
        def __init__(self, path):
            assert path == 'restart.nc'
            events.append('local_restart_open')

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def __getitem__(self, name):
            assert name == 'time'
            return TimeVariable()

    # The actual netCDF time variable must support the [0] indexing.
    class TimeVariable:
        units = 'days since 1858-11-17'

        def __getitem__(self, index):
            assert index == 0
            return 123.0

    monkeypatch.setattr(restart, 'FVCOM_grid', Grid)
    monkeypatch.setattr(restart, 'get_roms_grid', lambda mother: roms)
    monkeypatch.setattr(restart, 'N4ROMSRESTART', N4)
    monkeypatch.setattr(restart, 'Roms2FVCOMRestart', Downloader)
    monkeypatch.setattr(restart.netCDF4, 'Dataset', DummyDataset)
    monkeypatch.setattr(restart.netCDF4, 'num2date', lambda *args, **kwargs: object())

    def add_vertical_weights(n4, *, coords):
        events.append(('vertical', tuple(coords)))
        np.testing.assert_allclose(n4.FV.zeta, expected_zeta)
        np.testing.assert_allclose(n4.h, [11.5, 23.5])
        assert n4.ROMS.zeta is zeta
        np.testing.assert_array_equal(n4.ROMS.z, zeta.ravel())
        n4.vertical_ready = True
        return n4

    monkeypatch.setattr(restart.vi, 'add_vertical_interpolation2N4', add_vertical_weights)
    restart.main(restartfile='restart.nc', fvcom_grid='M.npy', mother='NKv3', uv=uv)

    assert len(constructed) == 1
    assert events.count('zeta_read') == 1
    assert sum(isinstance(item, tuple) and item[0] == 'download' for item in events) == 1
    assert events.count('n4_dump') == 2  # before and after vertical weights
    assert events.index('scope_close') < events.index(('vertical', ('rho', 'u', 'v') if uv else ('rho',)))
    assert events[-1] == 'write'


def test_download_reuses_full_zeta_and_preserves_existing_crop(monkeypatch):
    obj = restart.Roms2FVCOMRestart.__new__(restart.Roms2FVCOMRestart)
    obj.path = 'remote.nc'
    obj.index_here = 2
    mask = np.array([[True, False, True],
                     [False, True, False],
                     [True, False, True]])
    obj.N4 = SimpleNamespace(m_ri=1, x_ri=3, m_rj=2, x_rj=4,
                             cropped_rho_mask=mask)
    full_zeta = np.arange(30, dtype=float).reshape(5, 6)
    calls = []

    def reader(index, path, *, variables):
        calls.append((index, path, tuple(variables)))
        if 'zeta' in variables:
            return SimpleNamespace(zeta=full_zeta[1:4, 2:5].copy())
        return SimpleNamespace()

    monkeypatch.setattr(obj, 'read_timestep', reader)
    without_duplicate_read = obj.download(variables=['zeta'], full_zeta=full_zeta)
    assert calls == [(2, 'remote.nc', ())]
    expected = full_zeta[1:4, 2:5][mask]
    np.testing.assert_array_equal(without_duplicate_read.zeta, expected)

    calls.clear()
    legacy = obj.download(variables=['zeta'])
    assert calls == [(2, 'remote.nc', ('zeta',))]
    np.testing.assert_array_equal(legacy.zeta, expected)


def test_full_zeta_reader_uses_shared_dataset_context(monkeypatch):
    obj = restart.Roms2FVCOMRestart.__new__(restart.Roms2FVCOMRestart)
    obj.path = 'remote.nc'
    obj.index_here = 1
    full = np.arange(24).reshape(2, 3, 4)
    observed = []

    @contextmanager
    def dataset(path):
        observed.append(path)
        yield {'zeta': full}

    monkeypatch.setattr(obj, '_roms_dataset', dataset)
    np.testing.assert_array_equal(obj.z, full[1])
    assert observed == ['remote.nc']
