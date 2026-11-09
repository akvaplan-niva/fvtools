"""Regressions for ROMS hydrography-only restarts (uv=False)."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from fvtools.interpolators.roms_interpolators import N4ROMS
from fvtools.setup.initial.roms import Roms2FVCOMRestart


def test_restart_download_without_uv_needs_only_rho_crop(monkeypatch):
    """The real reader must not access absent U/V crop metadata."""
    restart = Roms2FVCOMRestart.__new__(Roms2FVCOMRestart)
    restart.path = 'roms.nc'
    restart.index_here = 1
    restart.N4 = SimpleNamespace(
        m_ri=1, x_ri=2, m_rj=0, x_rj=1,
        cropped_rho_mask=np.ones((2, 2), dtype=bool),
        # Deliberately NO u/v crop attributes (the uv=False case).
    )

    salt = np.arange(2 * 3 * 4 * 3, dtype=float).reshape(2, 3, 4, 3)
    temp = salt + 1000.
    full_zeta = np.arange(4 * 3, dtype=float).reshape(4, 3)
    used_files = []

    @contextmanager
    def source(path):
        used_files.append(path)
        yield {'salt': salt, 'temp': temp}

    monkeypatch.setattr(restart, '_roms_dataset', source)
    timestep = restart.download(
        variables=['salt', 'temp', 'zeta'], full_zeta=full_zeta)

    expected_salt = salt[1, :, 1:3, 0:2].reshape(3, 4).T
    expected_temp = temp[1, :, 1:3, 0:2].reshape(3, 4).T
    np.testing.assert_array_equal(timestep.salt, expected_salt)
    np.testing.assert_array_equal(timestep.temp, expected_temp)
    np.testing.assert_array_equal(timestep.zeta, full_zeta[1:3, 0:2].ravel())
    assert used_files == ['roms.nc']


def test_dump_without_uv_does_not_load_velocity_metadata():
    """The compact N4 should contain only scalar-field source information."""
    roms = SimpleNamespace(fv_rho_mask=np.ones((2, 2), dtype=bool))
    n4 = N4ROMS()
    n4.ROMS = roms
    n4.FV = SimpleNamespace(cell_utm_angle=np.zeros(1))
    n4.uv = False
    n4.rho_index = np.array([[0, 1, 2, 3]])
    n4.rho_coef = np.full((1, 4), 0.25)

    compact = n4.dump()

    assert compact.fv_rho_mask is roms.fv_rho_mask
    np.testing.assert_array_equal(compact.rho_index, n4.rho_index)
    assert not hasattr(compact, 'cropped_dsigma_u')
    assert not hasattr(compact, 'cropped_dsigma_v')
    assert not hasattr(compact, 'fv_u_mask')
    assert not hasattr(compact, 'fv_v_mask')
