"""FVCOM SETUP_SPONGE parity and restart write regressions."""
from pathlib import Path

import numpy as np
import pytest

from fvtools.grid.fvcom._components.sponge import (
    Sponge, _EARTH_RADIUS_M, _read_sponge_file,
)


class Grid(Sponge):
    def __init__(self, folder, *, spherical=False):
        self.casename = 'small'
        self.filepath = str(folder / 'M.npy')
        self._sponge_source_directory = folder
        self.info = {'latlon': spherical}
        self.x = np.array([0., 10.])
        self.y = np.zeros(2)
        self.xc = np.array([0., 5., 8., 18.])
        self.yc = np.zeros(4)
        self.lon = np.array([0., 1.])
        self.lat = np.zeros(2)
        self.lonc = np.array([0., .5, .8, 1.8])
        self.latc = np.zeros(4)


def _write(folder, text):
    dst = folder / 'input'
    dst.mkdir(exist_ok=True)
    filename = dst / 'small_spg.dat'
    filename.write_text(text)
    return filename


def test_fvcom_linear_taper_and_maximum_of_overlapping_sponges(tmp_path):
    grid = Grid(tmp_path)
    _write(tmp_path, 'Sponge Node Number = 2\n1 10.0 0.2\n2 10.0 0.1\n')
    np.testing.assert_allclose(grid.calculate_cc_sponge(), [.2, .1, .08, .02], atol=1e-7)
    assert grid.calculate_cc_sponge().dtype == np.dtype('float32')


def test_sponge_uses_original_grid_directory_across_cwd_changes(tmp_path, monkeypatch):
    grid = Grid(tmp_path)
    _write(tmp_path, 'Sponge Node Number = 1\n1 10.0 0.2\n')
    other = tmp_path / 'temporary'
    other.mkdir()
    monkeypatch.chdir(other)
    np.testing.assert_allclose(grid.calculate_cc_sponge()[:2], [.2, .1])
    assert grid.find_sponge_file() == tmp_path / 'input' / 'small_spg.dat'


def test_zero_sponge_and_missing_default_yield_zeros(tmp_path):
    grid = Grid(tmp_path)
    np.testing.assert_array_equal(grid.calculate_cc_sponge(), np.zeros(4))
    _write(tmp_path, 'Sponge Node Number = 0\n')
    np.testing.assert_array_equal(grid.calculate_cc_sponge(), np.zeros(4))
    with pytest.raises(FileNotFoundError):
        grid.calculate_cc_sponge('does_not_exist.dat')


def test_spherical_matches_fvcom_arc_and_metadata_selection(tmp_path):
    grid = Grid(tmp_path, spherical=True)
    _write(tmp_path, 'Sponge Node Number = 1\n1 100000.0 0.2\n')
    dist_half_degree = _EARTH_RADIUS_M * np.deg2rad(0.5)
    expected = .2 * (1 - dist_half_degree / 100000.)
    out = grid.calculate_cc_sponge()
    assert out[0] == pytest.approx(.2, abs=1e-6)
    assert out[1] == pytest.approx(expected, abs=1e-6)
    assert out[3] == 0.
    # Explicit spherical override of grid metadata.
    assert grid.calculate_cc_sponge(spherical=False)[1] == pytest.approx(.2 * (1 - 5. / 100000.))


def test_sponge_file_validates_rows_and_1_based_node_ids(tmp_path):
    grid = Grid(tmp_path)
    for contents, match in [
        ('Sponge Node Number = 2\n1 10 .1\n', 'declares 2'),
        ('Sponge Node Number = 1\n0 10 .1\n', 'outside grid'),
        ('Sponge Node Number = 1\n3 10 .1\n', 'outside grid'),
        ('Sponge Node Number = 1\n1 0 .1\n', 'positive radius'),
        ('Sponge Node Number = 1\n1 nan .1\n', 'non-finite'),
    ]:
        _write(tmp_path, contents)
        with pytest.raises(ValueError, match=match):
            grid.calculate_cc_sponge()


def test_fortran_exponent_and_all_nonpositive_factors(tmp_path):
    grid = Grid(tmp_path)
    _write(tmp_path, 'Sponge Node Number = 2\n1 10.D0 0.D0\n2 10.d0 -0.01d0\n')
    np.testing.assert_array_equal(grid.calculate_cc_sponge(), np.zeros(4))


def test_existing_restart_sponge_is_atomic_and_other_fields_unchanged(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    grid = Grid(tmp_path)
    _write(tmp_path, 'Sponge Node Number = 1\n1 10.0 0.2\n')
    dst = tmp_path / 'small_initial.nc'
    with netCDF4.Dataset(dst, 'w') as nc:
        nc.createDimension('nele', 4)
        nc.createVariable('lonc', 'f4', ('nele',))[:] = grid.lonc
        nc.createVariable('latc', 'f4', ('nele',))[:] = grid.latc
        nc.createVariable('cc_sponge', 'f4', ('nele',))[:] = -1.
        nc.createVariable('other', 'f4', ('nele',))[:] = [8, 9, 10, 11]
    grid.update_restart_sponge(dst)
    with netCDF4.Dataset(dst) as nc:
        np.testing.assert_allclose(nc['cc_sponge'][:], [.2, .1, .04, 0.], atol=1e-7)
        np.testing.assert_array_equal(nc['other'][:], [8, 9, 10, 11])

    # Failed validation must not alter any original bytes.
    with netCDF4.Dataset(dst, 'r+') as nc:
        nc['lonc'][0] = -1000
    before = dst.read_bytes()
    with pytest.raises(ValueError, match='lonc differs'):
        grid.update_restart_sponge(dst)
    assert dst.read_bytes() == before


def test_new_initial_file_populates_cc_sponge_automatically(tmp_path, monkeypatch):
    netCDF4 = pytest.importorskip('netCDF4')
    from fvtools.setup.initial._helpers.empty_initial import make_initial_file
    g = Grid(tmp_path)
    g.x = np.array([0., 10., 0.])
    g.y = np.array([0., 0., 10.])
    g.xc = np.array([10. / 3.])
    g.yc = np.array([10. / 3.])
    g.lon = np.array([5., 5.001, 5.])
    g.lat = np.array([60., 60., 60.001])
    g.lonc = np.array([5.00033])
    g.latc = np.array([60.00033])
    g.h = np.full(3, 20.)
    g.hc = np.array([20.])
    g.tri = np.array([[0, 1, 2]], dtype=np.int32)
    g.obc_nodes = [0, 1]
    g.siglev = np.tile([0., -.5, -1.], (3, 1))
    g.siglay = np.tile([-.25, -.75], (3, 1))
    g.siglev_center = np.array([[0., -.5, -1.]])
    g.siglay_center = np.array([[-.25, -.75]])
    g.casename = 'synthetic'
    g.info['latlon'] = False
    monkeypatch.chdir(tmp_path)
    g._sponge_source_directory = tmp_path
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'synthetic_spg.dat').write_text(
        'Sponge Node Number = 1\n1 10.0 0.5\n')
    path = make_initial_file(g, '2024-08-01-00', obc_type=3)
    with netCDF4.Dataset(path) as nc:
        assert nc['cc_sponge'].dtype == np.dtype('float32')
        assert float(nc['cc_sponge'][0]) > 0
