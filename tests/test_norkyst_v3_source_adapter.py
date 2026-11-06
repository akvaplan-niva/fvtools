"""NorKyst v3 native C-grid vs collocated hindcast offline regression tests."""
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.grid import roms_grid
from fvtools.grid.roms.source_adapter import NORKYST_V3_HINDCAST, NATIVE_ROMS
from fvtools.interpolators.roms_interpolators import N4ROMS


DATE = datetime(2024, 4, 18)


def test_forecast_and_hindcast_have_distinct_urls_and_adapters():
    forecast = roms_grid.get_roms_grid('NKv3')
    hindcast = roms_grid.get_roms_grid('NKv3-hindcast')
    assert isinstance(forecast, roms_grid.METNorKystV3)
    assert isinstance(hindcast, roms_grid.METNorKystV3Hindcast)
    assert forecast.source_adapter == NATIVE_ROMS
    assert hindcast.source_adapter == NORKYST_V3_HINDCAST
    assert 'fou-hi/new_norkyst800m/' in forecast.get_norkyst_url(DATE)
    assert forecast.get_norkyst_url(DATE).endswith('norkyst800_his_sdepth_20240418T00Z_m00_AN.nc')
    assert hindcast.get_norkyst_url(DATE).endswith('/2024/04/norkyst800-20240418.nc')
    assert roms_grid.get_roms_grid('NKv3H').get_norkyst_url(DATE) == hindcast.get_norkyst_url(DATE)


def test_source_field_mapping_is_explicit_and_native_unchanged():
    assert NORKYST_V3_HINDCAST.field('u') == 'u_eastward'
    assert NORKYST_V3_HINDCAST.field('v') == 'v_northward'
    assert NORKYST_V3_HINDCAST.field('ubar') == 'ubar_eastward'
    assert NORKYST_V3_HINDCAST.field('vbar') == 'vbar_northward'
    assert NORKYST_V3_HINDCAST.field('salt') == 'salinity'
    assert NATIVE_ROMS.field('u') == 'u'
    assert NATIVE_ROMS.velocity_frame == 'grid'


class FakeVar:
    def __init__(self, data, dims):
        self.data = np.asarray(data)
        self.dimensions = tuple(dims)
    def __getitem__(self, key):
        return self.data[key]


class FakeDataset:
    def __init__(self, vars):
        self.variables = vars
    def __getitem__(self, key):
        return self.variables[key]
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def _hindcast_data():
    yy, xx = np.mgrid[:3, :4]
    time = (0, 1)
    d2 = ('Y', 'X')
    d3 = ('time', 'Y', 'X')
    d4 = ('time', 's_rho', 'Y', 'X')
    data = {
        'lon': FakeVar(xx, d2), 'lat': FakeVar(yy, d2),
        'h': FakeVar(20 + yy + xx, d2),
        'sea_mask': FakeVar(np.ones((3, 4), dtype='i1'), d2),
        's_rho': FakeVar([-0.75, -0.25], ('s_rho',)),
        's_w': FakeVar([-1., -.5, 0.], ('s_w',)),
        'Cs_r': FakeVar([-0.75, -0.25], ('s_rho',)),
        'Cs_w': FakeVar([-1., -.5, 0.], ('s_w',)),
        'hc': FakeVar(3.0, ()),
    }
    for name, value in [('salinity', 32.), ('temperature', 8.),
                        ('u_eastward', 2.), ('v_northward', 3.)]:
        data[name] = FakeVar(np.full((2, 2, 3, 4), value), d4)
    for name, value in [('zeta', 0.2), ('ubar_eastward', 1.5),
                        ('vbar_northward', -2.5)]:
        data[name] = FakeVar(np.full((2, 3, 4), value), d3)
    return FakeDataset(data)


def _make_grid(monkeypatch):
    ds = _hindcast_data()
    monkeypatch.setattr(roms_grid, 'Dataset', lambda *args: ds)
    grid = roms_grid.get_roms_grid('NKv3-hindcast')
    grid.Proj = lambda lon, lat, inverse=False: (lon.astype(float), lat.astype(float))
    grid.load_grid_from_nc('synthetic.nc')
    grid.get_x_y_z()
    grid.xbounds = [0, 3]
    grid.ybounds = [0, 2]
    grid.offset = 0
    return grid, ds


def test_hindcast_grid_shared_geometry_and_local_sigma(monkeypatch):
    grid, ds = _make_grid(monkeypatch)
    assert grid.x_u is grid.x_v is grid.x_rho
    assert grid.y_u is grid.y_v is grid.y_rho
    assert grid.u_mask is grid.v_mask is grid.rho_mask
    assert grid.h_u.shape == grid.h_v.shape == grid.h_rho.shape
    np.testing.assert_array_equal(grid.cropped_dsigma_u, grid.cropped_dsigma_v)
    np.testing.assert_allclose(grid.cropped_dsigma_u.sum(axis=0), 1.)


def test_hindcast_schema_rejects_invalid_grid_dimensions(monkeypatch):
    ds = _hindcast_data()
    ds.variables['lon'].dimensions = ('xi_rho', 'eta_rho')
    monkeypatch.setattr(roms_grid, 'Dataset', lambda *args: ds)
    with pytest.raises(ValueError, match='lon must have'):
        roms_grid.get_roms_grid('NKv3-hindcast').load_grid_from_nc('bad.nc')


def test_hindcast_reader_maps_all_fields_and_preserves_geographic_velocities(monkeypatch):
    grid, ds = _make_grid(monkeypatch)
    r = roms_grid.RomsDownloader()
    n4 = SimpleNamespace(
        source_adapter=grid.source_adapter,
        m_ri=0, x_ri=2, m_rj=0, x_rj=3,
        m_ui=0, x_ui=2, m_uj=0, x_uj=3,
        m_vi=0, x_vi=2, m_vj=0, x_vj=3,
        cropped_rho_mask=np.ones((3, 4), bool),
        cropped_u_mask=np.ones((3, 4), bool),
        cropped_v_mask=np.ones((3, 4), bool),
        angle=np.array([0.7]), cell_utm_angle=np.array([0.3]),
    )
    r.N4 = n4
    @contextmanager
    def get_source(_):
        yield ds
    r._roms_dataset = get_source
    step = r.read_timestep(0, 'unused.nc', max_retries=0)
    np.testing.assert_allclose(step.salt, 32)
    np.testing.assert_allclose(step.temp, 8)
    np.testing.assert_allclose(step.u, 2)
    np.testing.assert_allclose(step.v, 3)
    np.testing.assert_allclose(step.ua, 1.5)
    np.testing.assert_allclose(step.va, -2.5)
    step = r.crop_and_transpose(step)
    assert step.u.shape == (12, 2)
    assert step.v.shape == (12, 2)
    assert step.ua.shape == (12,)
    r.latlon = True
    rotated = r.adjust_uv(step)
    np.testing.assert_allclose(rotated.u, 2.)
    np.testing.assert_allclose(rotated.v, 3.)
    np.testing.assert_allclose(rotated.ua, 1.5)
    np.testing.assert_allclose(rotated.va, -2.5)
    # UTM conversion should still be applied exactly once.
    r.latlon = False
    n4.cell_utm_angle = np.full(12, .3)
    # The production pipeline rotates only after vertical interpolation,
    # when three-dimensional velocities have shape (siglay, nele).
    step.u, step.v = step.u.T.copy(), step.v.T.copy()
    r.adjust_uv(step)
    np.testing.assert_allclose(step.u, 2*np.cos(.3) + 3*np.sin(.3))
    np.testing.assert_allclose(step.v, -2*np.sin(.3) + 3*np.cos(.3))


def test_collocated_interpolator_reuses_cell_weights():
    roms = SimpleNamespace(is_collocated=True)
    coeff = N4ROMS.__new__(N4ROMS)
    coeff.ROMS = roms
    coeff.x, coeff.y = np.array([0., 1., 0.]), np.array([0., 0., 1.])
    coeff.tri = np.array([[0, 1, 2]])
    coeff.uv = True
    coeff.land_check = False
    roms.cropped_x_rho = roms.cropped_x_u = roms.cropped_x_v = np.arange(4.)
    roms.cropped_y_rho = roms.cropped_y_u = roms.cropped_y_v = np.arange(4.)
    roms.fv_rho_mask = roms.fv_u_mask = roms.fv_v_mask = np.ones((2, 2), bool)
    calls = []
    def weights(**kwargs):
        calls.append(kwargs['widget_title'])
        n = len(kwargs['xy_fvcom'])
        return np.zeros((n, 4), int), np.full((n, 4), .25)
    coeff.get_interpolation_matrices = weights
    coeff.nearest4()
    assert calls == ['rho', 'u']
    assert coeff.u_index is coeff.v_index
    assert coeff.u_coef is coeff.v_coef


def test_collocated_vertical_weights_reuse_u_for_v(monkeypatch):
    from fvtools.interpolators import vertical_interpolation as vi
    grid, _ = _make_grid(monkeypatch)
    n4 = SimpleNamespace(
        ROMS=grid,
        FV=SimpleNamespace(
            siglay=np.tile([-.75, -.25], (3, 1)),
            siglay_center=np.tile([-.75, -.25], (2, 1)),
        ),
        rho_index=np.zeros((3, 4), dtype=int),
        rho_coef=np.full((3, 4), .25),
        u_index=np.zeros((2, 4), dtype=int),
        u_coef=np.full((2, 4), .25),
        v_index=np.zeros((2, 4), dtype=int),
        v_coef=np.full((2, 4), .25),
        fvcom_rho_dpt=np.full(3, 20.),
        fvcom_u_dpt=np.full(2, 20.),
        fvcom_v_dpt=np.full(2, 20.),
    )
    calls = []
    def fake_weights(src, dst):
        calls.append((src.shape, dst.shape))
        return (np.zeros(dst.shape, dtype=int),
                np.zeros(dst.shape, dtype=int),
                np.ones(dst.shape), np.zeros(dst.shape))
    monkeypatch.setattr(vi, 'calc_interp_matrices', fake_weights)
    vi.add_vertical_interpolation2N4(n4)
    assert calls == [((2, 3), (2, 3)), ((2, 2), (2, 2))]
    assert n4.vi_ind1_v is n4.vi_ind1_u
    assert n4.vi_weigths2_v is n4.vi_weigths2_u
