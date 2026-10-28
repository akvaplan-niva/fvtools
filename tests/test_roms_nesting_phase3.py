"""Phase 3: static ROMS sigma interpolation and safe MatchTopo blending.

Synthetic tests only: no THREDDS access, no FVCOM input/output files.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.nesting.vertical_interpolation import (
    add_vertical_interpolation2N4,
    calc_interp_matrices,
)
from fvtools.nesting import matchtopo
from fvtools.interpolators.roms_interpolators import LinearInterpolation


def _evaluate(z1, z2):
    a, b, wa, wb = calc_interp_matrices(z1, z2)
    assert a.shape == b.shape == wa.shape == wb.shape == z2.shape
    assert np.all(np.isfinite(wa)) and np.all(np.isfinite(wb))
    np.testing.assert_allclose(wa + wb, 1., atol=1e-14)
    for col in range(z1.shape[1]):
        np.testing.assert_allclose(
            z1[a[:, col], col] * wa[:, col] + z1[b[:, col], col] * wb[:, col],
            np.clip(z2[:, col], z1[0, col], z1[-1, col]), atol=1e-12,
        )
    return a, b, wa, wb


def test_monotone_depths_are_linearly_interpolated_and_clamped():
    z1 = np.array([[1., 2.], [3., 5.], [6., 8.], [9., 11.]])
    z2 = np.array([[0., 1.], [1., 2.], [2., 6.], [4., 9.],
                   [6., 11.], [7., 15.], [12., 20.]])
    _evaluate(z1, z2)


def test_repeated_roms_sigma_depths_skip_zero_thickness_intervals():
    # In the old code, the nearest point at z=2 could select the other
    # coincident z=2 as its neighbour, dividing by zero for z=3.
    z1 = np.array([[1.], [2.], [2.], [4.]])
    z2 = np.array([[1.5], [2.], [2.5], [3.], [4.]])
    _evaluate(z1, z2)


def test_all_equal_depths_are_nearest_neighbour_not_nan():
    z1 = np.full((4, 1), 3.)
    z2 = np.array([[1.], [3.], [5.]])
    _evaluate(z1, z2)


@pytest.mark.parametrize('source,target,reason', [
    (np.array([[1., 2.], [3., 4.]]), np.zeros((3, 1)), 'matching columns'),
    (np.array([[1.], [np.nan], [3.]]), np.array([[2.]]), 'finite'),
    (np.array([[1.], [4.], [3.]]), np.array([[2.]]), 'shallow-to-deep'),
])
def test_invalid_vertical_coordinates_fail_clearly(source, target, reason):
    with pytest.raises(ValueError, match=reason):
        calc_interp_matrices(source, target)


def test_full_static_roms_to_fvcom_sigma_weights_and_field_interpolation():
    # ROMS values are bottom-to-top; the FVCOM target is surface-to-bottom.
    levels = np.array([-9., -6., -3., -1.])
    roms = SimpleNamespace(
        z_rho=np.broadcast_to(levels, (3, 3, 4)).copy(),
        z_u=np.broadcast_to(levels, (3, 2, 4)).copy(),
        z_v=np.broadcast_to(levels, (2, 3, 4)).copy(),
        fv_rho_mask=np.ones((3, 3), dtype=bool),
        fv_u_mask=np.ones((3, 2), dtype=bool),
        fv_v_mask=np.ones((2, 3), dtype=bool),
    )
    fv = SimpleNamespace(
        siglay=np.array([[-.1, -.4, -.7]]),
        siglay_center=np.array([[-.1, -.4, -.7]]),
    )
    weights = np.full((1, 4), .25)
    n4 = SimpleNamespace(
        ROMS=roms, FV=fv,
        rho_index=np.array([[0, 1, 3, 4]]), rho_coef=weights.copy(),
        u_index=np.array([[0, 1, 2, 3]]), u_coef=weights.copy(),
        v_index=np.array([[0, 1, 2, 3]]), v_coef=weights.copy(),
        fvcom_rho_dpt=np.array([10.]), fvcom_u_dpt=np.array([10.]),
        fvcom_v_dpt=np.array([10.]),
    )
    add_vertical_interpolation2N4(n4)
    values = 2. * levels + 5.  # The linear physical z-profile.
    t = SimpleNamespace(**{key: values[None, :].copy()
                           for key in ('salt', 'temp', 'u', 'v')})
    interpolator = LinearInterpolation()
    interpolator.N4 = n4
    interpolator.vertical_interpolation(t, variables=['salt', 'temp', 'u', 'v'])
    expected = np.array([[3.], [-3.], [-9.]])
    for field in ('salt', 'temp', 'u', 'v'):
        np.testing.assert_allclose(getattr(t, field), expected, atol=1e-12)


def _case(x=(0., 1., 2.5, 5.5, 7.)):
    x = np.array(x)
    m = SimpleNamespace(x=x, y=np.zeros_like(x), obc_nodes=np.array([0]),
                        h=np.full(len(x), 10.), info={})
    nest = SimpleNamespace(R=1.)
    return m, nest


def test_matchtopo_transition_weights_preserve_existing_interpolation():
    m, nest = _case()
    match = matchtopo.MatchTopo(None, m, nest, latlon=False)
    # Interior 0..1 => 1, transition x=2.5 and x=5.5 => 2/3 and 0.
    np.testing.assert_allclose(match.weight, [1., 1., 2/3, 0., 0.], atol=1e-13)


def test_matchtopo_no_transition_nodes_does_not_crash():
    m, nest = _case(x=(0., 1., 7., 8.))
    match = matchtopo.MatchTopo(None, m, nest, latlon=False)
    np.testing.assert_array_equal(match.weight, [1., 1., 0., 0.])


def test_matchtopo_missing_interior_nodes_has_clear_error():
    m, nest = _case(x=(0., 3., 5., 7.))
    match = matchtopo.MatchTopo(None, m, nest, latlon=False)
    # Override the geometry so no sampled node is inside the nesting radius.
    match.Rdst = np.array([3., 3.5, 5., 7.])
    with pytest.raises(ValueError, match='No FVCOM nodes'):
        _ = match.weight


def test_blending_rejects_invalid_roms_depth_before_changing_mesh():
    m, nest = _case()
    match = matchtopo.MatchTopo(None, m, nest, latlon=False)
    original = m.h.copy()
    with pytest.raises(ValueError, match='Nonfinite ROMS bathymetry'):
        match._blend_coarse_depth(np.array([80., np.nan, 90., 90.]))
    np.testing.assert_array_equal(m.h, original)


def test_blending_finite_roms_depth_preserves_weighted_output():
    m, nest = _case()
    match = matchtopo.MatchTopo(None, m, nest, latlon=False)
    match._blend_coarse_depth(np.array([80., 80., 80., 80.]))
    np.testing.assert_allclose(m.h, [80., 80., 10. + 70. * 2/3, 10., 10.])


def test_invalid_coarse_at_zero_weight_cannot_poison_output():
    m, nest = _case()
    match = matchtopo.MatchTopo(None, m, nest, latlon=False)
    match._blend_coarse_depth(np.array([80., 80., 80., np.nan]))
    assert np.all(np.isfinite(m.h))
    np.testing.assert_allclose(m.h, [80., 80., 10. + 70. * 2/3, 10., 10.])


def test_glorys_missing_bathymetry_keeps_original_fvcom_depth():
    m, nest = _case()
    match = matchtopo.MatchTopo(None, m, nest, latlon=False)
    match._blend_coarse_depth(np.array([80., np.nan, 80., np.inf]), fallback_invalid=True)
    np.testing.assert_allclose(m.h, [80., 10., 10. + 70. * 2/3, 10., 10.])


def test_roms_matchtopo_failure_does_not_call_disk_update(monkeypatch):
    m, nest = _case()
    coarse = SimpleNamespace(h_rho=np.array([[np.nan, 10., 20., 30., 40.]]),
                             fv_rho_mask=np.ones((1, 5), dtype=bool))
    obj = matchtopo.MatchTopo(coarse, m, nest, latlon=False)
    calls = []
    monkeypatch.setattr(obj, '_update_depth_and_sigma', lambda: calls.append('saved'))

    class FakeN4:
        def __init__(self, source, x, y, uv, land_check):
            self.rho_index = np.tile(np.arange(len(x))[:, None], (1, 4))
            self.rho_coef = np.full((len(x), 4), 0.25)
        def nearest4(self):
            return None

    monkeypatch.setattr(matchtopo, 'N4ROMSNESTING', FakeN4)
    with pytest.raises(ValueError, match='Nonfinite ROMS bathymetry'):
        obj.add_ROMS_bathymetry_to_FVCOM_and_NEST()
    np.testing.assert_array_equal(m.h, 10.)
    assert not calls
