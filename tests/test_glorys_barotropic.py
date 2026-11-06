"""Independent regression tests for direct GLORYS -> FVCOM barotropic currents."""

import numpy as np
import pytest

from fvtools.interpolators.glorys_barotropic import (
    depth_average_glorys, correct_fvcom_barotropic,
)


def _column(depths, u, v, *, deptho=20., zos=0.):
    return depth_average_glorys(
        np.asarray(u, dtype=float)[:, None, None],
        np.asarray(v, dtype=float)[:, None, None],
        np.asarray(depths), np.array([[deptho]]), np.array([[zos]]),
    )


def test_linear_profile_integrates_analytically():
    # u(z)=1+2z, v(z)=4-z, z from 0 to 20 m.
    # Mean u = 21, mean v = -6.
    levels = [0., 10., 20.]
    ua, va = _column(levels, [1, 21, 41], [4, -6, -16])
    np.testing.assert_allclose(ua, 21.)
    np.testing.assert_allclose(va, -6.)


def test_sea_level_perturbation_changes_depth_mean():
    # zeta=+2: top -2 m, bottom +20 m.
    # linear below first data level, constant above z=0.
    ua, _ = _column([0., 10., 20.], [1., 11., 21.], [2., 2., 2.],
                    deptho=20, zos=2)
    np.testing.assert_allclose(ua, (2. * 1. + 20. * 11.) / 22.)


def test_bottom_partial_column_uses_last_wet_velocity():
    # Points below the deepest valid GLORYS level are extended with last
    # valid velocity, not filled with 0.
    ua, va = _column([0., 10., 20., 30.],
                     [1., 3., np.nan, np.nan],
                     [5., 7., np.nan, np.nan], deptho=25.)
    np.testing.assert_allclose(ua, (10. * 2. + 15. * 3.) / 25.)
    np.testing.assert_allclose(va, (10. * 6. + 15. * 7.) / 25.)


def test_mask_land_stays_nan_and_wet_ocean_is_calculated():
    levels = np.array([0., 10.])
    u = np.array([[[np.nan, 2.]], [[np.nan, 2.]]])
    v = np.array([[[np.nan, 4.]], [[np.nan, 4.]]])
    deptho = np.array([[np.nan, 10.]])
    zos = np.array([[np.nan, 1.]])
    ua, va = depth_average_glorys(u, v, levels, deptho, zos)
    assert np.isnan(ua[0, 0]) and np.isnan(va[0, 0])
    np.testing.assert_allclose(ua[0, 1], 2.)
    np.testing.assert_allclose(va[0, 1], 4.)


def test_interior_missing_velocity_is_not_silently_filled():
    with pytest.raises(ValueError, match='interior missing'):
        _column([0., 10., 20.], [1., np.nan, 2.], [1., 2., 3.])


def test_missing_wet_surface_rejected():
    with pytest.raises(ValueError, match='surface'):
        _column([0., 10.], [np.nan, 2.], [1., 2.])


def test_invalid_deptho_or_dry_column_rejected():
    with pytest.raises(ValueError, match='No selected GLORYS wet'):
        _column([0., 10.], [1., 2.], [3., 4.], deptho=0.)
    with pytest.raises(ValueError, match='invalid deptho or zos'):
        _column([0., 10.], [1., 2.], [3., 4.], deptho=1., zos=-2.)


def test_reject_invalid_coordinate_and_shapes():
    with pytest.raises(ValueError, match='increase'):
        _column([0., 0.], [1., 2.], [1., 2.])
    with pytest.raises(ValueError, match='dimensions'):
        depth_average_glorys(np.ones((2, 3, 3)), np.ones((2, 3, 3)),
                             [0., 10.], np.ones((2, 2)), np.ones((2, 2)))


def test_barotropic_not_forced_to_match_fvcom_sigma_average():
    # Source has 3 fixed z-levels; an FVCOM two-layer mean of its sampled
    # endpoints is not necessarily the depth integral of the GLORYS profile.
    ua, _ = _column([0., 5., 20.], [1., 2., 10.], [0., 0., 0.])
    sigma_values = np.interp([5., 15.], [0., 5., 20.], [1., 2., 10.])
    sigma_average = sigma_values.mean()
    assert not np.isclose(float(ua[0, 0]), sigma_average)
    np.testing.assert_allclose(ua, (5 * 1.5 + 15 * 6.) / 20.)


def test_correct_fvcom_barotropic_uses_nonuniform_sigma_thickness():
    # Two FVCOM cells with different layer thickness distributions.
    u = np.array([[1., 2.], [7., 12.], [9., 6.]])
    v = np.array([[5., -2.], [3., 4.], [-1., 8.]])
    ua = np.array([4.5, 7.0])
    va = np.array([2.0, 0.5])
    interfaces = np.array([
        [0., -.1, -.6, -1.],
        [0., -.4, -.5, -1.],
    ])
    u0 = u.copy()
    v0 = v.copy()
    corrected_u, corrected_v = correct_fvcom_barotropic(
        u, v, ua, va, interfaces)
    w = -np.diff(interfaces, axis=1).T
    w = w / w.sum(axis=0)
    np.testing.assert_allclose((corrected_u*w).sum(axis=0), ua, atol=1e-14)
    np.testing.assert_allclose((corrected_v*w).sum(axis=0), va, atol=1e-14)
    np.testing.assert_allclose(np.diff(corrected_u, axis=0), np.diff(u0, axis=0))
    np.testing.assert_allclose(np.diff(corrected_v, axis=0), np.diff(v0, axis=0))
    np.testing.assert_array_equal(u, u0)
    np.testing.assert_array_equal(v, v0)
    # The unweighted average is not sufficient in variable-thickness layers.
    assert not np.allclose(corrected_u.mean(axis=0), ua)


def test_correct_fvcom_barotropic_rejects_invalid_sigma_or_nan():
    u = np.ones((2, 1))
    v = np.ones((2, 1))
    with pytest.raises(ValueError, match='decrease'):
        correct_fvcom_barotropic(u, v, [1.], [2.], [[0., -.7, -.6]])
    with pytest.raises(ValueError, match='dimensions'):
        correct_fvcom_barotropic(u, v, [1., 2.], [2.], [[0., -.3, -1.]])
    with pytest.raises(ValueError, match='Nonfinite'):
        correct_fvcom_barotropic(u*np.nan, v, [1.], [2.], [[0., -.3, -1.]])


def test_correct_barotropic_mode_is_idempotent():
    u = np.array([[3., 1.], [7., 5.]])
    v = np.array([[2., 0.], [4., 6.]])
    ua = np.array([5., -2.])
    va = np.array([-1., 8.])
    sigma = np.array([[0., -.2, -1.], [0., -.8, -1.]])
    u1, v1 = correct_fvcom_barotropic(u, v, ua, va, sigma)
    u2, v2 = correct_fvcom_barotropic(u1, v1, ua, va, sigma)
    np.testing.assert_allclose(u2, u1, atol=1e-14)
    np.testing.assert_allclose(v2, v1, atol=1e-14)
