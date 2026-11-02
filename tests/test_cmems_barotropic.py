"""Barotropic velocity in CMEMS nesting must come from native GLORYS z profiles."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from fvtools.setup.nesting import cmems
from fvtools.interpolators.glorys_barotropic import depth_average_glorys


def _glorys():
    z = np.array([0.5, 10., 20.])
    data = np.zeros((1, 3, 2, 2), dtype=float)
    u = data.copy()
    v = data.copy()
    u[0] = np.array([1., 10., 1.])[:, None, None]
    v[0] = np.array([2., 4., 0.])[:, None, None]
    return xr.Dataset({
        'thetao': (('time','depth','latitude','longitude'), data + 5.),
        'so': (('time','depth','latitude','longitude'), data + 35.),
        'uo': (('time','depth','latitude','longitude'), u),
        'vo': (('time','depth','latitude','longitude'), v),
        'zos': (('time','latitude','longitude'), np.ones((1,2,2))*2.),
        'deptho': (('latitude','longitude'), np.ones((2,2))*20., {'units':'m'}),
    }, coords={'time': np.array(['2023-07-01T12'],dtype='datetime64[m]'),
               'depth':z, 'latitude':[60., 61.], 'longitude':[5., 6.]})


def _interpolate(ds, *, latlon=True):
    mask = np.ones((2,2), dtype=bool)
    corners = np.array([[0,1,2,3]], dtype=int)
    weights = np.full((1,4), .25)
    # Two FVCOM sigma layers, with weighting unrelated to GLORYS layer thickness.
    mat = (np.array([[0],[1]]), np.array([[1],[2]]),
           np.array([[.5], [.25]]), np.array([[.5], [.75]]))
    nest = SimpleNamespace(
        cell_utm_angle=np.array([0.]),
        siglev_center=np.array([[0., -.2, -1.]]),
        rotate_vector_to_utm=lambda u,v,angle: (u,v),
    )
    return cmems._read_and_interpolate(ds, 0, mask, (corners,weights),
                                       (corners,weights), mat, mat, nest,
                                       latlon=latlon)


def test_barotropic_fields_are_source_integrals():
    ds = _glorys()
    out = _interpolate(ds)
    ua, va = depth_average_glorys(
        ds.uo.isel(time=0).values, ds.vo.isel(time=0).values,
        ds.depth.values, ds.deptho.values, ds.zos.isel(time=0).values,
    )
    np.testing.assert_allclose(out['ua'], ua.mean(), rtol=1e-12)
    np.testing.assert_allclose(out['va'], va.mean(), rtol=1e-12)
    # 20%/80% relative layer thickness: corrected u/v must match ua/va.
    thickness = np.array([.2, .8])
    np.testing.assert_allclose(out['u'][:,0] @ thickness, out['ua'][0])
    np.testing.assert_allclose(out['v'][:,0] @ thickness, out['va'][0])
    # Vertical shear from the originally interpolated profile is unchanged.
    np.testing.assert_allclose(np.diff(out['u'][:,0]), [3.25 - 5.5])
    np.testing.assert_allclose(np.diff(out['v'][:,0]), [1.0 - 3.0])
    assert out['u'].shape == (2,1)
    assert out['zeta'].shape == (1,)


def test_missing_static_bathymetry_rejected():
    ds = _glorys().drop_vars('deptho')
    with pytest.raises(ValueError, match='requires static deptho'):
        _interpolate(ds)


def test_baro_interpolated_after_native_depth_integration():
    ds = _glorys()
    # The integral depends on local GLORYS bathymetry, so it is generally
    # NOT equivalent to depth-averaging an interpolated u profile.
    ds['deptho'].values[:] = [[10., 15.], [20., 30.]]
    out = _interpolate(ds)
    native_u, _ = depth_average_glorys(
        ds.uo.isel(time=0).values, ds.vo.isel(time=0).values,
        ds.depth.values, ds.deptho.values, ds.zos.isel(time=0).values,
    )
    np.testing.assert_allclose(out['ua'][0], native_u.mean())
    np.testing.assert_allclose(np.dot(out['u'][:,0], [.2, .8]), out['ua'][0])


def test_corrected_nesting_velocity_matches_rotated_barotropic_mode(monkeypatch):
    from fvtools.setup.nesting import cmems
    ds = _glorys()
    monkeypatch.setattr(cmems, 'depth_average_glorys',
                        lambda *args, **kwargs: (np.full((2, 2), 3.),
                                                 np.full((2, 2), 4.)))
    # 90-degree rotation: u_rot=v, v_rot=-u.
    original = cmems._read_and_interpolate
    # Route through an adapter whose nest exposes a real vector rotation.
    mask = np.ones((2, 2), dtype=bool)
    corners = np.array([[0, 1, 2, 3]], dtype=int)
    weights = np.full((1, 4), .25)
    matrices = (np.array([[0], [1]]), np.array([[1], [2]]),
                np.full((2, 1), .5), np.full((2, 1), .5))
    nest = SimpleNamespace(
        cell_utm_angle=np.array([np.pi/2]),
        siglev_center=np.array([[0., -.3, -1.]]),
        rotate_vector_to_utm=lambda u, v, angle: (
            u*np.cos(angle)+v*np.sin(angle),
            -u*np.sin(angle)+v*np.cos(angle)),
    )
    out = original(ds, 0, mask, (corners, weights), (corners, weights),
                   matrices, matrices, nest, latlon=False)
    np.testing.assert_allclose(out['ua'], [4.], atol=1e-12)
    np.testing.assert_allclose(out['va'], [-3.], atol=1e-12)
    np.testing.assert_allclose(np.dot(out['u'][:, 0], [.3, .7]), [4.], atol=1e-12)
    np.testing.assert_allclose(np.dot(out['v'][:, 0], [.3, .7]), [-3.], atol=1e-12)
