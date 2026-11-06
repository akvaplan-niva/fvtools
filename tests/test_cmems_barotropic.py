"""Offline tests: GLORYS-depth-integrated barotropic mode and moving FVCOM sigma."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from fvtools.grid.glorys_grid import GLORYS
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


def _geometry(ds, *, angle=0.):
    # Three FVCOM nodes forming one element whose center is the center of
    # GLORYS' 2x2 source cell. This exercises real interpolation factories.
    x = np.array([5.25, 5.75, 5.5])
    y = np.array([60.25, 60.25, 61.])
    tri = np.array([[0, 1, 2]])
    nest = SimpleNamespace(
        x=x, y=y, xc=np.array([x.mean()]), yc=np.array([y.mean()]),
        h=np.full(3, 20.), tri=tri,
        siglay=np.tile([-.15, -.65], (3, 1)),
        siglay_center=np.array([[-.15, -.65]]),
        siglev_center=np.array([[0., -.2, -1.]]),
        cell_utm_angle=np.array([angle]),
        rotate_vector_to_utm=lambda u, v, theta: (
            u*np.cos(theta)+v*np.sin(theta),
            -u*np.sin(theta)+v*np.cos(theta)),
    )
    model = SimpleNamespace(Proj=lambda lon, lat: (lon, lat))
    return nest, cmems._prepare_geometry(model, nest, GLORYS(ds))


def _interpolate(ds, *, latlon=True, angle=0.):
    nest, (mask, nm, cm, nv, cv) = _geometry(ds, angle=angle)
    return cmems._read_and_interpolate(
        ds, 0, mask, nm, cm, nv, cv, nest, latlon=latlon)


def test_barotropic_fields_are_source_integrals():
    ds = _glorys()
    out = _interpolate(ds)
    ua, va = depth_average_glorys(
        ds.uo.isel(time=0).values, ds.vo.isel(time=0).values,
        ds.depth.values, ds.deptho.values, ds.zos.isel(time=0).values,
    )
    np.testing.assert_allclose(out['ua'], ua.mean(), rtol=1e-12)
    np.testing.assert_allclose(out['va'], va.mean(), rtol=1e-12)
    thickness = np.array([.2, .8])
    np.testing.assert_allclose(out['u'][:,0] @ thickness, out['ua'][0])
    np.testing.assert_allclose(out['v'][:,0] @ thickness, out['va'][0])
    # For zeta=2 and H=20, sigma midpoints refer to 1.3 and 12.3 m
    # below the GLORYS mean sea-level reference.
    target = [-2 + .15*22, -2 + .65*22]
    raw_u = np.interp(target, ds.depth.values, [1., 10., 1.])
    raw_v = np.interp(target, ds.depth.values, [2., 4., 0.])
    np.testing.assert_allclose(np.diff(out['u'][:,0]), np.diff(raw_u))
    np.testing.assert_allclose(np.diff(out['v'][:,0]), np.diff(raw_v))
    assert out['u'].shape == (2,1)
    assert out['zeta'].shape == (3,)


def test_missing_static_bathymetry_rejected():
    ds = _glorys().drop_vars('deptho')
    # The per-timestep routine validates this before writing anything.
    nest, _ = _geometry(_glorys())
    mask, nm, cm, nv, cv = _geometry(_glorys())[1]
    with pytest.raises(ValueError, match='requires static deptho'):
        cmems._read_and_interpolate(ds, 0, mask, nm, cm, nv, cv, nest, latlon=True)


def test_baro_interpolated_after_native_depth_integration():
    ds = _glorys()
    ds['deptho'].values[:] = [[10., 15.], [20., 30.]]
    out = _interpolate(ds)
    native_u, _ = depth_average_glorys(
        ds.uo.isel(time=0).values, ds.vo.isel(time=0).values,
        ds.depth.values, ds.deptho.values, ds.zos.isel(time=0).values,
    )
    np.testing.assert_allclose(out['ua'][0], native_u.mean())
    np.testing.assert_allclose(np.dot(out['u'][:,0], [.2, .8]), out['ua'][0])


def test_corrected_nesting_velocity_matches_rotated_barotropic_mode(monkeypatch):
    ds = _glorys()
    monkeypatch.setattr(cmems, 'depth_average_glorys',
                        lambda *args, **kwargs: (np.full((2,2), 3.),
                                                 np.full((2,2), 4.)))
    out = _interpolate(ds, latlon=False, angle=np.pi / 2)
    np.testing.assert_allclose(out['ua'], [4.], atol=1e-12)
    np.testing.assert_allclose(out['va'], [-3.], atol=1e-12)
    np.testing.assert_allclose(np.dot(out['u'][:,0], [.2, .8]), [4.], atol=1e-12)
    np.testing.assert_allclose(np.dot(out['v'][:,0], [.2, .8]), [-3.], atol=1e-12)


def test_changing_zeta_changes_vertical_interpolation_not_baro_consistency():
    ds = _glorys()
    z = np.asarray(ds.depth.values)
    ds['thetao'].values[0] = z[:, None, None] + 5.
    ds['uo'].values[0] = (z * .1)[:, None, None]
    ds['vo'].values[0] = (z * .2)[:, None, None]
    first = _interpolate(ds)
    np.testing.assert_allclose(first['temp'][:,0], [6.3, 17.3])
    ds['zos'].values[:] = 0.
    second = _interpolate(ds)
    np.testing.assert_allclose(second['temp'][:,0], [8., 18.])
    # The FVCOM vertical shear follows moving sigma coordinates.
    assert not np.allclose(np.diff(first['u'][:,0]), np.diff(second['u'][:,0]))
    for result in (first, second):
        np.testing.assert_allclose(result['u'][:,0] @ np.array([.2,.8]), result['ua'][0])
        np.testing.assert_allclose(result['v'][:,0] @ np.array([.2,.8]), result['va'][0])


def test_geometry_weights_reused_and_vertical_recomputed_each_timestep(monkeypatch):
    ds = _glorys()
    real_horizontal = cmems.structured_bilinear_weights
    real_vertical = cmems.calc_interp_matrices
    calls = {'horizontal':0, 'vertical':0}

    def h(*args, **kwargs):
        calls['horizontal'] += 1
        return real_horizontal(*args, **kwargs)

    def v(*args, **kwargs):
        calls['vertical'] += 1
        return real_vertical(*args, **kwargs)

    monkeypatch.setattr(cmems, 'structured_bilinear_weights', h)
    monkeypatch.setattr(cmems, 'calc_interp_matrices', v)
    nest, (mask, nm, cm, nv, cv) = _geometry(ds)
    assert calls == {'horizontal':2, 'vertical':0}
    cmems._read_and_interpolate(ds,0,mask,nm,cm,nv,cv,nest,latlon=True)
    ds['zos'].values[:] = -1.
    cmems._read_and_interpolate(ds,0,mask,nm,cm,nv,cv,nest,latlon=True)
    assert calls == {'horizontal':2, 'vertical':4}


def test_extreme_negative_zeta_rejected():
    ds = _glorys()
    ds['zos'].values[:] = -21.
    with pytest.raises(ValueError, match='non-positive water columns|invalid deptho or zos'):
        _interpolate(ds)
