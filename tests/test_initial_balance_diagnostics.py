"""Controlled checks of FVCOM's Cartesian continuity sign convention."""
from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.setup.initial.diagnostics import (
    diagnose_initial_balance, finite_difference_eta_tendency,
)


def grid(*, obc=False):
    # One CCW triangle, a single CV half edge from (node 0) to (node 1).
    # The geometry is intentionally sparse to test EXACT FVCOM signs.
    tri = np.array([[0,1,2]], dtype=np.int64)
    t = SimpleNamespace(
        NTRG=np.array([0]), NIEC=np.array([[0,1]]),
        DLTXE=np.array([2.0]), DLTYE=np.array([0.0]), NCV=1,
        art1=np.array([2.0,4.0,4.0]),
        ISONB=np.array([0,2 if obc else 0,2 if obc else 0]),
        NBE=np.array([[-1,-1,-1]]),
    )
    return SimpleNamespace(
        tri=tri, x=np.array([0.,2.,0.]), y=np.array([0.,0.,1.]),
        xc=np.array([2/3]), yc=np.array([1/3]),
        h=np.array([10.,10.,10.]),
        siglev=np.tile([0.,-0.25,-1.], (3,1)), T=t,
        lat=np.array([60.,60.,60.]),
    )


def test_exact_fvcom_half_edge_sign_and_continuity():
    m = grid()
    result = diagnose_initial_balance(m, zeta=np.zeros(3),
                                      ua=np.array([0.]), va=np.array([1.]),
                                      coriolis=0.)
    # EXFLUX=D*v*DX=10*1*2=20 m3/s; XFLUX(IA)-=20, XFLUX(IB)+=20.
    np.testing.assert_allclose(result.nodal_flux_from_half_edges, [-20,20,0])
    np.testing.assert_allclose(result.eta_tendency_from_transport, [10,-5,0])
    assert 'continuity_residual' not in result
    assert result.attrs['n_interior_nodes'] == 3


def test_continuity_residual_vanishes_with_correct_parent_tendency():
    m = grid()
    target = [10,-5,0]
    ds = diagnose_initial_balance(m, zeta=np.zeros(3),
         ua=np.array([0.]), va=np.array([1.]), eta_tendency=target,
         coriolis=0.)
    np.testing.assert_allclose(ds.continuity_residual, 0.)


def test_boundary_nodes_are_masked_from_interior_continuity():
    m = grid(obc=True)
    ds = diagnose_initial_balance(m, zeta=np.zeros(3),
         ua=np.array([0.]), va=np.array([1.]), coriolis=0.)
    assert np.isfinite(ds.eta_tendency_from_transport.values[0])
    assert np.isnan(ds.eta_tendency_from_transport.values[1:]).all()


def test_linear_eta_gradient_and_coriolis_balance():
    m = grid()
    # eta = a*x + b*y -> exact triangle gradient.
    x, y = m.x, m.y
    ds = diagnose_initial_balance(m, zeta=0.002*x-0.003*y,
         ua=np.array([0.]), va=np.array([0.]), coriolis=0., gravity=10.)
    np.testing.assert_allclose(ds.eta_gradient_x, [0.002])
    np.testing.assert_allclose(ds.eta_gradient_y, [-0.003])
    np.testing.assert_allclose(ds.geostrophic_proxy_acceleration_x, [-0.02])
    np.testing.assert_allclose(ds.geostrophic_proxy_acceleration_y, [0.03])
    # Geostrophic balance: -g*eta_x+f*v=0; -g*eta_y-f*u=0.
    ds2 = diagnose_initial_balance(m, zeta=0.002*x-0.003*y,
          ua=np.array([0.03]), va=np.array([0.02]), coriolis=1., gravity=10.)
    np.testing.assert_allclose(ds2.geostrophic_proxy_acceleration, 0, atol=1e-14)


def test_3d_mean_and_surface_shear_are_separate():
    m = grid()
    # weights [0.25,0.75]; mean u = 1, v=0; surface 4 -> shear=3.
    ds = diagnose_initial_balance(m, zeta=np.zeros(3),
        ua=np.array([1.]), va=np.array([0.]), coriolis=0.,
        u=np.array([[4.],[0.]]), v=np.array([[0.],[0.]]))
    np.testing.assert_allclose(ds.mean_2d_3d_error, 0.)
    np.testing.assert_allclose(ds.surface_shear, 3.)
    ds2 = diagnose_initial_balance(m, zeta=np.zeros(3),
        ua=np.array([2.]), va=np.array([0.]), coriolis=0.,
        u=np.array([[4.],[0.]]), v=np.array([[0.],[0.]]))
    np.testing.assert_allclose(ds2.mean_2d_3d_error, 1.)
    np.testing.assert_allclose(ds2.surface_shear, 3.)


def test_finite_difference_and_input_validation():
    np.testing.assert_allclose(finite_difference_eta_tendency([1,2],[3,4],2), [1,1])
    with pytest.raises(ValueError, match='seconds'):
        finite_difference_eta_tendency([1],[2],0)
    with pytest.raises(ValueError, match='zeta'):
        diagnose_initial_balance(grid(), zeta=[0,0], ua=[0], va=[0], coriolis=0.)
    with pytest.raises(ValueError, match='u and v'):
        diagnose_initial_balance(grid(), zeta=[0,0,0], ua=[0], va=[0],
                                 coriolis=0., u=[[0.]])


def test_restart_wrapper_reads_velocity_in_chunks_without_writing(monkeypatch):
    import sys
    from types import SimpleNamespace
    from fvtools.setup.initial.diagnostics import diagnose_restart

    arrays = {
        'zeta': np.zeros((1,3)),
        'ua': np.array([[1.]]), 'va': np.array([[0.]]),
        'u': np.array([[[4.], [0.]]]),
        'v': np.array([[[0.], [0.]]]),
    }
    calls = []

    class FakeFile:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        @property
        def variables(self):
            return arrays
        def __getitem__(self, name):
            return arrays[name]

    def open_file(path, mode):
        calls.append((path, mode))
        return FakeFile()

    monkeypatch.setitem(sys.modules, 'netCDF4', SimpleNamespace(Dataset=open_file))
    ds = diagnose_restart(grid(), 'read_only_restart.nc', coriolis=0., chunk_size=1)
    np.testing.assert_allclose(ds.mean_2d_3d_error, [0.])
    np.testing.assert_allclose(ds.surface_shear, [3.])
    assert calls == [('read_only_restart.nc','r')]
