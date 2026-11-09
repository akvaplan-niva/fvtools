"""Finite-volume tests for conservative FVCOM restart initialization."""

from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.setup.initial.continuity import (
    _half_edge_matrix, project_initial_continuity,
    correct_restart_continuity, apply_initial_continuity,
    select_time_indices, quadratic_eta_tendency,
)


def small_mesh(n=6):
    """Synthetic projected grid with genuine centroid-to-midpoint CV halfedges."""
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    x, y = x.ravel(), y.ravel()
    tri = []
    for j in range(n-1):
        for i in range(n-1):
            a = j*n+i
            tri.append((a, a+1, a+n+1))
            tri.append((a, a+n+1, a+n))
    tri = np.array(tri, dtype=int)
    # CCW orientation throughout
    xc = np.mean(x[tri], axis=1)
    yc = np.mean(y[tri], axis=1)
    nbr = np.full((len(tri), 3), -1, dtype=int)
    by_edge = {}
    for cell, corners in enumerate(tri):
        for k in range(3):
            edge = tuple(sorted((corners[(k+1)%3], corners[(k+2)%3])))
            if edge in by_edge:
                other, kk = by_edge[edge]
                nbr[cell,k] = other
                nbr[other,kk] = cell
            else:
                by_edge[edge] = cell,k
    btype = np.zeros(len(x), dtype=int)
    perimeter = ((x==0)|(x==n-1)|(y==0)|(y==n-1))
    btype[perimeter] = 2  # all outer boundary nodes are OBC

    niec, cell_ids, dx, dy = [], [], [], []
    for cell, corners in enumerate(tri):
        for k in range(3):
            a,b = corners[k], corners[(k+1)%3]
            niec.append((a,b))
            cell_ids.append(cell)
            dx.append(0.5*(x[a]+x[b])-xc[cell])
            dy.append(0.5*(y[a]+y[b])-yc[cell])

    metrics = SimpleNamespace(
        NCV=len(cell_ids),
        NIEC=np.array(niec), NTRG=np.array(cell_ids),
        DLTXE=np.array(dx), DLTYE=np.array(dy),
        art1=np.ones(len(x)), ISONB=btype, NBE=nbr,
    )
    return SimpleNamespace(x=x,y=y,tri=tri,h=np.ones(len(x))*5,
                           siglev=np.tile([0.,-0.4,-1.],(len(x),1)),T=metrics)


def state(grid):
    ncell=len(grid.tri)
    zeta=np.zeros(len(grid.x))
    ua=0.08 + .02*np.sin(np.arange(ncell))
    va=-0.03+.01*np.cos(np.arange(ncell))
    return zeta,ua,va


def eta_target(grid, zeta, ua, va):
    B, area, _, _, _ = _half_edge_matrix(grid)
    D=np.mean((grid.h+zeta)[grid.tri],axis=1)
    return -(B @ (np.column_stack((ua*D,va*D)).ravel())) / area


def test_operator_constant_velocity_gives_zero_interior_tendency():
    grid=small_mesh()
    zeta=np.zeros(len(grid.x))
    target=eta_target(grid,zeta,
                      np.full(len(grid.tri), 0.12),
                      np.full(len(grid.tri),-0.22))
    assert np.allclose(target[grid.T.ISONB==0],0,atol=1e-14)


def test_projection_matches_target_and_preserves_all_outer_cell_velocities():
    grid=small_mesh()
    zeta,ua,va=state(grid)
    target=eta_target(grid,zeta,ua,va)
    # Perturb only interior cells; projection must remove continuity errors
    free=~np.any(grid.T.NBE<0,axis=1)
    ua2=ua.copy(); va2=va.copy()
    ua2[free]+=np.linspace(-0.025,0.025,free.sum())
    va2[free]+=0.012
    result=project_initial_continuity(grid,zeta=zeta,ua=ua2,va=va2,
                eta_tendency=target,max_velocity_correction=0.3)
    assert result.attrs['rms_residual_before_m_s']>1e-5
    assert result.attrs['rms_residual_after_m_s']<1e-9
    assert np.all(result['delta_ua'].values[~free]==0)
    assert np.all(result['delta_va'].values[~free]==0)
    assert result.attrs['n_fixed_boundary_cells']>0


def test_identity_when_already_continuous():
    grid=small_mesh()
    zeta,ua,va=state(grid)
    target=eta_target(grid,zeta,ua,va)
    result=project_initial_continuity(grid,zeta=zeta,ua=ua,va=va,
                                       eta_tendency=target)
    np.testing.assert_array_equal(result['corrected_ua'].values, ua)
    np.testing.assert_array_equal(result['corrected_va'].values, va)
    assert result.attrs['max_velocity_correction_m_s']==0.


def test_refuses_correction_exceeding_speed_limit():
    grid=small_mesh()
    zeta,ua,va=state(grid)
    target=eta_target(grid,zeta,ua,va)
    free=~np.any(grid.T.NBE<0,axis=1)
    ua[free]+=0.5
    with pytest.raises(ValueError,match='safety limit'):
        project_initial_continuity(grid,zeta=zeta,ua=ua,va=va,
                                   eta_tendency=target,max_velocity_correction=1e-4)


def test_unreachable_target_fails_closed():
    grid=small_mesh()
    zeta,ua,va=state(grid)
    target=eta_target(grid,zeta,ua,va)
    # Constrain all but one otherwise interior triangle to be fixed.
    # The remaining two velocity DOF cannot fit every interior nodal target.
    grid.T.NBE[:,:] = -1
    grid.T.NBE[len(grid.tri)//2,:] = 0
    target[grid.T.ISONB==0]+=0.1
    with pytest.raises(ValueError, match='Continuity target not achieved|safety limit'):
        project_initial_continuity(grid,zeta=zeta,ua=ua,va=va,
                                   eta_tendency=target,max_velocity_correction=100.)


def test_nonfinite_and_bad_inputs():
    grid=small_mesh()
    zeta,ua,va=state(grid)
    target=eta_target(grid,zeta,ua,va)
    with pytest.raises(ValueError,match='finite'):
        project_initial_continuity(grid,zeta=zeta,ua=ua,va=va,
                                   eta_tendency=target*np.nan)
    with pytest.raises(ValueError,match='Dry'):
        project_initial_continuity(grid,zeta=zeta-10,ua=ua,va=va,
                                   eta_tendency=target)


def test_atomic_restart_writes_both_2d_and_3d_without_shear_change(tmp_path):
    netcdf4=pytest.importorskip('netCDF4')
    grid=small_mesh()
    zeta,ua,va=state(grid)
    target=eta_target(grid,zeta,ua,va)
    free=~np.any(grid.T.NBE<0,axis=1)
    u0=ua.copy();v0=va.copy()
    ua[free]+=0.015;va[free]-=0.008
    source=tmp_path/'original.nc';output=tmp_path/'projected.nc'
    with netcdf4.Dataset(source,'w') as f:
        for dim,size in [('time',1),('node',len(zeta)),('nele',len(ua)),('siglay',2)]:
            f.createDimension(dim,size)
        f.createVariable('zeta','f8',('time','node'))[0,:]=zeta
        f.createVariable('ua','f8',('time','nele'))[0,:]=ua
        f.createVariable('va','f8',('time','nele'))[0,:]=va
        # Velocities consistent with 2D means, but with nonzero shear.
        f.createVariable('u','f8',('time','siglay','nele'))[0,:,:]=np.vstack((ua+0.06,ua-0.04))
        f.createVariable('v','f8',('time','siglay','nele'))[0,:,:]=np.vstack((va-0.03,va+0.02))
    before=source.read_bytes()
    diagnostics=correct_restart_continuity(grid,source,output,eta_tendency=target)
    assert source.read_bytes()==before
    with netcdf4.Dataset(source) as f0,netcdf4.Dataset(output) as f:
        np.testing.assert_allclose(f['ua'][0],diagnostics['corrected_ua'].values)
        np.testing.assert_allclose(f['va'][0],diagnostics['corrected_va'].values)
        np.testing.assert_allclose(f['u'][0,0]-f['u'][0,1], f0['u'][0,0]-f0['u'][0,1])
        np.testing.assert_allclose(f['v'][0,0]-f['v'][0,1], f0['v'][0,0]-f0['v'][0,1])
        np.testing.assert_array_equal(f['zeta'][0],f0['zeta'][0])
    with pytest.raises(FileExistsError):
        correct_restart_continuity(grid,source,output,eta_tendency=target)



def test_select_time_indices_returns_nearest_distinct_samples_in_time_order():
    times = np.array([-4.0, -2.0, 0.0, 2.0, 2.0, 8.0])
    selected = select_time_indices(times, 1.0)
    np.testing.assert_array_equal(times[selected], [-2.0, 0.0, 2.0])


def test_quadratic_eta_tendency_is_exact_for_unequal_time_intervals():
    times = np.array([-3.0, 0.5, 4.0])
    target = 0.25
    zeta = np.stack((
        2.0 + 3.0 * times + 0.5 * times**2,
        -1.0 + 4.0 * times - 2.0 * times**2,
    ), axis=1)
    expected = np.array([3.0 + target, 4.0 - 4.0 * target])
    np.testing.assert_allclose(
        quadratic_eta_tendency(times, zeta, target),
        expected, rtol=0, atol=1e-12,
    )


def test_apply_initial_continuity_preserves_shear_and_updates_2d_and_3d(tmp_path):
    netcdf4 = pytest.importorskip('netCDF4')
    grid = small_mesh()
    zeta, ua, va = state(grid)
    target = eta_target(grid, zeta, ua, va)
    free = ~np.any(grid.T.NBE < 0, axis=1)
    ua[free] += 0.015
    va[free] -= 0.008
    source = tmp_path / 'staged.nc'
    with netcdf4.Dataset(source, 'w') as nc:
        for dim, size in (
            ('time', 1), ('node', len(zeta)), ('nele', len(ua)), ('siglay', 2),
        ):
            nc.createDimension(dim, size)
        nc.createVariable('zeta', 'f8', ('time', 'node'))[0, :] = zeta
        nc.createVariable('ua', 'f8', ('time', 'nele'))[0, :] = ua
        nc.createVariable('va', 'f8', ('time', 'nele'))[0, :] = va
        nc.createVariable('u', 'f8', ('time', 'siglay', 'nele'))[0, :, :] = np.vstack(
            (ua + 0.06, ua - 0.04))
        nc.createVariable('v', 'f8', ('time', 'siglay', 'nele'))[0, :, :] = np.vstack(
            (va - 0.03, va + 0.02))

    with netcdf4.Dataset(source, 'r+') as nc:
        u_before = nc['u'][0, :, :].copy()
        v_before = nc['v'][0, :, :].copy()
        report = apply_initial_continuity(nc, grid, target)

    with netcdf4.Dataset(source) as nc:
        np.testing.assert_allclose(nc['ua'][0], report['corrected_ua'].values)
        np.testing.assert_allclose(nc['va'][0], report['corrected_va'].values)
        np.testing.assert_allclose(
            nc['u'][0, 0] - nc['u'][0, 1], u_before[0] - u_before[1])
        np.testing.assert_allclose(
            nc['v'][0, 0] - nc['v'][0, 1], v_before[0] - v_before[1])
        np.testing.assert_array_equal(nc['zeta'][0], zeta)
        assert 'initial_continuity_rms_after_m_s' in nc.ncattrs()
