"""Read-only diagnostics of an interpolated FVCOM initial/restart state.

The nodal transport operator matches the Cartesian control-volume half-edge
summation in FVCOM's EXTEL_EDGE: T.NTRG, T.NIEC, T.DLTXE, T.DLTYE and T.art1.
The open-boundary continuity equation, spherical correction, and the full
momentum equation are *not* represented by this diagnostic.

For momentum the routine reports the *linear geostrophic acceleration proxy*
(-g grad(eta) + f(v,-u)), not the complete FVCOM momentum residual. Do not
interpret a large value as proof of an erroneous initial condition: advection,
friction, time derivatives, density forces and other physics are omitted.

The input restart file is never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from fvtools.grid.fvcom_grd import FVCOM_grid


EARTH_OMEGA = 7.2921159e-5  # rad/s


def _array(value, name, shape):
    arr = np.asarray(np.ma.filled(np.ma.asarray(value, dtype=np.float64), np.nan))
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite and have shape {shape}; got {arr.shape}")
    return arr


def _sigma_weights(fvcom_grid: FVCOM_grid, start: int, stop: int, layers: int):
    """FVCOM's cell sigma thickness fractions for a contiguous cell chunk."""
    interfaces = getattr(fvcom_grid, 'siglev', None)
    if interfaces is None:
        raise ValueError('FVCOM_grid.siglev is needed for the 2D/3D consistency check')
    interfaces = np.asarray(interfaces)
    tri = np.asarray(fvcom_grid.tri, dtype=np.int64)[start:stop]
    if interfaces.ndim != 2 or interfaces.shape != (len(fvcom_grid.x), layers + 1):
        raise ValueError('FVCOM_grid.siglev needs shape (node, siglay+1)')
    weights = -np.diff(np.mean(interfaces[tri], axis=1), axis=1)
    if (not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.allclose(np.sum(weights, axis=1), 1.0, rtol=0, atol=2e-4)):
        raise ValueError('FVCOM sigma layers must have positive thickness and sum to one')
    return weights


def _add_velocity_profile_checks(result, fvcom_grid, read_pair, layers, *, chunk_size):
    """Read (layers, chunk) velocities without holding the 3D domain in RAM."""
    ncell = len(fvcom_grid.tri)
    ua = np.asarray(result['ua'].values)
    va = np.asarray(result['va'].values)
    mismatch = np.empty(ncell)
    shear = np.empty(ncell)
    for start in range(0, ncell, chunk_size):
        stop = min(start + chunk_size, ncell)
        u, v = read_pair(start, stop)
        u = _array(u, 'u chunk', (layers, stop - start))
        v = _array(v, 'v chunk', (layers, stop - start))
        w = _sigma_weights(fvcom_grid, start, stop, layers)
        umean = np.sum(u.T * w, axis=1)
        vmean = np.sum(v.T * w, axis=1)
        mismatch[start:stop] = np.hypot(umean - ua[start:stop], vmean - va[start:stop])
        # Relative to the *actual* 3D mean, not to ua/va; isolates shear.
        shear[start:stop] = np.hypot(u[0] - umean, v[0] - vmean)
    result['mean_2d_3d_error'] = ('nele', mismatch, {'units': 'm s-1',
        'description': 'Norm of sigma-weighted u/v mean minus ua/va'})
    result['surface_shear'] = ('nele', shear, {'units': 'm s-1',
        'description': 'Surface velocity minus actual sigma-weighted 3D mean'})
    result.attrs['max_mean_2d_3d_error_m_s'] = float(mismatch.max())
    result.attrs['p95_surface_shear_m_s'] = float(np.percentile(shear, 95))


def _pressure_gradient(fvcom_grid: FVCOM_grid, zeta):
    """Exact gradient of a piecewise-linear nodal field in each triangle."""
    tri = np.asarray(fvcom_grid.tri, dtype=np.int64)
    x = np.asarray(fvcom_grid.x, dtype=np.float64)[tri]
    y = np.asarray(fvcom_grid.y, dtype=np.float64)[tri]
    eta = zeta[tri]
    det = (x[:, 1]-x[:, 0])*(y[:, 2]-y[:, 0]) - (x[:, 2]-x[:, 0])*(y[:, 1]-y[:, 0])
    if not np.isfinite(det).all() or np.any(np.abs(det) <= 1e-20):
        raise ValueError('Zero-area or nonfinite triangle in projected mesh')
    gx = (eta[:, 0]*(y[:, 1]-y[:, 2]) + eta[:, 1]*(y[:, 2]-y[:, 0])
          + eta[:, 2]*(y[:, 0]-y[:, 1])) / det
    gy = (eta[:, 0]*(x[:, 2]-x[:, 1]) + eta[:, 1]*(x[:, 0]-x[:, 2])
          + eta[:, 2]*(x[:, 1]-x[:, 0])) / det
    return gx, gy


def _coriolis(fvcom_grid: FVCOM_grid, coriolis, ncell):
    if coriolis is not None:
        if np.isscalar(coriolis):
            return np.full(ncell, float(coriolis))
        return _array(coriolis, 'coriolis', (ncell,))
    lat = getattr(fvcom_grid, 'lat', None)
    if lat is None:
        return None
    lat = np.asarray(lat, dtype=np.float64)
    if (lat.shape != (len(fvcom_grid.x),) or not np.isfinite(lat).all()
            or np.any(np.abs(lat) > 90) or np.all(lat == 0)):
        return None
    return 2 * EARTH_OMEGA * np.sin(np.deg2rad(np.mean(lat[fvcom_grid.tri], axis=1)))


def _open_boundary_volume_flux(fvcom_grid: FVCOM_grid, tge, depth, ua, va):
    """Positive outward; only exterior edges with *both* endpoints on OBC."""
    tri = np.asarray(fvcom_grid.tri, dtype=np.int64)
    nbe = np.asarray(tge.NBE)
    isonb = np.asarray(tge.ISONB)
    if nbe.shape != tri.shape:
        raise ValueError('T.NBE must have shape (nele,3)')
    edge_cells, corners = np.nonzero(nbe < 0)
    a = tri[edge_cells, (corners+1) % 3]
    b = tri[edge_cells, (corners+2) % 3]
    wet = (isonb[a] == 2) & (isonb[b] == 2)
    edge_cells, a, b = edge_cells[wet], a[wet], b[wet]
    if len(a) == 0:
        return 0.0, 0
    x = np.asarray(fvcom_grid.x, dtype=np.float64)
    y = np.asarray(fvcom_grid.y, dtype=np.float64)
    dx, dy = x[b]-x[a], y[b]-y[a]
    length = np.hypot(dx, dy)
    if np.any(length <= 0):
        raise ValueError('Degenerate open-boundary edge')
    nx, ny = dy / length, -dx / length
    ix = np.asarray(fvcom_grid.xc)[edge_cells] - (x[a] + x[b])/2
    iy = np.asarray(fvcom_grid.yc)[edge_cells] - (y[a] + y[b])/2
    sign = np.where(nx*ix+ny*iy > 0, -1.0, 1.0)
    q = depth[edge_cells] * (ua[edge_cells] * nx + va[edge_cells] * ny) * length * sign
    return float(np.sum(q)), len(q)


def diagnose_initial_balance(
    fvcom_grid: FVCOM_grid, *, zeta, ua, va, eta_tendency=None,
    coriolis=None, gravity=9.81, u=None, v=None, chunk_size=50000,
) -> xr.Dataset:
    """Diagnose initial FVCOM consistency without modifying the fields.

    Parameters
    ----------
    fvcom_grid : FVCOM_grid
        Full, uncropped projected mesh in metres, with matching node/cell IDs,
        bathymetry, sigma interfaces (if u/v provided) and OBC node flags.
        Uses ``fvcom_grid.T`` to obtain FVCOM's half-edge geometry.
    zeta, ua, va : arrays
        Nodal sea surface elevation [m] and cell barotropic velocity [m/s].
    eta_tendency : array (node,), optional
        *Independent* target d(eta)/dt from the parent model, already
        interpolated to the FVCOM nodes [m/s]. Without it, only the implied
        eta tendency is computed; a true continuity residual is unavailable.
    coriolis : float or (nele,) array, optional
        Coriolis frequency [s-1]. Otherwise computed from grid.lat if available.
    u, v : arrays (siglay,nele), optional
        Include 3D velocity consistency. Both must be passed together.
    gravity : float
        Gravity [m/s2].

    Returns
    -------
    xarray.Dataset
        Read-only analysis; physical units and caveats in variable attributes.
        Boundary-node divergence is masked because EXTEL_EDGE treats OBC
        nodes separately (and closed-boundary CVs also need special handling).
    """
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if not np.isfinite(gravity) or gravity <= 0:
        raise ValueError('gravity must be positive and finite')
    tri = np.asarray(fvcom_grid.tri, dtype=np.int64)
    x, y = np.asarray(fvcom_grid.x), np.asarray(fvcom_grid.y)
    if tri.ndim != 2 or tri.shape[1] != 3 or (tri.min() < 0 or tri.max() >= len(x)):
        raise ValueError('FVCOM_grid.tri must be zero-based shape (nele,3)')
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Valid projected x/y coordinates in metres are required')
    nnode, ncell = len(x), len(tri)
    zeta = _array(zeta, 'zeta', (nnode,))
    ua = _array(ua, 'ua', (ncell,))
    va = _array(va, 'va', (ncell,))
    h = _array(fvcom_grid.h, 'fvcom_grid.h', (nnode,))
    if np.any(h <= 0):
        raise ValueError('Bathymetry must be positive')
    depth = np.mean((h+zeta)[tri], axis=1)
    if np.any(h+zeta <= 0) or np.any(depth <= 0):
        raise ValueError('Initial water-column depth must be positive')

    t = fvcom_grid.T
    ntrg = np.asarray(t.NTRG, dtype=np.int64)[:int(t.NCV)]
    niec = np.asarray(t.NIEC, dtype=np.int64)[:int(t.NCV)]
    dx = np.asarray(t.DLTXE, dtype=np.float64)[:int(t.NCV)]
    dy = np.asarray(t.DLTYE, dtype=np.float64)[:int(t.NCV)]
    area = _array(t.art1, 'fvcom_grid.T.art1', (nnode,))
    btype = np.asarray(t.ISONB)
    if (niec.shape != (len(ntrg),2) or dx.shape != ntrg.shape or dy.shape != ntrg.shape
        or btype.shape != (nnode,) or np.any(area <= 0) or
        np.any(ntrg < 0) or np.any(ntrg >= ncell) or
        np.any(niec < 0) or np.any(niec >= nnode)):
        raise ValueError('Invalid FVCOM_grid.T control-volume metrics')

    # EXTEL_EDGE's sign convention, using float64 accumulation for diagnosis.
    exflux = depth[ntrg] * (-ua[ntrg] * dy + va[ntrg] * dx)
    nodal_flux = (-np.bincount(niec[:,0], weights=exflux, minlength=nnode)
                  + np.bincount(niec[:,1], weights=exflux, minlength=nnode))
    interior = btype == 0
    if not interior.any():
        raise ValueError('No interior nodes; cannot diagnose interior continuity')
    eta_from_transport = np.full(nnode, np.nan)
    eta_from_transport[interior] = -nodal_flux[interior] / area[interior]

    ds = xr.Dataset(coords={'node': np.arange(nnode), 'nele': np.arange(ncell)})
    ds['zeta'] = ('node', zeta, {'units': 'm'})
    ds['ua'] = ('nele', ua, {'units': 'm s-1'})
    ds['va'] = ('nele', va, {'units': 'm s-1'})
    ds['depth_cell'] = ('nele', depth, {'units': 'm'})
    ds['barotropic_speed'] = ('nele', np.hypot(ua,va), {'units': 'm s-1'})
    ds['nodal_flux_from_half_edges'] = ('node', nodal_flux, {'units': 'm3 s-1',
        'description': 'EXTEL_EDGE Cartesian control-volume half-edge sum; boundary nodes incomplete'})
    ds['eta_tendency_from_transport'] = ('node', eta_from_transport, {'units': 'm s-1',
        'description': 'Implied d(eta)/dt=-nodal_flux/ART1, interior nodes only'})
    ds['is_interior_node'] = ('node', interior)
    ds.attrs.update({
        'operator': 'Cartesian FVCOM EXTEL_EDGE half-edge continuity',
        'boundary_caveat': 'OBC/land boundary nodes excluded; no external boundary flux included',
        'n_interior_nodes': int(interior.sum()),
        'max_abs_transport_eta_tendency_m_s': float(np.nanmax(np.abs(eta_from_transport))),
        'p95_abs_transport_eta_tendency_m_s': float(np.percentile(np.abs(eta_from_transport[interior]),95)),
    })
    if eta_tendency is not None:
        reference = _array(eta_tendency, 'eta_tendency', (nnode,))
        residual = np.full(nnode, np.nan)
        residual[interior] = reference[interior] - eta_from_transport[interior]
        ds['parent_eta_tendency'] = ('node', reference, {'units': 'm s-1'})
        ds['continuity_residual'] = ('node', residual, {'units': 'm s-1',
            'description': 'Parent d(eta)/dt + FVCOM discrete div(Du)'} )
        ds.attrs['p95_abs_continuity_residual_m_s'] = float(np.percentile(np.abs(residual[interior]),95))

    gx, gy = _pressure_gradient(fvcom_grid, zeta)
    ds['eta_gradient_x'] = ('nele', gx, {'units': 'm m-1'})
    ds['eta_gradient_y'] = ('nele', gy, {'units': 'm m-1'})
    f = _coriolis(fvcom_grid, coriolis, ncell)
    if f is not None:
        ax = -gravity*gx + f*va
        ay = -gravity*gy - f*ua
        ds['geostrophic_proxy_acceleration_x'] = ('nele', ax, {'units': 'm s-2'})
        ds['geostrophic_proxy_acceleration_y'] = ('nele', ay, {'units': 'm s-2'})
        ds['geostrophic_proxy_acceleration'] = ('nele', np.hypot(ax, ay), {
            'units': 'm s-2', 'description': 'Only -g grad(eta)+f(v,-u); NOT full momentum residual'})
        ds.attrs['p95_geostrophic_proxy_acceleration_m_s2'] = float(np.percentile(np.hypot(ax,ay),95))
    else:
        ds.attrs['momentum_caveat'] = 'No valid latitudes or coriolis supplied; momentum proxy omitted'

    flux_obc, nedge = _open_boundary_volume_flux(fvcom_grid, t, depth, ua, va)
    ds.attrs['obc_outward_transport_m3_s'] = flux_obc
    ds.attrs['n_open_boundary_edges'] = int(nedge)

    if (u is None) != (v is None):
        raise ValueError('u and v must either both be provided or both omitted')
    if u is not None:
        u = np.asarray(u)
        v = np.asarray(v)
        if u.ndim != 2 or u.shape != v.shape or u.shape[1] != ncell:
            raise ValueError('u/v must have shape (siglay,nele)')
        _add_velocity_profile_checks(ds, fvcom_grid,
            lambda start, stop: (u[:,start:stop], v[:,start:stop]), u.shape[0],
            chunk_size=chunk_size)
    return ds


def diagnose_restart(
    fvcom_grid: FVCOM_grid, restartfile: str | Path, *, time_index=0,
    eta_tendency=None, coriolis=None, gravity=9.81, include_3d=True,
    chunk_size=50000,
) -> xr.Dataset:
    """Read an FVCOM restart NetCDF and diagnose it; never writes to file.

    ``eta_tendency`` must be an independently estimated parent-model SSH
    tendency on the FVCOM nodes, in m/s, at the *same time* as the restart.
    """
    from netCDF4 import Dataset

    with Dataset(restartfile, 'r') as nc:
        required = ('zeta','ua','va')
        missing = [name for name in required if name not in nc.variables]
        if missing:
            raise ValueError(f'Restart is missing: {", ".join(missing)}')
        ds = diagnose_initial_balance(
            fvcom_grid, zeta=nc['zeta'][time_index,:],
            ua=nc['ua'][time_index,:], va=nc['va'][time_index,:],
            eta_tendency=eta_tendency, coriolis=coriolis, gravity=gravity,
            chunk_size=chunk_size)
        if include_3d:
            missing = [name for name in ('u','v') if name not in nc.variables]
            if missing:
                raise ValueError(f'Restart is missing: {", ".join(missing)}')
            uvar, vvar = nc['u'], nc['v']
            if uvar.ndim != 3 or vvar.shape != uvar.shape or uvar.shape[2] != len(fvcom_grid.tri):
                raise ValueError('Restart u/v must have dimensions (time,siglay,nele)')
            _add_velocity_profile_checks(ds, fvcom_grid,
                lambda start, stop: (uvar[time_index,:,start:stop],
                                     vvar[time_index,:,start:stop]),
                uvar.shape[1], chunk_size=chunk_size)
        ds.attrs['source_restart'] = str(restartfile)
        ds.attrs['time_index'] = int(time_index)
    return ds


def finite_difference_eta_tendency(zeta_before, zeta_after, seconds):
    """Compute d(eta)/dt on FVCOM nodes from two parent snapshots [m/s].

    For an initial time use times bracketing the restart whenever possible.
    Both snapshots must already have been interpolated to the FVCOM nodes.
    """
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be positive and finite')
    before = np.asanyarray(zeta_before, dtype=np.float64)
    after = np.asanyarray(zeta_after, dtype=np.float64)
    if before.ndim != 1 or before.shape != after.shape:
        raise ValueError('zeta_before/after must have matching (node,) shape')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('zeta_before/after must be finite')
    return (after-before) / seconds
