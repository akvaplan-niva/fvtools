"""Constrained, read-only-in-memory FVCOM restart continuity projection.

The matrix reproduces the Cartesian half-edge mass-flux sum in FVCOM's
EXTEL_EDGE (excluding separately treated boundary nodes). Boundary *cells*
are fixed, so open/closed boundary edge normal transports are unchanged.

The projection minimizes the barotropic kinetic-energy norm of the velocity
correction, subject to matching a supplied, independent parent-model eta
trend at interior nodes. It does not enforce barotropic momentum balance or
continuity at OBC nodes; it is an initialization experiment, not a solver
for FVCOM's full coupled boundary conditions.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

import numpy as np
import xarray as xr
from scipy import sparse
from scipy.sparse.linalg import lsmr

from fvtools.setup.initial._helpers.restart_io import publish_staged_restart


def _vector(value, shape, label):
    result = np.asarray(np.ma.filled(np.ma.asarray(value, dtype=np.float64), np.nan))
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f'{label} must be finite with shape {shape}; got {result.shape}')
    return result


def _half_edge_matrix(grid):
    """Return node-by-cell-transport operator and CV metrics.

    B @ [D*ua, D*va] reproduces the nodal XFLUX contributions summed by
    EXTEL_EDGE before OBC handling and other source/sink terms.
    """
    triangle = np.asarray(grid.tri, dtype=np.int64)
    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    if triangle.ndim != 2 or triangle.shape[1] != 3 or triangle.size == 0:
        raise ValueError('FVCOM_grid.tri must be a nonempty (nele,3) array')
    nnode, ncell = x.size, len(triangle)
    if (x.shape != y.shape or not np.isfinite(x).all() or
            not np.isfinite(y).all() or triangle.min() < 0 or
            triangle.max() >= nnode):
        raise ValueError('Invalid projected mesh coordinates or node indices')

    t = grid.T
    ncv = int(t.NCV)
    cell = np.asarray(t.NTRG, dtype=np.int64)[:ncv]
    node = np.asarray(t.NIEC, dtype=np.int64)[:ncv]
    dx = np.asarray(t.DLTXE, dtype=np.float64)[:ncv]
    dy = np.asarray(t.DLTYE, dtype=np.float64)[:ncv]
    area = _vector(t.art1, (nnode,), 'FVCOM_grid.T.art1')
    kind = np.asarray(t.ISONB, dtype=np.int64)
    neighbour = np.asarray(t.NBE, dtype=np.int64)
    if (kind.shape != (nnode,) or neighbour.shape != (ncell, 3)
            or cell.shape != (ncv,) or node.shape != (ncv, 2)
            or dx.shape != (ncv,) or dy.shape != (ncv,)
            or not np.isfinite(dx).all() or not np.isfinite(dy).all()
            or np.any(area <= 0)
            or np.any(cell < 0) or np.any(cell >= ncell)
            or np.any(node < 0) or np.any(node >= nnode)):
        raise ValueError('Invalid FVCOM_grid.T half-edge/CV metrics')

    # EXFLUX = D * (-ua * DLTYE + va * DLTXE)
    # XFLUX(IA) -= EXFLUX; XFLUX(IB) += EXFLUX.
    cols = np.column_stack((2 * cell, 2 * cell + 1)).ravel()
    left_data = np.column_stack((dy, -dx)).ravel()
    right_data = -left_data
    rows = np.concatenate((np.repeat(node[:, 0], 2), np.repeat(node[:, 1], 2)))
    columns = np.concatenate((cols, cols))
    values = np.concatenate((left_data, right_data))
    operator = sparse.coo_matrix((values, (rows, columns)),
                                 shape=(nnode, 2 * ncell)).tocsr()
    operator.sum_duplicates()

    # The exact triangle area in metric coordinates, used in the kinetic
    # energy norm A*D*|delta_u|^2 (common density factor omitted).
    xx = x[triangle]
    yy = y[triangle]
    det = ((xx[:, 1] - xx[:, 0]) * (yy[:, 2] - yy[:, 0])
           - (xx[:, 2] - xx[:, 0]) * (yy[:, 1] - yy[:, 0]))
    cell_area = 0.5 * np.abs(det)
    if np.any(cell_area <= 0) or not np.isfinite(cell_area).all():
        raise ValueError('Degenerate triangles in projected mesh')
    return operator, area, kind, neighbour, cell_area


def _weighted_rms(residual, area):
    """Nodal-CV-area-weighted RMS, units unchanged."""
    return float(np.sqrt(np.sum(area * residual**2) / np.sum(area)))


def project_initial_continuity(
    FVCOM_grid, *, zeta, ua, va, eta_tendency,
    max_velocity_correction=0.25, relative_residual_tolerance=0.02,
    absolute_residual_tolerance=1e-8, max_iterations=4000,
):
    """Minimal-energy initial velocity projection constrained at interior nodes.

    Parameters
    ----------
    FVCOM_grid : FVCOM_grid
        Full, projected, uncropped grid with TGE/CV metrics.
    zeta : (node,) [m]
    ua, va : (nele,) [m/s]
    eta_tendency : (node,) [m/s]
        Independent target from the parent model, ideally centered at the
        restart time. A one-hour forward difference is a time-average,
        not an instantaneous tendency.
    max_velocity_correction : float [m/s]
        Safety limit, checked on the *actual* final velocity correction.
        Never silently clip: clipping would violate the continuity constraint.
    relative_residual_tolerance : float
        Maximum corrected / original CV-area-weighted RMS of interior
        continuity residual, except near zero original error.
    absolute_residual_tolerance : float [m/s]
        Absolute alternative RMS acceptance threshold.

    Notes
    -----
    Minimize sum(A_cell * D_cell * |delta_u|**2) (density constant omitted)
    subject to d(eta)/dt = -B[D_cell * (u + delta_u)]/A_node
    for INTERIOR nodes only. All cells with ANY exterior mesh edge are fixed;
    this preserves their open/land boundary normal transports. If the
    discrete target is unreachable under these restrictions, raise ValueError.

    Returns a diagnostic xarray.Dataset including corrected ua/va, the
    velocity increments, masks and before/after continuity residuals.
    No input arrays or files are modified.
    """
    if (not np.isfinite(max_velocity_correction) or max_velocity_correction <= 0
            or not np.isfinite(relative_residual_tolerance)
            or not 0 < relative_residual_tolerance < 1
            or not np.isfinite(absolute_residual_tolerance)
            or absolute_residual_tolerance <= 0
            or not isinstance(max_iterations, (int, np.integer))
            or max_iterations < 1):
        raise ValueError('Invalid projection safety limits / iteration count')

    B, area, kind, neighbour, cell_area = _half_edge_matrix(FVCOM_grid)
    nnode, ncell = B.shape[0], B.shape[1] // 2
    triangle = np.asarray(FVCOM_grid.tri, dtype=np.int64)
    h = _vector(FVCOM_grid.h, (nnode,), 'grid.h')
    zeta = _vector(zeta, (nnode,), 'zeta')
    ua = _vector(ua, (ncell,), 'ua')
    va = _vector(va, (ncell,), 'va')
    target = _vector(eta_tendency, (nnode,), 'eta_tendency')

    if np.any(h + zeta <= 0):
        raise ValueError('Dry/negative initial water-column depth')
    depth = np.mean((h + zeta)[triangle], axis=1)
    interior = kind == 0
    fixed_cell = np.any(neighbour < 0, axis=1)
    if not np.any(interior):
        raise ValueError('No interior nodes for a constrained continuity projection')
    if not np.any(~fixed_cell):
        raise ValueError('No free interior cells; boundary cells are fixed')

    uv = np.column_stack((ua, va)).ravel()
    D2 = np.repeat(depth, 2)
    initial_eta = -(B @ (D2 * uv)) / area
    before = target[interior] - initial_eta[interior]
    old_rms = _weighted_rms(before, area[interior])
    residual_limit = max(absolute_residual_tolerance,
                         relative_residual_tolerance * old_rms)

    correction = np.zeros(2 * ncell, dtype=np.float64)
    if old_rms > residual_limit:
        # Minimize A_cell*D_cell*(du^2+dv^2). Median normalization has
        # no effect on the minimizer; it improves numerical conditioning.
        mass = cell_area * depth
        free = np.repeat(~fixed_cell, 2)
        mobility = np.repeat(np.sqrt(np.median(mass[~fixed_cell]) /
                                         mass[~fixed_cell]), 2)
        Bfree = B[interior, :][:, free]
        factors = D2[free] * mobility
        # -A_node^-1 B (D * delta_velocity) * mobility = mismatch
        G = sparse.diags(-1.0 / area[interior]) @ Bfree @ sparse.diags(factors)
        solution = lsmr(G, before, atol=1e-11, btol=1e-11,
                        maxiter=int(max_iterations))
        correction[free] = mobility * solution[0]

    du = correction[0::2]
    dv = correction[1::2]
    speed = np.hypot(du, dv)
    largest = float(np.max(speed))
    if largest > max_velocity_correction:
        raise ValueError(
            f'Continuity correction exceeds safety limit: {largest:.6g} m/s '
            f'> {max_velocity_correction:.6g} m/s. No restart written.')

    projected = uv + correction
    # Monitor (but do not claim to conserve) barotropic kinetic energy.
    # Density is omitted from both totals, so their ratio is physical.
    cell_mass_per_density = cell_area * depth
    energy_before = 0.5 * np.sum(cell_mass_per_density * (ua**2 + va**2))
    energy_after = 0.5 * np.sum(cell_mass_per_density *
                                (projected[0::2]**2 + projected[1::2]**2))
    energy_change_pct = (100.0 * (energy_after - energy_before) / energy_before
                         if energy_before > 0 else np.nan)
    after_eta = -(B @ (D2 * projected)) / area
    after = target[interior] - after_eta[interior]
    new_rms = _weighted_rms(after, area[interior])
    if new_rms > residual_limit * (1 + 1e-8):
        raise ValueError(
            f'Continuity target not achieved with boundary cells fixed: '
            f'RMS {new_rms:.6g} m/s (initial {old_rms:.6g}, '
            f'allowed {residual_limit:.6g}); no restart written.')

    err_before = np.full(nnode, np.nan)
    err_after = np.full(nnode, np.nan)
    implied_before = np.full(nnode, np.nan)
    implied_after = np.full(nnode, np.nan)
    err_before[interior], err_after[interior] = before, after
    implied_before[interior] = initial_eta[interior]
    implied_after[interior] = after_eta[interior]
    ds = xr.Dataset(
        data_vars={
            'corrected_ua': ('nele', projected[0::2], {'units': 'm s-1'}),
            'corrected_va': ('nele', projected[1::2], {'units': 'm s-1'}),
            'delta_ua': ('nele', du, {'units': 'm s-1'}),
            'delta_va': ('nele', dv, {'units': 'm s-1'}),
            'velocity_correction_speed': ('nele', speed, {'units': 'm s-1'}),
            'fixed_boundary_cell': ('nele', fixed_cell),
            'interior_node': ('node', interior),
            'parent_eta_tendency': ('node', target, {'units': 'm s-1'}),
            'initial_eta_tendency': ('node', implied_before, {'units': 'm s-1'}),
            'corrected_eta_tendency': ('node', implied_after, {'units': 'm s-1'}),
            'initial_continuity_residual': ('node', err_before, {'units': 'm s-1'}),
            'corrected_continuity_residual': ('node', err_after, {'units': 'm s-1'}),
        },
        attrs={
            'method': 'Constrained minimal barotropic-kinetic-energy correction',
            'operator': 'FVCOM Cartesian EXTEL_EDGE half-edge CV divergence',
            'boundary_rule': 'ALL exterior-edge cells fixed; boundary nodes unconstrained',
            'continuity_caveat': 'Interior nodal equation only; no OBC node continuity or momentum balance enforced',
            'n_fixed_boundary_cells': int(fixed_cell.sum()),
            'n_interior_nodes': int(interior.sum()),
            'rms_residual_before_m_s': old_rms,
            'rms_residual_after_m_s': new_rms,
            'p95_abs_residual_before_m_s': float(np.percentile(np.abs(before), 95)),
            'p95_abs_residual_after_m_s': float(np.percentile(np.abs(after), 95)),
            'max_velocity_correction_m_s': largest,
            'p95_velocity_correction_m_s': float(np.percentile(speed, 95)),
            'barotropic_kinetic_energy_change_pct': float(energy_change_pct),
            'delta_transport_momentum_proxy_x': float(np.sum(cell_mass_per_density * du)),
            'delta_transport_momentum_proxy_y': float(np.sum(cell_mass_per_density * dv)),
        },
    )
    return ds


def _check_3d_mean(grid, nc, time_index, ua, va, chunk_size, max_error):
    """Refuse to silently preserve an already inconsistent 2D/3D restart."""
    ncell = len(ua)
    tri = np.asarray(grid.tri, dtype=np.int64)
    sigma = np.asarray(grid.siglev, dtype=np.float64)
    uvar, vvar = nc['u'], nc['v']
    if uvar.ndim != 3 or vvar.shape != uvar.shape or uvar.shape[2] != ncell:
        raise ValueError('Restart u/v must have shape (time, siglay, nele)')
    if sigma.shape != (len(grid.x), uvar.shape[1] + 1):
        raise ValueError('FVCOM_grid.siglev dimensions do not match restart u/v')
    worst = 0.0
    for start in range(0, ncell, chunk_size):
        stop = min(start + chunk_size, ncell)
        u = _vector(uvar[time_index, :, start:stop],
                    (uvar.shape[1], stop-start), 'u')
        v = _vector(vvar[time_index, :, start:stop],
                    (vvar.shape[1], stop-start), 'v')
        w = -np.diff(np.mean(sigma[tri[start:stop]], axis=1), axis=1)
        if (not np.isfinite(w).all() or np.any(w <= 0)
                or not np.allclose(w.sum(axis=1), 1.0, atol=2e-4, rtol=0)):
            raise ValueError('Invalid FVCOM sigma-layer interfaces')
        ub = np.sum(u.T * w, axis=1)
        vb = np.sum(v.T * w, axis=1)
        worst = max(worst, float(np.max(np.hypot(ub-ua[start:stop],
                                               vb-va[start:stop]))))
    if worst > max_error:
        raise ValueError(
            f'Initial 2D/3D velocity mismatch {worst:.6g} m/s exceeds '
            f'{max_error:.6g} m/s; refusing to hide pre-existing errors')
    return worst


def select_time_indices(times, target, count=3):
    """Indices of the nearest distinct times, returned chronologically."""
    values = np.asarray(
        np.ma.filled(np.ma.asarray(times, dtype=np.float64), np.nan))
    try:
        target = float(target)
    except (TypeError, ValueError) as exc:
        raise ValueError('target must be a finite numeric time') from exc
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Parent times must be a nonempty finite 1-D array')
    if (not np.isfinite(target) or not isinstance(count, (int, np.integer))
            or count < 2):
        raise ValueError('A finite target and integer count >= 2 are required')

    selected, seen = [], set()
    for index in np.argsort(np.abs(values - target), kind='stable'):
        value = float(values[index])
        if value in seen:
            continue
        seen.add(value)
        selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(
            f'Continuity correction needs {count} distinct parent times; '
            f'only {len(selected)} are available')
    return np.asarray(sorted(selected, key=lambda i: values[i]), dtype=np.int64)


def quadratic_eta_tendency(times_seconds, zeta_samples, target_seconds=0.0):
    """Differentiate a quadratic interpolant through three parent SSH samples.

    ``times_seconds`` is relative to any epoch and ``zeta_samples`` has shape
    ``(3, node)``. Unequal intervals and one-sided endpoint stencils work.
    """
    times = np.asarray(
        np.ma.filled(np.ma.asarray(times_seconds, dtype=np.float64), np.nan))
    zeta = np.asarray(
        np.ma.filled(np.ma.asarray(zeta_samples, dtype=np.float64), np.nan))
    try:
        target_seconds = float(target_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError('target_seconds must be finite') from exc
    if times.shape != (3,) or zeta.ndim != 2 or zeta.shape[0] != 3:
        raise ValueError('Expected three times and zeta_samples with shape (3, node)')
    if (not np.isfinite(times).all() or not np.isfinite(target_seconds)
            or not np.isfinite(zeta).all()):
        raise ValueError('Parent times and zeta samples must be finite')
    x = times - target_seconds
    if len(np.unique(x)) != 3:
        raise ValueError('Parent timestamps must be distinct')
    derivative_weights = np.array([
        -(x[1] + x[2]) / ((x[0] - x[1]) * (x[0] - x[2])),
        -(x[0] + x[2]) / ((x[1] - x[0]) * (x[1] - x[2])),
        -(x[0] + x[1]) / ((x[2] - x[0]) * (x[2] - x[1])),
    ])
    tendency = derivative_weights @ zeta
    if not np.isfinite(tendency).all():
        raise ValueError('Computed parent eta tendency is nonfinite')
    return tendency


def apply_initial_continuity(
    nc, FVCOM_grid, eta_tendency, *, time_index=0,
    max_velocity_correction=0.25, relative_residual_tolerance=0.02,
    absolute_residual_tolerance=1e-8, max_2d_3d_mismatch=1e-5,
    max_iterations=4000, chunk_size=50000,
):
    """Apply the projection to an already-open, staged FVCOM restart.

    The caller owns atomic staging/publication. The same horizontal increment
    is added to every sigma layer, preserving vertical shear.
    """
    from fvtools.setup.initial._helpers.fields import validate_restart_fields

    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if (not isinstance(time_index, (int, np.integer))
            or 'time' not in nc.dimensions
            or not 0 <= time_index < len(nc.dimensions['time'])):
        raise IndexError('Restart time_index is out of range')
    if not np.isfinite(max_2d_3d_mismatch) or max_2d_3d_mismatch < 0:
        raise ValueError('max_2d_3d_mismatch must be finite and non-negative')

    validate_restart_fields(nc, ('zeta', 'ua', 'va', 'u', 'v'))
    nnode, ncell = len(FVCOM_grid.x), len(FVCOM_grid.tri)
    if (nc['zeta'].shape != (len(nc.dimensions['time']), nnode)
            or nc['ua'].shape != (len(nc.dimensions['time']), ncell)
            or nc['va'].shape != (len(nc.dimensions['time']), ncell)):
        raise ValueError('Restart and FVCOM grid node/element dimensions do not match')

    zeta = _vector(nc['zeta'][time_index, :], (nnode,), 'restart zeta')
    ua = _vector(nc['ua'][time_index, :], (ncell,), 'restart ua')
    va = _vector(nc['va'][time_index, :], (ncell,), 'restart va')
    original_error = _check_3d_mean(
        FVCOM_grid, nc, time_index, ua, va, int(chunk_size),
        max_2d_3d_mismatch)
    report = project_initial_continuity(
        FVCOM_grid, zeta=zeta, ua=ua, va=va, eta_tendency=eta_tendency,
        max_velocity_correction=max_velocity_correction,
        relative_residual_tolerance=relative_residual_tolerance,
        absolute_residual_tolerance=absolute_residual_tolerance,
        max_iterations=max_iterations,
    )
    report.attrs['max_original_2d_3d_mismatch_m_s'] = original_error
    du = np.asarray(report['delta_ua'].values)
    dv = np.asarray(report['delta_va'].values)

    # Projection and all safety checks are complete before the first write.
    nc['ua'][time_index, :] = report['corrected_ua'].values
    nc['va'][time_index, :] = report['corrected_va'].values
    for start in range(0, ncell, int(chunk_size)):
        stop = min(start + int(chunk_size), ncell)
        for name, increment in (('u', du), ('v', dv)):
            profile = _vector(
                nc[name][time_index, :, start:stop],
                (nc[name].shape[1], stop - start), name)
            nc[name][time_index, :, start:stop] = (
                profile + increment[None, start:stop])

    corrected_ua = _vector(nc['ua'][time_index, :], (ncell,), 'written ua')
    corrected_va = _vector(nc['va'][time_index, :], (ncell,), 'written va')
    corrected_error = _check_3d_mean(
        FVCOM_grid, nc, time_index, corrected_ua, corrected_va,
        int(chunk_size), max_2d_3d_mismatch)
    nc.setncattr(
        'initial_continuity_projection',
        'Constrained kinetic-energy minimum; all exterior-edge cells fixed')
    nc.setncattr('initial_continuity_rms_before_m_s',
                 float(report.attrs['rms_residual_before_m_s']))
    nc.setncattr('initial_continuity_rms_after_m_s',
                 float(report.attrs['rms_residual_after_m_s']))
    nc.setncattr('initial_continuity_max_velocity_correction_m_s',
                 float(report.attrs['max_velocity_correction_m_s']))
    nc.setncattr('initial_continuity_max_original_2d_3d_mismatch_m_s',
                 float(original_error))
    nc.setncattr('initial_continuity_max_corrected_2d_3d_mismatch_m_s',
                 float(corrected_error))
    nc.sync()
    return report


def correct_restart_continuity(
    FVCOM_grid, restartfile, outputfile, *, eta_tendency, time_index=0,
    max_velocity_correction=0.25, relative_residual_tolerance=0.02,
    absolute_residual_tolerance=1e-8, max_2d_3d_mismatch=1e-5,
    max_iterations=4000, chunk_size=50000,
):
    """Write a NEW FVCOM restart, preserving 3D shear and OBC cell transports.

    Returns the diagnostic Dataset (never a file path). The source restart is
    never altered. The new file is published atomically only after all checks
    and NetCDF writes finish. Refuses an existing output destination.
    """
    from netCDF4 import Dataset

    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    source, dest = Path(restartfile).resolve(), Path(outputfile).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == dest:
        raise ValueError('Output must differ from the source restart')
    if dest.exists():
        raise FileExistsError(dest)
    if not dest.parent.is_dir():
        raise FileNotFoundError(f'Output directory does not exist: {dest.parent}')

    with Dataset(source, 'r') as nc:
        required = ('zeta', 'ua', 'va', 'u', 'v')
        missing = [name for name in required if name not in nc.variables]
        if missing:
            raise ValueError(f'Restart missing required velocity fields: {missing}')
        ncell, nnode = len(FVCOM_grid.tri), len(FVCOM_grid.x)
        if nc['zeta'].shape[1:] != (nnode,) or nc['ua'].shape[1:] != (ncell,) or nc['va'].shape[1:] != (ncell,):
            raise ValueError('Restart and FVCOM grid node/element dimensions do not match')
        zeta = _vector(nc['zeta'][time_index, :], (nnode,), 'restart zeta')
        ua = _vector(nc['ua'][time_index, :], (ncell,), 'restart ua')
        va = _vector(nc['va'][time_index, :], (ncell,), 'restart va')
        original_error = _check_3d_mean(FVCOM_grid, nc, time_index, ua, va,
                                         chunk_size, max_2d_3d_mismatch)

    ds = project_initial_continuity(
        FVCOM_grid, zeta=zeta, ua=ua, va=va,
        eta_tendency=eta_tendency,
        max_velocity_correction=max_velocity_correction,
        relative_residual_tolerance=relative_residual_tolerance,
        absolute_residual_tolerance=absolute_residual_tolerance,
        max_iterations=max_iterations,
    )
    ds.attrs['max_original_2d_3d_mismatch_m_s'] = original_error
    du = np.asarray(ds['delta_ua'].values)
    dv = np.asarray(ds['delta_va'].values)

    with TemporaryDirectory(prefix='.restart_continuity_', dir=dest.parent) as workdir:
        staged = Path(workdir) / dest.name
        shutil.copy2(source, staged)
        with Dataset(staged, 'r+') as nc:
            nc['ua'][time_index, :] = ds['corrected_ua'].values
            nc['va'][time_index, :] = ds['corrected_va'].values
            for start in range(0, len(du), chunk_size):
                stop = min(start + chunk_size, len(du))
                for key, increments in (('u', du), ('v', dv)):
                    # Same correction at every sigma level: 3D shear preserved.
                    data = _vector(nc[key][time_index, :, start:stop],
                                   (nc[key].shape[1], stop-start), key)
                    nc[key][time_index, :, start:stop] = (
                        data + increments[None, start:stop])
            nc.setncattr('initial_continuity_projection',
                         'Constrained kinetic-energy minimum; OBC/land edge cells fixed')
            nc.setncattr('initial_continuity_rms_before_m_s',
                         ds.attrs['rms_residual_before_m_s'])
            nc.setncattr('initial_continuity_rms_after_m_s',
                         ds.attrs['rms_residual_after_m_s'])
            nc.sync()
        publish_staged_restart(staged, dest)
    return ds
