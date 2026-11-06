"""Depth-average GLORYS east/north currents on their native horizontal grid.

GLORYS's distributed 50-level product is sampled at fixed z levels, not
NEMO's native layer interfaces. Integrate the piecewise-linear profile and
extend its first/last wet sample to the actual sea surface / seabed. This
integral is independent of any FVCOM sigma vertical discretization.
"""

from __future__ import annotations

import numpy as np


def _integrate_profile(data, levels, top, bottom):
    """Integrate each valid GLORYS column between ``top`` and ``bottom``.

    Parameters
    ----------
    data : ndarray (depth, points)
        NaNs are allowed *only* below the deepest valid z-level.
    levels : ndarray (depth,)
        Strictly increasing depths in metres, positive down.
    top, bottom : ndarray (points,)
        Depths below reference sea level, in metres.
    """
    valid = np.isfinite(data)
    if not valid[0].all():
        raise ValueError('GLORYS velocity missing at surface in a wet column')
    if np.any(~valid[:-1] & valid[1:]):
        raise ValueError('GLORYS velocity has interior missing levels')

    # Constant continuation beneath last valid sample, never zero filling.
    last_valid = np.maximum.accumulate(
        np.where(valid, np.arange(len(levels))[:, None], 0), axis=0)
    filled = np.take_along_axis(data, last_valid, axis=0)

    # Constant above first available sample (including elevated sea surface).
    area = filled[0] * np.maximum(0., np.minimum(bottom, levels[0]) - top)

    # Exact trapezoidal integration of the linearly interpolated z profile.
    for k in range(len(levels) - 1):
        lo = np.maximum(top, levels[k])
        hi = np.minimum(bottom, levels[k + 1])
        thickness = np.maximum(hi - lo, 0.)
        slope = (filled[k + 1] - filled[k]) / (levels[k + 1] - levels[k])
        a = filled[k] + (lo - levels[k]) * slope
        b = filled[k] + (hi - levels[k]) * slope
        area += 0.5 * (a + b) * thickness

    # Constant continuation between last GLORYS depth and seabed.
    area += filled[-1] * np.maximum(bottom - np.maximum(top, levels[-1]), 0.)
    return area / (bottom - top)


def depth_average_glorys(uo, vo, depth, deptho, zos, *, wet_mask=None):
    """Compute barotropic east/north velocities at native GLORYS grid points.

    ``uo, vo``: (depth, latitude, longitude), metres/second.
    ``depth``: (depth,), positive-down GLORYS standard z levels, metres.
    ``deptho``: (latitude, longitude), positive-down seabed depth, metres.
    ``zos``: (latitude, longitude), positive-up sea surface height, metres.

    The water column spans from ``-zos`` to ``deptho`` in positive-down
    coordinates. No FVCOM sigma levels are used. Undefined land points remain
    NaN; invalid source profiles at selected wet points raise ValueError.
    """
    levels = np.asarray(depth, dtype=np.float64)
    if (levels.ndim != 1 or not len(levels) or
            not np.isfinite(levels).all() or levels[0] < 0 or
            np.any(np.diff(levels) <= 0)):
        raise ValueError('GLORYS depth levels must increase from the surface')

    seabed = np.ma.filled(np.ma.asarray(deptho, dtype=np.float64), np.nan)
    surface = np.ma.filled(np.ma.asarray(zos, dtype=np.float64), np.nan)
    u = np.ma.filled(np.ma.asarray(uo, dtype=np.float64), np.nan)
    v = np.ma.filled(np.ma.asarray(vo, dtype=np.float64), np.nan)
    if (seabed.ndim != 2 or surface.shape != seabed.shape or
            u.shape != (len(levels), *seabed.shape) or v.shape != u.shape):
        raise ValueError('Incompatible GLORYS velocity/deptho/zos dimensions')

    if wet_mask is None:
        wet = np.isfinite(seabed) & (seabed > 0.)
    else:
        wet = np.asarray(wet_mask, dtype=bool)
        if wet.shape != seabed.shape:
            raise ValueError('GLORYS wet mask has wrong dimensions')

    result_u = np.full(seabed.shape, np.nan, dtype=np.float64)
    result_v = np.full(seabed.shape, np.nan, dtype=np.float64)
    if not wet.any():
        raise ValueError('No selected GLORYS wet grid points')

    h = seabed[wet]
    s = surface[wet]
    if (not np.isfinite(h).all() or not np.isfinite(s).all() or
            np.any(h <= 0) or np.any(h + s <= 0)):
        raise ValueError('GLORYS wet columns have invalid deptho or zos')

    top = -s
    result_u[wet] = _integrate_profile(u[:, wet], levels, top, h)
    result_v[wet] = _integrate_profile(v[:, wet], levels, top, h)
    if not (np.isfinite(result_u[wet]).all() and np.isfinite(result_v[wet]).all()):
        raise ValueError('Nonfinite GLORYS barotropic velocities')
    return result_u, result_v


def fvcom_target_depths(sigma, depth, zeta):
    """FVCOM sigma midpoint depths relative to the GLORYS z-level datum.

    The FVCOM vertical coordinate is ``z = zeta + sigma * (h + zeta)``
    (sigma is negative, surface first). GLORYS depth is positive down,
    measured from mean sea level. Above-datum targets are clipped to zero
    because there are no GLORYS z-levels above the surface reference.

    Parameters
    ----------
    sigma : ndarray (point, siglay)
    depth, zeta : ndarray (point,)

    Returns
    -------
    ndarray (siglay, point), positive-down reference depths.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    h = np.asarray(depth, dtype=np.float64)
    z = np.asarray(zeta, dtype=np.float64)
    if sigma.ndim != 2 or h.ndim != 1 or sigma.shape[0] != len(h) or h.shape != z.shape:
        raise ValueError('FVCOM sigma, h and zeta have incompatible shapes')
    if not (np.isfinite(sigma).all() and np.isfinite(h).all() and np.isfinite(z).all()):
        raise ValueError('Nonfinite FVCOM sigma, h or zeta')
    if np.any(h <= 0) or np.any(h + z <= 0):
        raise ValueError('FVCOM contains dry or non-positive water columns')
    return np.maximum(0., -z[None, :] + np.abs(sigma).T * (h + z)[None, :])


def correct_fvcom_barotropic(u, v, ua, va, siglev_center):
    """Match the sigma-weighted 3-D velocity means to GLORYS ua/va.

    Parameters
    ----------
    u, v : array_like, shape (siglay, nele)
        Vertically interpolated velocities at FVCOM element centers.
    ua, va : array_like, shape (nele,)
        GLORYS-depth-integrated barotropic currents, already interpolated
        horizontally to FVCOM elements. All four fields must be expressed
        in the same horizontal coordinate system (after any rotation).
    siglev_center : array_like, shape (nele, siglay+1)
        FVCOM sigma interfaces, surface first, decreasing from 0 to -1.
        The layer-thickness weights may vary between elements.

    Returns
    -------
    u_corrected, v_corrected : float64 arrays, shape (siglay, nele)
        ``u - depth_mean(u) + ua`` and equivalent for ``v``. The same
        offset is applied to every layer within each element; vertical
        shear is therefore unchanged.
    """
    uu = np.ma.filled(np.ma.asarray(u, dtype=np.float64), np.nan)
    vv = np.ma.filled(np.ma.asarray(v, dtype=np.float64), np.nan)
    ubar = np.ma.filled(np.ma.asarray(ua, dtype=np.float64), np.nan)
    vbar = np.ma.filled(np.ma.asarray(va, dtype=np.float64), np.nan)
    interfaces = np.ma.filled(
        np.ma.asarray(siglev_center, dtype=np.float64), np.nan)

    if uu.ndim != 2 or vv.shape != uu.shape:
        raise ValueError('FVCOM u and v must have shape (siglay, nele)')
    nlayer, ncell = uu.shape
    if (ubar.shape != (ncell,) or vbar.shape != (ncell,) or
            interfaces.shape != (ncell, nlayer + 1)):
        raise ValueError('Incompatible barotropic velocities or sigma interface dimensions')
    if not all(np.isfinite(a).all() for a in (uu, vv, ubar, vbar, interfaces)):
        raise ValueError('Nonfinite FVCOM velocities or sigma interfaces')

    layer_thickness = -np.diff(interfaces, axis=1).T  # (siglay, nele)
    if np.any(layer_thickness <= 0):
        raise ValueError('FVCOM sigma interfaces must decrease from surface to bottom')
    weights = layer_thickness / layer_thickness.sum(axis=0, keepdims=True)
    fv_ubar = np.sum(uu * weights, axis=0)
    fv_vbar = np.sum(vv * weights, axis=0)
    return (uu + (ubar - fv_ubar)[None, :],
            vv + (vbar - fv_vbar)[None, :])
