
"""FVCOM sigma coordinates from MOD_SETUP (including the modified Souza method).

All output arrays are surface-first. The nodal coordinates ``z`` are
*dimensionless* and have shape ``(n_nodes, n_layers + 1)``.
"""

from __future__ import annotations

import numpy as np


def _validate_inputs(depth, n_layers: int, dtype) -> tuple[np.ndarray, int, np.dtype]:
    """Validate shared inputs and return a 1D depth array, n_layers and dtype."""
    if (isinstance(n_layers, (bool, np.bool_))
            or not isinstance(n_layers, (int, np.integer))
            or n_layers < 2):
        raise ValueError("n_layers must be an integer >= 2")

    dt = np.dtype(dtype)
    if dt not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be np.float32 or np.float64")

    h = np.atleast_1d(np.asarray(depth, dtype=dt))
    if h.ndim != 1 or not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("depth must be a positive scalar or a 1D array of positive depths")
    return h, int(n_layers), dt


def _geometric_profile(n: int, p_sigma: float, dtype: np.dtype) -> np.ndarray:
    """Compute the 1D FVCOM UNIFORM/GEOMETRIC interface profile."""
    kb = n + 1
    if not np.isfinite(p_sigma) or p_sigma <= 0:
        raise ValueError("p_sigma must be positive")
    if p_sigma > 1 and kb % 2 == 0:
        raise ValueError("FVCOM requires an odd number of vertical levels (KB) when p_sigma > 1")

    k = np.arange(kb, dtype=dtype)
    if p_sigma == 1:
        return -(k / n)**p_sigma

    mid = (kb + 1) // 2  # Equivalent to Fortran integer division.
    denominator = mid - 1
    s = np.empty(kb, dtype=dtype)
    s[:mid] = -0.5 * (k[:mid] / denominator)**p_sigma
    s[mid:] = 0.5 * ((n - k[mid:]) / denominator)**p_sigma - 1
    return s


def _tanh_profile(n: int, dl2: float, du2: float, dtype: np.dtype) -> np.ndarray:
    """Compute the 1D SIGMA_TANH profile, including the top boundary."""
    dl = dtype.type(dl2)
    du = dtype.type(du2)
    denominator = np.tanh(dl) + np.tanh(du)
    if abs(float(denominator)) < 1e-14:
        raise ValueError("tanh(dl2) + tanh(du2) must not be zero")

    k = np.arange(1, n + 1, dtype=dtype)
    x = np.tanh((dl + du) * (n - k) / n - dl)
    s = np.empty(n + 1, dtype=dtype)
    s[0] = 0
    s[1:] = (x + np.tanh(dl)) / denominator - 1
    return s


def _generalized_profile(
    h: np.ndarray,
    n: int,
    dtype: np.dtype,
    hmin1: float | None,
    duu: float | None,
    dll: float | None,
    ku: int | None,
    kl: int | None,
    zku,
    zkl,
) -> np.ndarray:
    """Compute SIGMA_GENERALIZED for shallow and deep nodes."""
    if any(v is None for v in (hmin1, duu, dll, ku, kl)):
        raise ValueError("generalized requires hmin1, duu, dll, ku, and kl")
    if (not isinstance(ku, (int, np.integer))
            or not isinstance(kl, (int, np.integer))
            or ku < 0 or kl < 0 or ku + kl >= n):
        raise ValueError("ku and kl must be non-negative integers with ku + kl < n_layers")

    upper = np.asarray([] if zku is None else zku, dtype=dtype)
    lower = np.asarray([] if zkl is None else zkl, dtype=dtype)
    if upper.shape != (ku,) or lower.shape != (kl,):
        raise ValueError("zku and zkl must contain ku and kl elements, respectively")

    z = np.zeros((h.size, n + 1), dtype=dtype)
    shallow = h < hmin1
    if np.any(shallow):
        # Same hardcoded shallow-water fallback as SIGMA_GENERALIZED.
        z[shallow] = _tanh_profile(n, 0.001, 0.001, dtype)

    if np.any(~shallow):
        hd = h[~shallow]
        zd = z[~shallow].copy()
        dr = (hd - duu - dll) / hd / (n - ku - kl)
        for k in range(1, n + 1):
            if k <= ku:
                zd[:, k] = zd[:, k - 1] - upper[k - 1] / hd
            elif k <= n - kl:
                zd[:, k] = zd[:, k - 1] - dr
            else:
                zd[:, k] = zd[:, k - 1] - lower[k - (n - kl) - 1] / hd
        z[~shallow] = zd
    return z


def _souza_stretching(sigma: np.ndarray, theta_s: float, theta_b: float) -> np.ndarray:
    """Apply surface and bottom stretching (ROMS Vstretching=5)."""
    if theta_s > 0:
        cs = -(np.sinh(0.5 * theta_s * sigma)
               / np.sinh(0.5 * theta_s))**2
    else:
        cs = -sigma**2

    if theta_b > 0:
        cs = (np.exp(0.5 * theta_b * (cs + 1))
              * np.sinh(0.5 * theta_b * cs) / np.sinh(0.5 * theta_b))
    return cs


def _souza_profile(
    h: np.ndarray,
    n: int,
    dtype: np.dtype,
    theta_s: float | None,
    theta_b: float | None,
    tcline: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute nodal interfaces Z, quadratic sigma_w, and stretched cs_w."""
    if any(v is None for v in (theta_s, theta_b, tcline)):
        raise ValueError("souza requires theta_s, theta_b, and tcline")
    if not (0 <= theta_s <= 10 and 0 <= theta_b <= 4):
        raise ValueError("theta_s must be in [0, 10] and theta_b must be in [0, 4]")
    if not np.isfinite(tcline) or tcline <= 0:
        raise ValueError("tcline must be positive")
    if not (8 <= n <= 199):
        raise ValueError("SIGMA_SOUZA requires 8 <= n_layers <= 199")

    # FVCOM K=1..KB, zero-based k=0..n, X=n-k and Q=k.
    q = np.arange(n + 1, dtype=dtype)
    x = n - q
    sigma_w = (-q * (q - 1) / (n * (n - 1))
               - dtype.type(0.01) * x * q / (n - 1))
    cs_w = _souza_stretching(sigma_w, theta_s, theta_b)

    # FVCOM stores dimensionless interfaces; this is Vtransform=2 at zeta=0.
    z = ((tcline * sigma_w[None, :] + h[:, None] * cs_w[None, :])
         / (tcline + h[:, None])).astype(dtype, copy=False)
    return z, sigma_w, cs_w


def _element_sigma(z: np.ndarray, nv) -> np.ndarray | None:
    """Average nodal sigma interfaces at the three triangle vertices (Z1)."""
    if nv is None:
        return None
    triangles = np.asarray(nv)
    if (triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)):
        raise ValueError("nv must be an integer array of shape (n_elements, 3)")
    triangles = triangles.astype(np.int64, copy=False)
    if np.any(triangles < 0) or np.any(triangles >= z.shape[0]):
        raise ValueError("nv contains node indices outside the depth array")
    return z[triangles].mean(axis=1)


def _fvcom_derivatives(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute DZ, ZZ and DZZ as in SETUP_SIGMA_DERIVATIVES."""
    dz = np.zeros_like(z)
    dz[:, :-1] = z[:, :-1] - z[:, 1:]

    zz = np.empty_like(z)
    zz[:, :-1] = 0.5 * (z[:, :-1] + z[:, 1:])
    zz[:, -1] = 2 * zz[:, -2] - zz[:, -3]

    dzz = np.zeros((z.shape[0], z.shape[1] - 1), dtype=z.dtype)
    dzz[:, :-1] = zz[:, :-2] - zz[:, 1:-1]
    return dz, zz, dzz


def setup_sigma(
    depth,
    n_layers: int,
    stype: str = "souza",
    *,
    nv=None,
    p_sigma: float = 1.0,
    dl2: float | None = None,
    du2: float | None = None,
    theta_s: float | None = None,
    theta_b: float | None = None,
    tcline: float | None = None,
    hmin1: float | None = None,
    duu: float | None = None,
    dll: float | None = None,
    ku: int | None = None,
    kl: int | None = None,
    zku=None,
    zkl=None,
    dtype=np.float64,
) -> dict[str, np.ndarray | None]:
    """Compute FVCOM sigma levels from ``mod_setup.F``.

    ``depth`` is positive bathymetry at mesh nodes (scalar or 1D array).
    ``n_layers`` is FVCOM KBM1, with KB = n_layers + 1.
    ``stype`` may be 'uniform', 'geometric', 'tanh', 'generalized', or 'souza'.
    ``nv`` is optional triangle connectivity, one-based unless
    ``nv_one_based=False``. Method-specific settings have the same names as
    in the Fortran source; for generalized, ``zku``/``zkl`` are physical
    top/bottom layer thicknesses in meters.

    Returns a dictionary with ``z``, ``z1``, ``dz``, ``zz``, ``dzz``,
    ``dz1``, ``zz1``, ``dzz1``, ``sigma_w``, ``cs_w``. All vertical columns
    run from surface (0) to bottom (-1). The ``z``/``z1`` arrays are
    dimensionless; ``zz`` is the FVCOM arithmetic layer midpoint, not the
    independent ROMS rho-point coordinate.
    """
    h, n, dt = _validate_inputs(depth, n_layers, dtype)
    method = stype.strip().lower()
    sigma_w = cs_w = None

    if method in ("uniform", "geometric"):
        s = _geometric_profile(n, p_sigma, dt)
        z = np.broadcast_to(s, (h.size, n + 1)).copy()
    elif method == "tanh":
        if dl2 is None or du2 is None:
            raise ValueError("tanh requires dl2 and du2")
        s = _tanh_profile(n, dl2, du2, dt)
        z = np.broadcast_to(s, (h.size, n + 1)).copy()
    elif method == "generalized":
        z = _generalized_profile(h, n, dt, hmin1, duu, dll, ku, kl, zku, zkl)
    elif method == "souza":
        z, _, s = _souza_profile(h, n, dt, theta_s, theta_b, tcline)
    else:
        raise ValueError("Unknown stype. Choose uniform, geometric, tanh, generalized, or souza")

    z1 = _element_sigma(z, nv)
    dz, zz, dzz = _fvcom_derivatives(z)
    dz1 = zz1 = dzz1 = None
    if z1 is not None:
        dz1, zz1, dzz1 = _fvcom_derivatives(z1)

    return {
        "z": z, "z1": z1, "s": s
    }