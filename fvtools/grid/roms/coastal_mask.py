"""Coastal validity and local, wet-corner interpolation for collocated ROMS fields.

Do not use a static sea/land mask as proof that a packed data variable is valid.
Preflight only the requested source crop once and reuse the resulting weights.
"""
import numpy as np


_SOURCE_FIELDS = {
    'zeta': False,
    'ubar_eastward': False,
    'vbar_northward': False,
    'salinity': True,
    'temperature': True,
    'u_eastward': True,
    'v_northward': True,
}


def invalid_values(values):
    """Treat masked values, NaN/inf and large unmasked NetCDF fills as missing."""
    a = np.ma.asarray(values)
    raw = np.asarray(np.ma.getdata(a))
    return np.ma.getmaskarray(a) | ~np.isfinite(raw) | (np.abs(raw) >= 1e30)


def probe_hindcast_wet_crop(dataset, window, static_land, bathymetry):
    """Wet mask shared by all 7 nesting inputs at a representative timestep.

    Only reads the spatial bounding rectangle already needed by FVCOM, not
    the full 1148 x 2747 NorKyst grid. NetCDF4 applies scale/offset/masks.
    The 3-D fields must be valid at ALL sigma levels. One source file is
    opened by the caller and no time-series reads or writes occur here.
    """
    static_land = np.asarray(static_land, dtype=bool)
    h = np.ma.asarray(bathymetry)
    wet = ~static_land & ~invalid_values(h) & (np.ma.filled(h, 0) > 0)
    for name, is_3d in _SOURCE_FIELDS.items():
        if name not in dataset.variables:
            raise ValueError(f'NorKyst hindcast lacks required field {name!r}')
        selector = (0, slice(None), *window) if is_3d else (0, *window)
        values = dataset.variables[name][selector]
        bad = invalid_values(values)
        if is_3d:
            if bad.ndim != 3:
                raise ValueError(f'Hindcast {name} has unexpected source dimensions')
            bad = np.any(bad, axis=0)
        if bad.shape != wet.shape:
            raise ValueError(f'Hindcast {name} shape {bad.shape} != source crop {wet.shape}')
        wet &= ~bad
    return wet


def repair_wet_corners(indices, coefficients, valid_points, *, coordinate):
    """Renormalize a *containing cell's* surviving wet corners only.

    No distant-neighbor interpolation or silent masked-array reductions.
    Invalid source indices are replaced by an already-valid corner with ZERO
    weight so downstream strict source checks remain enabled. If all four
    corners are invalid, fail before creating nesting output.
    """
    idx = np.asarray(indices)
    coef = np.asarray(coefficients)
    wet = np.asarray(valid_points, dtype=bool)
    if idx.ndim != 2 or idx.shape[1] != 4 or idx.shape != coef.shape:
        raise ValueError('Expected four ROMS source corners and coefficients')
    if np.any(idx < 0) or np.any(idx >= len(wet)):
        raise ValueError('ROMS source index outside available wet-mask range')
    if not np.isfinite(coef).all() or np.any(coef < -1e-12):
        raise ValueError('Invalid ROMS interpolation coefficients')
    invalid = ~wet[idx]
    affected = np.any(invalid, axis=1)
    count = int(affected.sum())
    if not count:
        return idx, coef, 0
    bad_idx = np.flatnonzero(affected)
    all_dry = np.all(invalid[bad_idx], axis=1)
    if np.any(all_dry):
        samples = bad_idx[all_dry][:8].tolist()
        raise ValueError(
            f'{coordinate}: {int(all_dry.sum())} FVCOM targets have no valid '
            f'NorKyst hindcast corner in their containing source cell; '
            f'target indices {samples}. Move the nesting zone into the '
            f'available hindcast ocean domain; no extrapolation was made.'
        )
    out_idx, out_coef = idx.copy(), coef.copy()
    rows = bad_idx
    valid_weights = np.where(~invalid[rows], coef[rows], 0.)
    total = valid_weights.sum(axis=1)
    if np.any(total <= 1e-10):
        samples = rows[total <= 1e-10][:8].tolist()
        raise ValueError(
            f'{coordinate}: target(s) {samples} only have zero-weight valid '
            'corners; moving the nest or allowing explicit extrapolation is required'
        )
    out_coef[rows] = valid_weights / total[:, None]
    # Select the first surviving wet corner of the same containing source cell.
    substitutes = idx[rows, np.argmax(~invalid[rows], axis=1)]
    out_idx[rows] = np.where(invalid[rows], substitutes[:, None], idx[rows])
    assert np.all(wet[out_idx])
    return out_idx, out_coef, count
