"""FVCOM sigma-to-z interpolation mixin; unchanged implementation."""

import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True)
def _interpolate_to_z_numba(f, z, depths):
    """One-pass column interpolation without full-grid temporary arrays.

    Preserve the original search semantics (including first-occurrence ties,
    masked/invalid levels, shallow extrapolation, and below-bottom NaNs).
    The output remains float64, even for float32 FVCOM NetCDF inputs.
    """
    nlevels, npoints = f.shape
    out = np.empty(npoints, dtype=np.float64)
    for point in prange(npoints):
        upper_depth, lower_depth = -np.inf, np.inf
        upper_value, lower_value = np.nan, np.nan
        for layer in range(nlevels):
            depth, value = depths[layer, point], f[layer, point]
            if not np.isfinite(depth) or not np.isfinite(value):
                continue
            if depth <= z and depth > upper_depth:
                upper_depth, upper_value = depth, value
            if depth >= z and depth < lower_depth:
                lower_depth, lower_value = depth, value
        if np.isfinite(upper_depth) and np.isfinite(lower_depth):
            if upper_depth == lower_depth:
                out[point] = upper_value
            else:
                out[point] = (
                    upper_value * (lower_depth - z)
                    + lower_value * (z - upper_depth)
                ) / (lower_depth - upper_depth)
        elif np.isfinite(lower_depth):
            out[point] = lower_value
        else:
            out[point] = np.nan
    return out


class InterpolateToZ:
    def make_interpolation_matrices_TS(self, interpolation_depths=[-5]):
        '''Make matrices (numpy arrays) that, when multiplied with fvcom T or S matrix, interpolates data to a given depth.'''
        import fvtools.util.tools as tools
        for depth in interpolation_depths:
            interp_matrix = tools.make_interpolation_matrices(self.siglayz, depth)
            setattr(self, 'interpolation_matrix_TS_' + str(abs(int(depth))) + '_m', interp_matrix)

    def make_interpolation_matrices_uv(self, interpolation_depths=[-5]):
        '''Make matrices (numpy arrays) that, when multiplied with fvcom u or v matrix, interpolates data to a given depth.'''
        import fvtools.util.tools as tools
        for depth in interpolation_depths:
            interp_matrix = tools.make_interpolation_matrices(self.siglayz_uv, depth)
            setattr(self, f'interpolation_matrix_uv_{abs(int(depth))}_m', interp_matrix)

    def interpolate_to_z(self, f, z, depths=None):
        """Interpolate (sigma, location) values to z metres below surface.

        Nearest-layer values are returned above the shallowest sigma point;
        locations below the deepest valid sigma point return NaN. Exact
        matches to a sigma depth are returned without interpolation.
        """
        f = self._as_array(f) if hasattr(self, '_as_array') else np.ma.filled(
            np.ma.asarray(f, dtype=float), np.nan)
        # Avoid full-size float64 copies of the 3-D FVCOM input. Numba
        # converts scalar values to float64 internally for interpolation.
        f = np.asarray(f)
        if f.dtype.kind not in 'fiub' or (f.dtype.kind == 'f' and f.dtype.itemsize not in (4, 8)):
            f = np.asarray(f, dtype=float)
        if depths is None:
            candidates = ('siglayz', 'siglevz', 'siglayz_uv', 'siglevz_uv')
            depths = next((-np.asarray(getattr(self, attr)).T
                           for attr in candidates
                           if f.shape == getattr(self, attr).T.shape), None)
            if depths is None:
                raise ValueError('Field shape does not match sigma-depth coordinates')
        depths = np.asarray(depths)
        if depths.dtype.kind not in 'fiub' or (depths.dtype.kind == 'f' and depths.dtype.itemsize not in (4, 8)):
            depths = np.asarray(depths, dtype=float)
        if f.ndim != 2 or f.shape != depths.shape:
            raise ValueError('f and depths must have shape (sigma, location)')
        if not np.isscalar(z) or not np.isfinite(z) or z < 0:
            raise ValueError('z must be a nonnegative, finite depth')
        return _interpolate_to_z_numba(f, float(z), depths)
