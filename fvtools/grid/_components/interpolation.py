"""FVCOM sigma-to-z interpolation mixin; unchanged implementation."""

import numpy as np

class InterpolateToZ:
    def make_interpolation_matrices_TS(self, interpolation_depths=[-5]):
        '''Make matrices (numpy arrays) that, when multiplied with fvcom T or S matrix, interpolates data to a given depth.'''
        import fvtools.grid.tools as tools
        for depth in interpolation_depths:
            interp_matrix = tools.make_interpolation_matrices(self.siglayz, depth)
            setattr(self, 'interpolation_matrix_TS_' + str(abs(int(depth))) + '_m', interp_matrix)

    def make_interpolation_matrices_uv(self, interpolation_depths=[-5]):
        '''Make matrices (numpy arrays) that, when multiplied with fvcom u or v matrix, interpolates data to a given depth.'''
        import fvtools.grid.tools as tools
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
        f = np.asarray(f, dtype=float)
        if depths is None:
            candidates = ('siglayz', 'siglevz', 'siglayz_uv', 'siglevz_uv')
            depths = next(( -np.asarray(getattr(self, attr)).T
                            for attr in candidates
                            if f.shape == getattr(self, attr).T.shape), None)
            if depths is None:
                raise ValueError('Field shape does not match sigma-depth coordinates')
        depths = np.asarray(depths, dtype=float)
        if f.ndim != 2 or f.shape != depths.shape:
            raise ValueError('f and depths must have shape (sigma, location)')
        if not np.isscalar(z) or not np.isfinite(z) or z < 0:
            raise ValueError('z must be a nonnegative, finite depth')

        valid = np.isfinite(f) & np.isfinite(depths)
        above = np.where(valid & (depths <= z), depths, -np.inf)
        below = np.where(valid & (depths >= z), depths, np.inf)
        upper_index = np.argmax(above, axis=0)
        lower_index = np.argmin(below, axis=0)
        loc = np.arange(f.shape[1])
        upper_depth, lower_depth = above[upper_index, loc], below[lower_index, loc]
        upper_value = f[upper_index, loc]
        lower_value = f[lower_index, loc]

        out = np.full(f.shape[1], np.nan, dtype=float)
        exact = np.isfinite(upper_depth) & np.isfinite(lower_depth) & (upper_depth == lower_depth)
        out[exact] = upper_value[exact]
        interpolate = (np.isfinite(upper_depth) & np.isfinite(lower_depth)
                       & (lower_depth > upper_depth))
        out[interpolate] = (
            upper_value[interpolate] * (lower_depth[interpolate] - z)
            + lower_value[interpolate] * (z - upper_depth[interpolate])
        ) / (lower_depth[interpolate] - upper_depth[interpolate])

        # Shallower than topmost sigma point: reuse the shallowest value.
        shallower = (~np.isfinite(upper_depth)) & np.isfinite(lower_depth)
        out[shallower] = lower_value[shallower]
        return out
