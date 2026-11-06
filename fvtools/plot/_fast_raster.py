"""Cached FVCOM triangulation-to-pixel mapper for fast horizontal animations.

Each target pixel's containing triangle and barycentric weights are computed
only once. Subsequent timesteps use vectorized gathers followed by one imshow
update, with masked/dry triangles made transparent.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import BoundaryNorm


class FastRasterRenderer:
    """Rasterize a nodal or cell-centre triangulation at fixed map resolution.

    ``triangulation`` must be the topology used by the corresponding contour
    renderer. Interpolation is piecewise linear, except pixel resolution is
    fixed, and colour thresholds are applied by BoundaryNorm.
    """

    def __init__(self, ax, triangulation, *, cmap, levels, bounds, resolution=1000,
                 zorder=5, geometry=None):
        if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)) or resolution < 16:
            raise ValueError('resolution must be an integer >= 16')
        xmin, xmax, ymin, ymax = map(float, bounds)
        dx, dy = xmax - xmin, ymax - ymin
        if not (np.isfinite([xmin, xmax, ymin, ymax]).all() and dx > 0 and dy > 0):
            raise ValueError('Expected finite, increasing map bounds')
        extent = (xmin, xmax, ymin, ymax)
        scale = max(dx, dy)
        self.width = max(2, int(round(resolution * dx / scale)))
        self.height = max(2, int(round(resolution * dy / scale)))
        self.n_vertices = len(triangulation.x)
        if geometry is not None:
            (width, height, n_vertices, self.positions, self.nodes,
             self.w0, self.w1) = geometry
            if (self.width, self.height, self.n_vertices) != (width, height, n_vertices):
                raise ValueError('Cached raster geometry does not match the mesh/extent')
        else:
            self._prepare_geometry(triangulation, xmin, ymin, dx, dy)

        self.artist = ax.imshow(
            np.ma.masked_all((self.height, self.width), dtype=np.float32),
            extent=extent, origin='lower', interpolation='nearest',
            cmap=cmap, norm=BoundaryNorm(np.asarray(levels, dtype=float), cmap.N),
            zorder=zorder,
        )

    def geometry(self):
        """Return immutable-style cached interpolation arrays for another movie field."""
        return (self.width, self.height, self.n_vertices, self.positions,
                self.nodes, self.w0, self.w1)

    def _prepare_geometry(self, triangulation, xmin, ymin, dx, dy):
        # Pixel *centres*, not pixel corners; this matches imshow's extent.
        xx = xmin + (np.arange(self.width) + 0.5) * dx / self.width
        yy = ymin + (np.arange(self.height) + 0.5) * dy / self.height
        px, py = np.meshgrid(xx, yy)
        px, py = px.ravel(), py.ravel()

        tidx = triangulation.get_trifinder()(px, py)
        self.positions = np.flatnonzero(tidx >= 0).astype(np.intp)
        vertices = np.asarray(triangulation.triangles, dtype=np.intp)[tidx[self.positions]]
        self.nodes = vertices.astype(np.int32, copy=True)  # 3 vertex IDs/pixel
        # Barycentric weights for the first two vertices. Last = 1-a-b.
        x0, x1, x2 = (triangulation.x[vertices[:,j]] for j in range(3))
        y0, y1, y2 = (triangulation.y[vertices[:,j]] for j in range(3))
        x, y = px[self.positions], py[self.positions]
        denominator = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        self.w0 = (((y1-y2)*(x-x2) + (x2-x1)*(y-y2))/denominator).astype(np.float32)
        self.w1 = (((y2-y0)*(x-x2) + (x0-x2)*(y-y2))/denominator).astype(np.float32)

    def update(self, field):
        values = np.ma.asarray(np.squeeze(field), dtype=float)
        if values.ndim != 1 or values.size != self.n_vertices:
            raise ValueError('Expected one value for each triangulation vertex')
        values = np.asarray(np.ma.masked_invalid(values).filled(np.nan))
        samples = values[self.nodes]
        valid = np.isfinite(samples).all(axis=1)
        result = np.full((self.height * self.width,), np.nan, dtype=np.float32)
        if np.any(valid):
            w0 = self.w0[valid]
            w1 = self.w1[valid]
            data = samples[valid]
            result[self.positions[valid]] = (w0 * data[:,0] + w1 * data[:,1]
                                             + (1-w0-w1) * data[:,2]).astype(np.float32)
        self.artist.set_data(np.ma.masked_invalid(result.reshape(self.height, self.width)))
        return self.artist if np.any(valid) else None
