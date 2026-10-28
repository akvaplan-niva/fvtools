"""BuildCase bathymetry diagnostics, compatible with JupyterLab ipympl widgets."""
import numpy as np
import matplotlib.pyplot as plt


def _draw_field(ax, x, y, tri, values, *, levels=None, cmap='terrain', contour=False):
    """Plot shallow, deep, and constant fields without invalid contour levels."""
    lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi, atol=1e-12, rtol=0.0):
        # tricontourf requires non-identical levels; flat bathymetry is valid.
        artist = ax.tripcolor(x, y, tri, values, cmap=cmap, shading='flat')
        delta = max(1., abs(lo) * 0.01)
        artist.set_clim(lo - delta, hi + delta)
    else:
        if levels is None:
            levels = np.linspace(lo, hi, 15)
        artist = ax.tricontourf(x, y, tri, values, levels=levels, cmap=cmap)
        if contour:
            ax.tricontour(x, y, tri, values, levels=levels[1:-1],
                          colors='k', linewidths=0.35)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


class BathymetryPlotter:
    def show_depth(self):
        """Display raw depth, smoothed depth, and their difference.

        Missing source-bathymetry nodes (saved before min_depth replacement)
        are marked red on the first panel.  The usual three interactive
        Matplotlib panels and public method name are preserved.
        """
        raw = np.asarray(self.h_raw, dtype=float)
        smooth = np.asarray(self.h, dtype=float)
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        if raw.ndim != 1 or smooth.shape != raw.shape or len(x) != len(raw) or len(y) != len(raw):
            raise ValueError('Bathymetry and grid coordinates must have matching node dimensions')
        if not np.isfinite(raw).all() or not np.isfinite(smooth).all():
            raise ValueError('Bathymetry QC requires finite nodal depths')
        if not len(raw):
            raise ValueError('Cannot plot an empty bathymetry grid')

        diff = raw - smooth
        missing = np.asarray(getattr(self, 'bathymetry_missing_mask',
                                     np.zeros(len(raw), dtype=bool)), dtype=bool)
        if missing.shape != raw.shape:
            raise ValueError('Missing-bathymetry mask does not match the grid nodes')

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
        # Raw and smoothed panels use the same color scale for comparison.
        lo = min(float(raw.min()), float(smooth.min()))
        hi = max(float(raw.max()), float(smooth.max()))
        depth_levels = np.linspace(lo, hi, 15) if hi > lo else None
        for axis, values, title in (
            (ax[0], raw, 'raw bathymetry'),
            (ax[1], smooth, 'smoothed bathymetry'),
        ):
            image = _draw_field(axis, x, y, self.tri, values,
                                levels=depth_levels, contour=True)
            axis.set_title(title)
            fig.colorbar(image, ax=axis, label='Depth (m)')

        if missing.any():
            ax[0].scatter(x[missing], y[missing], marker='x', s=14, c='red',
                          linewidths=0.8, zorder=10, rasterized=True,
                          label=f'Missing source depth ({int(missing.sum()):,})')
            ax[0].legend(loc='best', fontsize='small')

        max_delta = float(np.max(np.abs(diff)))
        change_levels = (np.linspace(-max_delta, max_delta, 21)
                         if max_delta > 1e-12 else None)
        changed = _draw_field(ax[2], x, y, self.tri, diff,
                              levels=change_levels, cmap='RdBu_r')
        ax[2].set_title('raw - smoothed bathymetry')
        fig.colorbar(changed, ax=ax[2], label='Depth change (m)')
        fig.suptitle(f'Missing source depths: {int(missing.sum()):,}/{len(raw):,} nodes'
                     f'  |  max |depth change|: {max_delta:.2f} m')
        return None
