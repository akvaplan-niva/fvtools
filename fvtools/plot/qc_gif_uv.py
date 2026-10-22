"""Animate horizontal FVCOM velocities and streamlines.

Use qc_gif.py for vertical sections and section-normal velocity instead.
Both scripts use fvtools.plot.movie_common for time/file selection.
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings

from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.plot.fvcom_streamlines import streamlines
from fvtools.plot.movie_common import (
    get_movie_frames,
    print_frame_range,
    save_animation,
    make_horizontal_figure,
    get_grid_georeference,
    validate_plot_bounds, validate_vertical_selection, make_movie_axes,
)

def main(filelist=None, folder=None, sigma=None, z=None, xlim=None,
         ylim=None, start=None, stop=None, vlev=None, hres=None,
         mname=None, verbose=False, fps=12, dpi=100,
         reference='epsg:32633', output_dir=None):
    """Save a velocity/streamline movie; defaults to depth-averaged ua/va.

    sigma: integer model-layer index; z: depth in metres below the surface.
    Both cannot be specified at once. start/stop use yyyy-mm-dd-HH (UTC).
    output_dir sets the destination folder; the absolute saved file path is returned.
    """
    validate_vertical_selection(sigma, z)
    validate_plot_bounds(xlim, ylim)
    if hres is not None and (not np.isscalar(hres) or
                             not np.isfinite(hres) or hres <= 0):
        raise ValueError('hres must be a positive finite distance')
    if vlev is not None:
        values = np.asarray(vlev)
        if (values.ndim != 1 or values.size < 2
                or not np.issubdtype(values.dtype, np.number)
                or not np.all(np.isfinite(values))
                or not np.all(np.diff(values) > 0)):
            raise ValueError('vlev must be strictly increasing finite contour levels')
        vlev = values

    frames = get_movie_frames(
        folder=folder, filelist=filelist, start=start, stop=stop,
    )
    print_frame_range(frames)
    grid = FVCOM_grid(frames.path[0], verbose=verbose, reference=reference)
    if xlim is not None and ylim is not None:
        grid = grid.subgrid(xlim, ylim)

    stream = streamlines(grid, color='w', verbose=verbose)
    maker = UVmov(
        grid, frames, stream, xlim, ylim, hres,
        avg=(sigma is None and z is None), vlev=vlev,
        verbose=verbose, dpi=dpi,
    )
    maker.sig, maker.z = sigma, z
    return save_animation(
        maker.fig,
        maker.animate,
        frame_count=len(frames),
        output_stem=mname or f'{grid.casename}_velocities',
        output_dir=output_dir,
        fps=fps,
        progress_label='- streamline movie: ',
        maker=maker,
        grid=grid,
    )



class UVmov:
    """Load and render one horizontal velocity frame at a time."""

    def __init__(self, M, fl, stream, xlim, ylim, hres,
                 avg, vlev=None, verbose=True, dpi=300):
        self.M = M
        self.fl = fl
        self.stream = stream
        self.xlim, self.ylim = xlim, ylim
        self.avg = avg
        self.vlev = vlev
        self.sig = None
        self.z = None

        self.stream.initialize_positions(xlim, ylim, hres)
        self.gp = get_grid_georeference(
            self.M, xlim=self.xlim, ylim=self.ylim, at='cell',
            sources=(None, 'hot', 'mapnik', 'voyager'), verbose=verbose,
        )
        self.fig = make_horizontal_figure(self.M, dpi=dpi)
        self._plot_axes = None


    def load_data(self, i):
        """Read two velocity components together, retaining a shared mask."""
        names = ('ua', 'va') if self.avg else ('u', 'v')
        uv = self.M.load_netCDF(
            self.fl.path[i], names, self.fl.index[i],
            sig=None if self.avg else self.sig,
            z=None if self.avg else self.z,
        )
        u = np.ma.asarray(uv[names[0]], dtype=float).filled(np.nan)
        v = np.ma.asarray(uv[names[1]], dtype=float).filled(np.nan)
        # FVCOM u/v and depth-averaged ua/va are always cell-centred.
        expected_shape = (self.M.cell_number,)
        if u.shape != expected_shape or v.shape != expected_shape:
            raise ValueError(
                f'Expected cell-centred velocities {expected_shape}; '
                f'got u={u.shape}, v={v.shape}'
            )

        invalid = ~np.isfinite(u) | ~np.isfinite(v)
        self.ufv = np.where(invalid, 0.0, u)
        self.vfv = np.where(invalid, 0.0, v)
        self.sp_fv = np.ma.array(np.hypot(u, v), mask=invalid)

        if self.vlev is None:
            valid = self.sp_fv.compressed()
            vmax = max(float(valid.max()), 1e-6) if valid.size else 0.1
            self.vlev = np.linspace(0, vmax, 20)

    def animate(self, i):
        """Draw contours, streamlines, and a single frame's title."""
        self.bar.update(i)
        self.load_data(i)
        if (getattr(self, '_plot_axes', None) is None
                or self._plot_axes.figure is not self.fig):
            self._plot_axes = make_movie_axes(
                self.fig, cmap='jet', levels=self.vlev,
                label=r'm$s^{-1}$', extend='max', background=self.gp,
            )
        ax = self._plot_axes
        ax.clear()  # Preserve the dedicated colorbar and basemap credit.
        plt.sca(ax)  # Some FVCOM/streamline helpers use pyplot's active axes.

        if self.gp is not None:
            ax.imshow(self.gp.img, extent=self.gp.extent)

        # A single valid cell is not enough for tricontourf: we need at
        # least one triangle whose three cell-centre vertices are valid.
        contour = self.M.plot_contour(
            self.sp_fv, levels=self.vlev, cmap='jet', extend='max',
            show=False, ax=ax, on_empty='skip',
        )
        if contour is None:
            ax.text(0.5, 0.5, 'No valid velocity triangles at this depth/time',
                    transform=ax.transAxes, ha='center', va='center')
        else:
            try:
                self.stream.get_streamlines(self.ufv, self.vfv)
            except Exception as exc:
                warnings.warn(f'Streamlines failed for frame {i}: {exc}',
                              RuntimeWarning, stacklevel=2)

        self.format_figure(i)
        return contour

    def format_figure(self, i):
        # Stable extent even when the valid (wet) cells change per frame.
        plt.xlim(self.xlim if self.xlim is not None else
                 (self.M.x.min(), self.M.x.max()))
        plt.ylim(self.ylim if self.ylim is not None else
                 (self.M.y.min(), self.M.y.max()))
        plt.xticks([])
        plt.yticks([])
        if self.avg:
            prefix = 'avg'
        elif self.sig is not None:
            prefix = f'sigma = {self.sig}'
        else:
            prefix = f'z = {self.z}'
        plt.title(f'{prefix}: {self.fl.datetime[i]:%m/%d, %H:%M:%S}')
