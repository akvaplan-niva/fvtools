"""Shared FVCOM movie input and writer helpers.

Both qc_gif and qc_gif_uv use this module.  Time conversion is delegated to
fvtools.grid.tools so the FVCOM time convention stays in one place.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import fill

import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from fvtools.grid.tools import Filelist, date2num, num2date


@dataclass(frozen=True)
class MovieFrames:
    """Selected frames, in playback order (the Filelist interface)."""

    time: np.ndarray
    datetime: np.ndarray
    path: list[str]
    index: list[int]

    def __len__(self):
        return len(self.path)

    def iter_files(self):
        """Yield (filepath, timestep indices) for each distinct source file."""
        grouped = {}
        for path, idx in zip(self.path, self.index):
            grouped.setdefault(path, []).append(idx)
        yield from grouped.items()


def list_nc_files(folder):
    """Return sorted FVCOM output files; exclude restart files and directories."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f'FVCOM output directory not found: {root}')
    files = sorted(
        str(path) for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == '.nc'
        and 'restart' not in path.name.lower()
    )
    if not files:
        raise ValueError(f'No FVCOM .nc files found in {root}')
    return files


def _as_1d(value):
    return np.atleast_1d(np.asarray(value)).reshape(-1)


def parse_time_bound(value):
    """Convert a UTC yyyy-mm-dd-HH string or datetime into FVCOM time units."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        try:
            value = datetime.strptime(value, '%Y-%m-%d-%H')
        except ValueError as exc:
            raise ValueError(
                f'Invalid time {value!r}; expected yyyy-mm-dd-HH'
            ) from exc
    if not isinstance(value, datetime):
        raise TypeError('Time bound must be a datetime, yyyy-mm-dd-HH, or number')
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return float(_as_1d(date2num([value]))[0])


def parse_time_range(start=None, stop=None):
    begin, end = parse_time_bound(start), parse_time_bound(stop)
    if begin is not None and end is not None and begin > end:
        raise ValueError('start must not be later than stop')
    return begin, end


def _validated_frames(time, dates, files, indices):
    times = _as_1d(time)
    dates = _as_1d(dates)
    paths = [str(path) for path in files]
    indices = [int(index) for index in indices]
    if not (len(times) == len(dates) == len(paths) == len(indices)):
        raise ValueError('Inconsistent lengths of FVCOM frame metadata')
    if not paths:
        raise ValueError('No FVCOM timesteps in the requested time range')
    return MovieFrames(times, dates, paths, indices)


def frames_from_files(files, start=None, stop=None):
    """Read only timestamps from NetCDF files; preserve original frame indices.

    start/stop can be user-facing date strings or already converted FVCOM times.
    Both endpoints are inclusive. Files are expected in playback order.
    """
    begin, end = parse_time_range(start, stop)
    times, paths, indices = [], [], []
    for path in files:
        with Dataset(path) as dataset:
            t = _as_1d(date2num(num2date(
                Itime=dataset['Itime'][:], Itime2=dataset['Itime2'][:]
            )))
        if not t.size:
            continue
        selected = np.ones(t.size, dtype=bool)
        if begin is not None:
            selected &= t >= begin
        if end is not None:
            selected &= t <= end
        keep = np.flatnonzero(selected)
        if keep.size:
            times.extend(t[keep].tolist())
            paths.extend([str(path)] * keep.size)
            indices.extend(keep.tolist())

    times = np.asarray(times, dtype=float)
    if not times.size:
        raise ValueError('No FVCOM timesteps in the requested time range')
    return _validated_frames(times, num2date(time=times), paths, indices)


def get_movie_frames(*, folder=None, fname=None, filelist=None,
                     start=None, stop=None):
    """Find frames from a folder, single file, or fvtools Filelist text file.

    Uses the existing fvtools Filelist parser when filelist is specified,
    avoiding a competing implementation of its text format.
    """
    if filelist is not None:
        # Filelist handles the public date-string arguments itself.
        fl = Filelist(filelist, start, stop)
        frames = _validated_frames(fl.time, fl.datetime, fl.path, fl.index)
    elif folder is not None:
        frames = frames_from_files(list_nc_files(folder), start=start, stop=stop)
    elif fname is not None:
        frames = frames_from_files([str(fname)], start=start, stop=stop)
    else:
        raise ValueError('Provide filelist, folder, or fname')
    return frames



def validate_plot_bounds(xlim=None, ylim=None):
    """Validate a paired projected map extent (also used by both movie scripts)."""
    if (xlim is None) != (ylim is None):
        raise ValueError('xlim and ylim must be supplied together')
    for name, value in (('xlim', xlim), ('ylim', ylim)):
        if value is None:
            continue
        bounds = np.asarray(value)
        if (bounds.shape != (2,)
                or not np.issubdtype(bounds.dtype, np.number)
                or not np.all(np.isfinite(bounds))
                or bounds[0] >= bounds[1]):
            raise ValueError(f'{name} must contain two increasing finite limits')


def validate_vertical_selection(sigma=None, z=None, *, allow_max=False):
    """Validate a sigma layer or depth; permit max only for scalar movies."""
    if sigma is not None and z is not None:
        raise ValueError('Specify either sigma or z, not both')
    if sigma is not None:
        valid_layer = (isinstance(sigma, (int, np.integer))
                       and not isinstance(sigma, (bool, np.bool_)))
        if not valid_layer and not (allow_max and sigma == 'max'):
            choices = "integer layer index, 'max', or None" if allow_max else 'integer layer index or None'
            raise ValueError(f'sigma must be an {choices}')
    if z is not None and (
        not isinstance(z, (int, float, np.number))
        or isinstance(z, (bool, np.bool_))
        or not np.isfinite(z) or z < 0
    ):
        raise ValueError('z must be a finite nonnegative depth in metres')

def print_frame_range(frames):
    print(f"Start: {frames.datetime[0].strftime('%d/%B-%Y, %H:%M:%S')}")
    print(f"End:   {frames.datetime[-1].strftime('%d/%B-%Y, %H:%M:%S')}")


class StablePalettePillowWriter(animation.PillowWriter):
    """GIF writer using one palette for every frame.

    PillowWriter normally converts each RGB image to a GIF palette
    independently.  A fixed Matplotlib colorbar can consequently change
    colour from frame to frame (especially above a detailed basemap).
    Reusing one palette prevents these *encoding* artefacts.  This affects
    GIFs only; MP4/ffmpeg output is unchanged.

    Like Matplotlib's PillowWriter, frames are buffered in memory before
    being written.  GIF output is limited to 256 colours by the format.
    """

    @staticmethod
    def _palette_for_frames(frames):
        from PIL import Image
        from math import ceil

        # Sample frames across the entire movie, not just the first one.
        # Work with small thumbnails to avoid constructing a full-size
        # montage, which would be expensive for long model runs.
        samples = np.unique(np.linspace(
            0, len(frames) - 1, min(len(frames), 32), dtype=int
        ))
        width, height, columns = 128, 96, 8
        rows = ceil(len(samples) / columns)
        sheet = Image.new('RGB', (columns * width, (rows + 1) * height),
                          color='white')
        for position, index in enumerate(samples):
            sample = frames[int(index)].convert('RGB').resize(
                (width, height), Image.Resampling.BILINEAR
            )
            sheet.paste(sample, (
                (position % columns) * width,
                (position // columns) * height,
            ))

        # Reserve a meaningful share of the colour samples for the fixed
        # right-hand colourbar and its labels, independently of the colours
        # elsewhere in the image. This helps when a basemap is present.
        first = (frames[0] if frames[0].mode == 'RGB'
                 else frames[0].convert('RGB'))
        w, h = first.size
        bar_region = first.crop((int(.86*w), int(.14*h),
                                 int(.99*w), int(.90*h)))
        bar_sample = bar_region.resize(
            (width, height), Image.Resampling.BILINEAR
        )
        for column in range(columns):
            sheet.paste(bar_sample, (column * width, rows * height))

        # Median-cut runs only on the small RGB sample image, never on
        # the full-resolution movie. The resulting palette is immutable.
        return sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)

    def finish(self):
        from PIL import Image

        if not self._frames:
            raise RuntimeError('No animation frames were rendered to the GIF')

        palette = self._palette_for_frames(self._frames)
        # Replace each buffered RGB frame in place with its compact P-mode
        # version. Keeping all original RGB frames *and* indexed copies would
        # unnecessarily increase peak RAM for large FVCOM movie exports.
        for i, frame in enumerate(self._frames):
            if frame.mode == 'RGBA':
                background = Image.new('RGBA', frame.size, 'white')
                rgb = Image.alpha_composite(background, frame).convert('RGB')
            else:
                rgb = frame if frame.mode == 'RGB' else frame.convert('RGB')
            self._frames[i] = rgb.quantize(
                palette=palette, dither=Image.Dither.NONE
            )

        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0, optimize=False,
        )
        # Release buffers immediately, rather than retaining all frames
        # on an object that can outlive a completed animation.
        self._frames.clear()


def get_animator():
    """Return a Matplotlib writer class and its file extension."""
    available = set(animation.writers.list())
    for name, suffix in (
        ('ffmpeg', 'mp4'),
        ('pillow', 'gif'),
        ('imagemagick', 'gif'),
        ('html', 'html'),
    ):
        if name in available:
            return (StablePalettePillowWriter if name == 'pillow'
                    else animation.writers[name]), suffix
    raise RuntimeError('No supported Matplotlib animation writer available')


class MovieProgress:
    """Progress indicator shared by FVCOM movie writers.

    Use as a context manager; completion and cleanup also happen on error.
    `update(frame_index)` takes a zero-based animation frame index.

    Matplotlib may request a frame more than once (especially frame zero),
    so repeated or out-of-order updates never move the bar backwards.
    """

    def __init__(self, total, label='- movie: '):
        if not isinstance(total, (int, np.integer)) or total < 1:
            raise ValueError('Movie progress requires at least one frame')
        self.total = int(total)
        self.label = str(label)
        self.bar = None  # Underlying progressbar.ProgressBar instance.
        self._last_value = 0
        self._finished = False

    def start(self):
        """Start once. Imports progressbar only when actually needed."""
        if self._finished:
            raise RuntimeError('Cannot restart a completed MovieProgress')
        if self.bar is None:
            import progressbar as pb
            self.bar = pb.ProgressBar(
                widgets=[self.label, pb.Percentage(), pb.Bar(), pb.ETA()],
                maxval=self.total,
            )
            self.bar.start()
        return self

    def update(self, frame_index):
        """Advance for a zero-based frame index; ignore duplicate draws."""
        if self._finished:
            return
        if self.bar is None:
            self.start()
        value = min(max(int(frame_index) + 1, 0), self.total)
        if value > self._last_value:
            self.bar.update(value)
            self._last_value = value

    def finish(self):
        """Finish at most once, even when movie.save raises."""
        if not self._finished:
            self._finished = True
            if self.bar is not None:
                self.bar.finish()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()
        return False


# ---------------------------------------------------------------------------
# Shared figure setup and optional geographic background
# ---------------------------------------------------------------------------

def make_horizontal_figure(grid, *, size=15.0, dpi=300):
    """Create the figure used for horizontal scalar and velocity movies.

    Retains the original FVCOM movie aspect ratio and figure sizing.
    The caller owns the figure, typically handing it to save_animation().
    """
    dx = float(np.ptp(grid.x))
    dy = float(np.ptp(grid.y))
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        raise ValueError('Grid must have nonzero finite horizontal extent')
    if size <= 0 or dpi <= 0:
        raise ValueError('Figure size and DPI must be positive')
    return plt.figure(figsize=(size * 1.2 * dx / dy, size), dpi=dpi)


def make_section_figure(*, dpi=100, figsize=(19.2, 8.74)):
    """Create a wide figure for FVCOM vertical-section animations."""
    if len(figsize) != 2 or any(v <= 0 for v in figsize) or dpi <= 0:
        raise ValueError('Section figure size and DPI must be positive')
    return plt.figure(figsize=figsize, dpi=dpi)


# geoplot downloads tiles using Contextily.  xyzservices provides the
# attribution metadata that Contextily associates with those providers.
# These strings are fallbacks only, for environments without xyzservices.
_BASEMAP_PROVIDERS = {
    'hot': ('OpenStreetMap', 'HOT'),
    'mapnik': ('OpenStreetMap', 'Mapnik'),
    'voyager': ('CartoDB', 'Voyager'),
}
_BASEMAP_FALLBACK_CREDITS = {
    'hot': ('© OpenStreetMap contributors; tiles styled by the '
            'Humanitarian OpenStreetMap Team (HOT), hosted by OpenStreetMap France'),
    'mapnik': '© OpenStreetMap contributors',
    'voyager': '© OpenStreetMap contributors; © CARTO',
}


def _provider_credit(source):
    """Use the real Contextily/xyzservices attribution when available."""
    if source is None:
        source = 'hot'  # geoplot's default Contextily provider.

    if not isinstance(source, str):
        attribution = getattr(source, 'attribution', None)
        if not attribution and hasattr(source, 'get'):
            attribution = source.get('attribution')
        return str(attribution) if attribution else None

    key = source.lower()
    if key not in _BASEMAP_PROVIDERS:
        return None  # Never invent credits for unknown tile servers.

    try:
        import xyzservices.providers as providers
        group, name = _BASEMAP_PROVIDERS[key]
        provider = getattr(getattr(providers, group), name)
        credit = provider.get('attribution')
    except (ImportError, AttributeError, KeyError):
        credit = _BASEMAP_FALLBACK_CREDITS[key]

    credit = str(credit).replace('(C)', '©')
    if key == 'hot' and 'HOT' not in credit:
        credit += ' (HOT)'
    return credit


def basemap_credit(source, background=None):
    """Attribution for the tile source that actually supplied the image.

    Prefer metadata returned by geoplot/Contextily, if present. Otherwise
    use Contextily's provider information for the successful source, never
    the attribution of a failed provider in the fallback chain.
    """
    if background is not None:
        supplied = getattr(background, 'attribution', None)
        if supplied:
            credit = str(supplied)
        else:
            credit = None
            for attribute in ('provider', 'source'):
                provider = getattr(background, attribute, None)
                if provider is not None:
                    credit = _provider_credit(provider)
                    if credit:
                        break
    else:
        credit = None

    if not credit:
        credit = _provider_credit(source)
    if not credit:
        return None

    # Plain-text licensing URLs remain legible when a GIF/MP4 is detached
    # from its source page. Contextily may otherwise supply only a label.
    if ('OpenStreetMap' in credit and 'openstreetmap.org/copyright' not in credit):
        credit += ' · openstreetmap.org/copyright'
    if ('CARTO' in credit and 'carto.com/attribution' not in credit):
        credit += ' · carto.com/attribution'
    return credit


def add_basemap_credit(figure, background):
    """Draw a *persistent*, legible credit on a horizontal movie figure.

    Called once per figure, not each frame. The figure itself must not be
    cleared during animation; only the map axes can be cleared.
    """
    credit = getattr(background, 'attribution', None) if background is not None else None
    if not credit:
        return None
    return figure.text(
        0.99, 0.012, fill(str(credit), width=95, break_long_words=False),
        transform=figure.transFigure,
        ha='right', va='bottom', fontsize=8, color='#171717',
        bbox=dict(facecolor='white', alpha=0.88, edgecolor='none', pad=2),
        zorder=100,
    )


def make_movie_axes(figure, *, cmap, levels, label='', extend='both',
                    background=None):
    """Reserve a fixed plot area and a persistent colorbar for every frame.

    The colorbar is based on the fixed frame-independent contour levels. It
    remains visible even on a time step without valid triangles. Redraw only
    the map axes with ``ax.clear()``; never call ``figure.clear()`` per frame.
    """
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or len(levels) < 2 or not np.all(np.diff(levels) > 0):
        raise ValueError('Movie contour levels must be strictly increasing')
    ax = figure.add_axes([0.08, 0.075, 0.765, 0.865])
    cax = figure.add_axes([0.875, 0.20, 0.018, 0.64])
    norm = Normalize(vmin=float(levels[0]), vmax=float(levels[-1]))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    figure.colorbar(mappable, cax=cax, label=label, boundaries=levels,
                    extend=extend, spacing='proportional')
    add_basemap_credit(figure, background)
    return ax


def get_georeference(x, y, reference, *, sources=('hot', 'mapnik', 'voyager'),
                     verbose=False):
    """Optionally fetch a georeferenced basemap for a horizontal movie.

    `sources` is tried in order. `None` uses geoplot's default source.
    Failure is non-fatal: callers can draw their data without a basemap.
    Import geoplot on demand so section-only workflows need no map setup.
    """
    if verbose:
        print('- Downloading georeference')
    try:
        from fvtools.plot.geoplot import geoplot
    except ImportError as exc:
        if verbose:
            print(f'  -> Georeference support unavailable: {exc}')
        return None
    for source in sources:
        try:
            options = {'projection': reference}
            if source is not None:
                options['source'] = source
            background = geoplot(x, y, **options)
            if background is not None:
                # Real geoplot objects contain the raster and its extent.
                # Assign only to such objects; small mocked objects in tests
                # are not necessarily mutable.
                if hasattr(background, 'img') and hasattr(background, 'extent'):
                    credit = basemap_credit(source, background)
                    if credit is None:
                        raise ValueError(
                            f'Unknown map attribution for source {source!r}; '
                            'set the provider attribution explicitly'
                        )
                    try:
                        background.attribution = credit
                    except (AttributeError, TypeError):
                        # The only properties used by the movie scripts are
                        # img, extent and attribution. Support read-only map
                        # result objects, too.
                        from types import SimpleNamespace
                        background = SimpleNamespace(
                            img=background.img, extent=background.extent,
                            attribution=credit,
                        )
                return background
        except Exception as exc:
            if verbose:
                print(f'  -> Georeference source {source or "default"} unavailable: {exc}')
    if verbose:
        print('  -> No georeference available; plotting without a background map.')
    return None


def get_grid_georeference(grid, *, xlim=None, ylim=None, at='node',
                          sources=('hot', 'mapnik', 'voyager'), verbose=False):
    """Select the map extent using plot limits or grid node/cell positions."""
    if xlim is not None and ylim is not None:
        x, y = xlim, ylim
    elif at == 'node':
        x, y = grid.x, grid.y
    elif at == 'cell':
        x, y = grid.xc, grid.yc
    else:
        raise ValueError("at must be 'node' or 'cell'")
    return get_georeference(x, y, grid.reference, sources=sources,
                            verbose=verbose)


# ---------------------------------------------------------------------------
# Matplotlib animation lifecycle
# ---------------------------------------------------------------------------

def save_animation(
    figure,
    draw_frame,
    *,
    frame_count,
    output_stem,
    output_dir=None,
    fps=12,
    progress_label='- movie: ',
    maker=None,
    grid=None,
):
    """Render and save an FVCOM animation, owning all writer resources.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure that the callback draws on. This function ALWAYS closes it.
    draw_frame : callable
        Callback accepting a zero-based frame index; performs plotting only.
    frame_count : int
        Number of frames to encode (positive).
    output_stem : str or pathlib.Path
        Destination filename/path without the writer-specific extension.
    output_dir : str or pathlib.Path, optional
        Directory for the movie. Defaults to Path.cwd(), regardless of
        any directory in output_stem. Created if necessary.
    fps : float
        Frames per second, positive.
    progress_label : str
        Label for the shared progress bar.
    maker : object, optional
        Object whose callback uses a progress attribute (e.g. maker.bar).
    grid : FVCOM_grid, optional
        If provided, reuse its NetCDF session while saving the frames.

    Returns
    -------
    pathlib.Path
        Path of the saved animation.

    Notes
    -----
    The caller prepares a figure, its data, and the callback. This helper
    chooses the writer, creates FuncAnimation, handles progress and NetCDF
    session lifetimes, and closes the figure even when encoding fails.
    """
    try:
        if not isinstance(frame_count, (int, np.integer)) or frame_count < 1:
            raise ValueError('An animation must contain at least one frame')
        if not np.isscalar(fps) or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Animation fps must be a positive finite number')

        Writer, extension = get_animator()
        # Always use the Python/Jupyter working directory unless the caller
        # explicitly selects another output directory.
        destination = Path.cwd() if output_dir is None else Path(output_dir).expanduser()
        filename = Path(output_stem).name
        if not filename or filename in ('.', '..'):
            raise ValueError('output_stem must specify a movie filename')
        output_path = (destination / f'{filename}.{extension}').resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with MovieProgress(frame_count, progress_label) as progress:
            if maker is not None:
                maker.bar = progress

            movie = animation.FuncAnimation(
                figure,
                draw_frame,
                frames=int(frame_count),
                repeat=False,
                blit=False,
                cache_frame_data=False,
            )
            session = grid.netcdf_session() if grid is not None else nullcontext()
            with session:
                movie.save(str(output_path), writer=Writer(fps=fps))

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f'Animation writer produced no output: {output_path}')
        print(f'--> Movie saved: {output_path}')
        return output_path
    finally:
        plt.close(figure)
