"""Shared FVCOM movie input and writer helpers.

Both qc_gif and qc_gif_uv use this module.  Time conversion is delegated to
fvtools.grid.tools so the FVCOM time convention stays in one place.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.animation as animation
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

def print_frame_range(frames):
    print(f"Start: {frames.datetime[0].strftime('%d/%B-%Y, %H:%M:%S')}")
    print(f"End:   {frames.datetime[-1].strftime('%d/%B-%Y, %H:%M:%S')}")

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
            return animation.writers[name], suffix
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
    return get_georeference(x, y, grid.reference, sources=sources, verbose=verbose)


# ---------------------------------------------------------------------------
# Matplotlib animation lifecycle
# ---------------------------------------------------------------------------

def save_animation(
    figure,
    draw_frame,
    *,
    frame_count,
    output_stem,
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
        Destination path without the writer-specific extension.
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
        output_path = Path(f'{output_stem}.{extension}')

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

        return output_path
    finally:
        plt.close(figure)
