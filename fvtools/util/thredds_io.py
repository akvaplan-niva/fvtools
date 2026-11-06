"""Shared NetCDF/OPeNDAP connection lifecycle and bounded retry policy.

The reader's caller controls the scope. No global Dataset cache is kept: a
Dataset must never be pickled or sent to a multiprocessing worker.

AROME and ROMS retain their separate variable selection, spatial masks,
vertical coordinates, and interpolation methods.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import isfinite
from time import sleep as _sleep


def is_retryable_netcdf_error(exc):
    """Classify transient netCDF4/OPeNDAP errors, not numerical/code errors."""
    if isinstance(exc, (OSError, TimeoutError, ConnectionError)):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).casefold()
    return any(fragment in message for fragment in (
        'netcdf: dap failure',
        'netcdf: dap server error',
        'netcdf: curl error',
        'netcdf: i/o failure',
    ))


class SourcePool:
    """Own only a small number of open source Dataset handles.

    ``opener(path, mode)`` is the usual netCDF4.Dataset signature. For a
    factory that takes *only* a path, pass ``mode=None``. The ``max_open=1``
    mode closes the old ROMS file as soon as the path changes; MetCoOp needs
    up to three simultaneous handles for normal/accumulation/radiation files.
    """

    def __init__(self, opener, *, max_open=3, mode='r'):
        if not isinstance(max_open, int) or isinstance(max_open, bool) or max_open < 1:
            raise ValueError('max_open must be a positive integer')
        self.opener = opener
        self.max_open = max_open
        self.mode = mode
        self.handles = {}

    def get(self, path):
        if path not in self.handles:
            if len(self.handles) >= self.max_open:
                self.close()
            handle = self.opener(path) if self.mode is None else self.opener(path, self.mode)
            self.handles[path] = handle
        return self.handles[path]

    def close(self, *, suppress_errors=False):
        handles, self.handles = self.handles, {}
        error = None
        for handle in handles.values():
            try:
                handle.close()
            except Exception as exc:
                if not suppress_errors and error is None:
                    error = exc
        if error is not None:
            raise error

    def invalidate(self):
        """Discard potentially damaged handles without masking the read failure."""
        self.close(suppress_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@contextmanager
def source_dataset(path, opener, *, pool=None, classifier=is_retryable_netcdf_error):
    """Yield one source Dataset; evict bad cached handles on transient errors.

    Without an explicit pool, preserve ordinary one-open/one-close semantics.
    In the pooled case the owner closes all handles at the scope boundary.
    """
    if pool is None:
        with opener(path) as dataset:
            yield dataset
        return

    try:
        yield pool.get(path)
    except Exception as exc:
        if classifier(exc):
            # Network errors can also occur during variable slicing, after
            # opening the file. Never retry on the same damaged connection.
            pool.invalidate()
        raise


def retry_read(operation, *, max_retries=5, retry_delay=30,
               classifier=is_retryable_netcdf_error, sleep=None,
               on_transient=None, failure_message='Could not read NetCDF source',
               log_prefix='  - NetCDF I/O error'):
    """Retry transient read failures, preserving the final cause and traceback.

    ``operation`` must perform *source reading only*. Keep interpolation and
    output-file writes outside this retry boundary, to avoid repeating or
    hiding programming errors and partially committed writes.
    """
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError('max_retries must be a nonnegative integer')
    if not isinstance(retry_delay, (int, float)) or not isfinite(retry_delay) or retry_delay < 0:
        raise ValueError('retry_delay must be nonnegative and finite')
    if sleep is None:
        sleep = _sleep

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            if not classifier(exc):
                raise
            if on_transient is not None:
                on_transient()
            if attempt == max_retries:
                raise RuntimeError(f'{failure_message} after {attempt + 1} attempts') from exc
            print(f'{log_prefix}: {exc!s}. '
                  f'Retrying ({attempt + 1}/{max_retries}) in {retry_delay}s', flush=True)
            sleep(retry_delay)
