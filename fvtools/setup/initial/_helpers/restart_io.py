"""Atomic replacement helpers for existing FVCOM restart NetCDF files.

Write a copy in the destination directory and replace the original only after
all writes complete. This intentionally does NOT implement the separate TPXO
application lock (which lives in nesting.tpxo).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil
import time
from netCDF4 import Dataset


def publish_staged_restart(staged, destination, *, attempts=6, delay=0.05):
    """Atomically publish a closed NetCDF file, retrying transient Windows locks.

    The staged file and original remain separate until os.replace succeeds.
    A persistent lock raises PermissionError without deleting the original.
    """
    for attempt in range(attempts):
        try:
            os.replace(staged, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay * (attempt + 1))


@contextmanager
def atomic_update_restart(destination):
    """Yield a writable NetCDF copy, replacing original only on success.

    Does not create missing restart files. The temporary file is on the same
    filesystem to preserve os.replace atomicity. No network I/O is involved.
    """
    destination = Path(destination)
    if not destination.is_file():
        raise FileNotFoundError(destination)
    with TemporaryDirectory(prefix='.restart_stage_', dir=destination.parent) as workdir:
        staged = Path(workdir) / destination.name
        shutil.copy2(destination, staged)
        with Dataset(staged, 'r+') as nc:
            yield nc
            nc.sync()
        publish_staged_restart(staged, destination)
