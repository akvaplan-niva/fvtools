"""Opt-in timing of ROMS nesting phase 6A/6B, without modifying fvtools.

This module temporarily wraps a single-process ROMS nesting run.  It records
Python NetCDF variable indexing, not the number of HTTP requests made by the
NetCDF-C/OPeNDAP libraries.  It neither reads additional variables nor opens
an additional remote connection.

Usage
-----
    import fvtools.nesting.roms_nesting_fg as rn
    from roms_nesting_profiler import profile_roms_nesting, compare_profiles

    with profile_roms_nesting(rn, 'profile_one.json'):
        rn.main(..., nprocs=1, time_block_size=1)
    with profile_roms_nesting(rn, 'profile_two.json'):
        rn.main(..., nprocs=1, time_block_size=2)
    compare_profiles('profile_one.json', 'profile_two.json')
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any


def _seconds(seconds: float) -> str:
    return f'{seconds:.3f} s'


@dataclass
class Profile:
    run_seconds: float = 0.0
    dump_seconds: float = 0.0
    read_seconds: float = 0.0
    interpolate_seconds: float = 0.0
    write_seconds: float = 0.0
    metadata_seconds: float = 0.0
    dataset_open_seconds: float = 0.0
    variable_slice_seconds: float = 0.0
    dataset_opens: int = 0
    netcdf_variable_slices: int = 0
    returned_bytes: int = 0
    timesteps_written: int = 0
    single_reads: int = 0
    block_reads: int = 0
    attempted_slices: int = 0
    attempted_block_sizes: list[int] = field(default_factory=list)
    slices_by_variable: Counter = field(default_factory=Counter)
    bytes_by_variable: Counter = field(default_factory=Counter)
    _active: bool = field(default=False, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            'run_seconds': self.run_seconds,
            'dump_seconds': self.dump_seconds,
            'read_seconds': self.read_seconds,
            'interpolate_seconds': self.interpolate_seconds,
            'write_seconds': self.write_seconds,
            'metadata_seconds': self.metadata_seconds,
            'dataset_open_seconds': self.dataset_open_seconds,
            'variable_slice_seconds': self.variable_slice_seconds,
            'dataset_opens': self.dataset_opens,
            'netcdf_variable_slices': self.netcdf_variable_slices,
            'returned_bytes': self.returned_bytes,
            'timesteps_written': self.timesteps_written,
            'single_reads': self.single_reads,
            'block_reads': self.block_reads,
            'attempted_slices': self.attempted_slices,
            'attempted_block_sizes': list(self.attempted_block_sizes),
            'slices_by_variable': dict(self.slices_by_variable),
            'bytes_by_variable': dict(self.bytes_by_variable),
            'definitions': {
                'netcdf_variable_slices': 'Successful Python netCDF variable __getitem__ calls; NOT HTTP requests',
                'returned_bytes': 'Uncompressed nbytes of returned array data; NOT network bytes',
                'dataset_opens': 'Dataset(filepath) calls within the serial dump loop',
                'run_seconds': 'Wall time spent inside the profile context (including setup)',
                'dump_seconds': 'Time spent in dump_single (including progressbar and control flow)',
                'read_seconds': 'Time in read_timestep/read_timestep_block, including retry and opening file',
                'interpolate_seconds': 'Time in _interpolate_downloaded_timestep',
                'write_seconds': 'Time in dump_timestep_to_nest',
            },
        }

    def display(self) -> None:
        residual = self.dump_seconds - self.read_seconds - self.interpolate_seconds - self.write_seconds
        print('\n=== ROMS nesting I/O profile (single process) ===')
        for title, val in [
            ('Entire rn.main() / context', self.run_seconds),
            ('Only dump_single()', self.dump_seconds),
            ('ROMS read (incl. opening)', self.read_seconds),
            ('  Of which: opening NetCDF', self.dataset_open_seconds),
            ('  Of which: var[...] slices', self.variable_slice_seconds),
            ('Interpolation', self.interpolate_seconds),
            ('Write to nesting file', self.write_seconds),
            ('Other dump overhead (incl. metadata)', residual),
            ('  Metadata estimate', self.metadata_seconds),
        ]:
            print(f'{title:38s} {_seconds(val):>13s}')
        print(f'Written timesteps: {self.timesteps_written}')
        print(f'Opened ROMS datasets: {self.dataset_opens}')
        print(f'Single-step reads: {self.single_reads}; block reads: {self.block_reads}')
        print(f'Block sizes attempted: {Counter(self.attempted_block_sizes)}')
        print(f'NetCDF variable slices: {self.netcdf_variable_slices}')
        print(f'Array data returned: {self.returned_bytes / (1024 ** 2):.1f} MiB (NOT transferred HTTP bytes)')
        print(f'Variable slice counts: {dict(self.slices_by_variable)}')
        print('Note: a Python NetCDF slice can trigger several HTTP requests internally.')


class _VariableProxy:
    def __init__(self, real: Any, name: str, profile: Profile):
        self._real = real
        self._name = name
        self._profile = profile

    def __getitem__(self, indices: Any) -> Any:
        profile = self._profile
        profile.attempted_slices += 1
        t0 = perf_counter()
        try:
            result = self._real[indices]
        finally:
            profile.variable_slice_seconds += perf_counter() - t0
        profile.netcdf_variable_slices += 1
        profile.slices_by_variable[self._name] += 1
        # An ndarray or MaskedArray's nbytes counts its uncompressed data.
        # Do not convert, copy, or otherwise alter the actual returned array.
        nbytes = getattr(result, 'nbytes', None)
        if nbytes is not None:
            count = int(nbytes)
            profile.returned_bytes += count
            profile.bytes_by_variable[self._name] += count
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _DatasetProxy:
    def __init__(self, real: Any, profile: Profile):
        self._real = real
        self._profile = profile

    def __getitem__(self, name: str) -> Any:
        return _VariableProxy(self._real[name], name, self._profile)

    def __enter__(self):
        # Respect a real Dataset's context-manager hooks, but preserve proxy.
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._real.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        return self._real.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@contextmanager
def profile_roms_nesting(rn_module, report_path='roms_nesting_profile.json', *, _grid_module=None):
    """Instrument a single-process rn.main call without changing fvtools files.

    All temporary monkeypatches are restored even when a run raises an error.
    The run must use nprocs=1; multiprocess profiling is unsupported.
    """
    grid_module = _grid_module or importlib.import_module('fvtools.grid.roms_grid')
    original_class = rn_module.Roms2FVCOMNest
    original_dataset_factory = grid_module.Dataset
    metrics = Profile()

    def opened_dataset(*args, **kwargs):
        if not metrics._active:
            return original_dataset_factory(*args, **kwargs)
        start = perf_counter()
        try:
            result = original_dataset_factory(*args, **kwargs)
        finally:
            metrics.dataset_open_seconds += perf_counter() - start
        metrics.dataset_opens += 1
        return _DatasetProxy(result, metrics)

    class ProfiledNesting(original_class):
        def dump_single(self, *args, **kwargs):
            if metrics._active:
                raise RuntimeError('Nested profiled serial ROMS loops are unsupported')
            t0 = perf_counter()
            metrics._active = True
            try:
                return super().dump_single(*args, **kwargs)
            finally:
                metrics._active = False
                metrics.dump_seconds += perf_counter() - t0

        def dump(self, *args, **kwargs):
            raise ValueError('Profiling is supported only with nprocs=1 (dump_single)')

        def read_timestep(self, *args, **kwargs):
            if not metrics._active:
                return super().read_timestep(*args, **kwargs)
            metrics.single_reads += 1
            start = perf_counter()
            try:
                return super().read_timestep(*args, **kwargs)
            finally:
                metrics.read_seconds += perf_counter() - start

        def read_timestep_block(self, first_index, count, filepath, **kwargs):
            if not metrics._active:
                return super().read_timestep_block(first_index, count, filepath, **kwargs)
            metrics.block_reads += 1
            metrics.attempted_block_sizes.append(int(count))
            start = perf_counter()
            try:
                return super().read_timestep_block(first_index, count, filepath, **kwargs)
            finally:
                metrics.read_seconds += perf_counter() - start

        def estimate_roms_timestep_bytes(self, *args, **kwargs):
            start = perf_counter()
            try:
                return super().estimate_roms_timestep_bytes(*args, **kwargs)
            finally:
                if metrics._active:
                    metrics.metadata_seconds += perf_counter() - start

        def _interpolate_downloaded_timestep(self, *args, **kwargs):
            start = perf_counter()
            try:
                return super()._interpolate_downloaded_timestep(*args, **kwargs)
            finally:
                if metrics._active:
                    metrics.interpolate_seconds += perf_counter() - start

        def dump_timestep_to_nest(self, *args, **kwargs):
            start = perf_counter()
            try:
                value = super().dump_timestep_to_nest(*args, **kwargs)
                if metrics._active:
                    metrics.timesteps_written += 1
                return value
            finally:
                if metrics._active:
                    metrics.write_seconds += perf_counter() - start

    if getattr(original_class, '_is_roms_profiled', False):
        raise RuntimeError('A profiler context is already active on this module')
    ProfiledNesting._is_roms_profiled = True
    start = perf_counter()
    rn_module.Roms2FVCOMNest = ProfiledNesting
    grid_module.Dataset = opened_dataset
    try:
        yield metrics
    finally:
        metrics.run_seconds = perf_counter() - start
        grid_module.Dataset = original_dataset_factory
        rn_module.Roms2FVCOMNest = original_class
        metrics.display()
        if report_path is not None:
            p = Path(report_path)
            p.write_text(json.dumps(metrics.as_dict(), indent=2, ensure_ascii=False), encoding='utf-8')
            print(f'Profiling saved to: {p.resolve()}')


def compare_profiles(single_path, block_path) -> None:
    """Compare two profiles from runs on identical inputs, preferably back-to-back."""
    a = json.loads(Path(single_path).read_text(encoding='utf-8'))
    b = json.loads(Path(block_path).read_text(encoding='utf-8'))
    print('\n=== Comparison: time_block_size=1 vs time_block_size=2 ===')
    labels = [
        ('Whole run', 'run_seconds', 's'),
        ('Nesting write loop', 'dump_seconds', 's'),
        ('ROMS read', 'read_seconds', 's'),
        ('Dataset open time', 'dataset_open_seconds', 's'),
        ('Variable slice time', 'variable_slice_seconds', 's'),
        ('Interpolation', 'interpolate_seconds', 's'),
        ('Write nest', 'write_seconds', 's'),
        ('ROMs file opens', 'dataset_opens', 'count'),
        ('NetCDF variable slices', 'netcdf_variable_slices', 'count'),
        ('Uncompressed array data', 'returned_bytes', 'MiB'),
    ]
    print(f'{"Metric":30s} {"Size=1":>14s} {"Size=2":>14s} {"Change":>12s}')
    for label, key, unit in labels:
        factor = 1024 ** 2 if unit == 'MiB' else 1
        x = a[key] / factor
        y = b[key] / factor
        delta = 100.0 * (y / x - 1) if x else float('nan')
        print(f'{label:30s} {x:14.2f} {y:14.2f} {delta:+11.1f}%')
    if a['timesteps_written'] != b['timesteps_written']:
        print('WARNING: Different number of timesteps: comparisons are not like-for-like.')
    print('NOTE: variable slices are Python NetCDF operations, NOT actual HTTP request counts.')
