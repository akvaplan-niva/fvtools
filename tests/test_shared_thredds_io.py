"""Offline regressions for shared ROMS/MetCoOp NetCDF access.

No external host is contacted. These tests guard connection ownership,
reopen-on-failure, exception policies, and public reader delegation.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from fvtools.util.thredds_io import (
    SourcePool, is_retryable_netcdf_error, retry_read, source_dataset,
)


class FakeDataset:
    def __init__(self, path, events):
        self.path, self.events = path, events
        self.closed = False
        events.append(('open', path))

    def close(self):
        assert not self.closed, 'Dataset double-closed'
        self.closed = True
        self.events.append(('close', self.path))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.mark.parametrize('max_open,paths,expected', [
    (1, ['A', 'A', 'B', 'B', 'A'], ['A', 'B', 'A']),
    (3, ['A', 'B', 'A', 'C', 'B'], ['A', 'B', 'C']),
])
def test_pool_reuses_and_closes_sources(max_open, paths, expected):
    events = []
    with SourcePool(lambda path: FakeDataset(path, events), max_open=max_open, mode=None) as pool:
        handles = [pool.get(p) for p in paths]
        if paths[0] == paths[1]:
            assert handles[0] is handles[1]
    assert [p for action, p in events if action == 'open'] == expected
    assert [p for action, p in events if action == 'close'] == expected


def test_pooled_source_evicted_after_transient_read_failure():
    events = []
    pool = SourcePool(lambda p: FakeDataset(p, events), max_open=1, mode=None)
    with pool:
        with pytest.raises(RuntimeError, match='DAP failure'):
            with source_dataset('roms.nc', lambda p: FakeDataset(p, events), pool=pool):
                raise RuntimeError('NetCDF: DAP failure')
        with source_dataset('roms.nc', lambda p: FakeDataset(p, events), pool=pool) as ds:
            assert not ds.closed
    assert events == [('open', 'roms.nc'), ('close', 'roms.nc'),
                      ('open', 'roms.nc'), ('close', 'roms.nc')]


def test_nonretryable_exception_does_not_evict_live_source():
    events = []
    with SourcePool(lambda p: FakeDataset(p, events), max_open=1, mode=None) as pool:
        with pytest.raises(ValueError):
            with source_dataset('roms.nc', None, pool=pool):
                raise ValueError('bad variable shape')
        assert pool.get('roms.nc') is pool.get('roms.nc')
        assert len(events) == 1
    assert events == [('open', 'roms.nc'), ('close', 'roms.nc')]


def test_unscoped_source_closes_immediately():
    events = []
    with source_dataset('roms.nc', lambda p: FakeDataset(p, events)):
        assert events == [('open', 'roms.nc')]
    assert events == [('open', 'roms.nc'), ('close', 'roms.nc')]


@pytest.mark.parametrize('exc,expected', [
    (OSError('connection reset'), True),
    (RuntimeError('NetCDF: DAP failure'), True),
    (RuntimeError('NetCDF: DAP server error'), True),
    (RuntimeError('NetCDF: curl error'), True),
    (RuntimeError('NetCDF: I/O failure'), True),
    (ValueError('bad depth'), False),
    (RuntimeError('coding error'), False),
])
def test_classifier_matches_roms_dap_errors(exc, expected):
    assert is_retryable_netcdf_error(exc) is expected


def test_retry_read_closes_before_reopening_and_preserves_values():
    events = []
    with SourcePool(lambda p: FakeDataset(p, events), max_open=1, mode=None) as pool:
        attempts = []
        def operation():
            with source_dataset('A', None, pool=pool):
                attempts.append(1)
                if len(attempts) == 1:
                    raise RuntimeError('NetCDF: DAP failure')
                return 42
        assert retry_read(operation, max_retries=1, retry_delay=0,
                          sleep=lambda s: None, on_transient=lambda: pool.close(suppress_errors=True)) == 42
    assert events == [('open','A'), ('close','A'), ('open','A'), ('close','A')]


def test_no_retry_for_invalid_data():
    calls = []
    def operation():
        calls.append(1)
        raise ValueError('invalid interpolation')
    with pytest.raises(ValueError, match='invalid interpolation'):
        retry_read(operation, max_retries=4, retry_delay=0, sleep=lambda s: None)
    assert calls == [1]


def test_retry_exhausted_preserves_original_failure():
    def operation():
        raise RuntimeError('NetCDF: DAP failure')
    with pytest.raises(RuntimeError, match='after 3 attempts') as exc:
        retry_read(operation, max_retries=2, retry_delay=0, sleep=lambda s: None)
    assert str(exc.value.__cause__) == 'NetCDF: DAP failure'


def test_roms_nesting_and_restart_inherit_same_reader():
    from fvtools.grid.roms_grid import RomsDownloader
    from fvtools.nesting.roms_nesting_fg import Roms2FVCOMNest
    from fvtools.prepro.interpol_roms_restart import Roms2FVCOMRestart
    assert issubclass(Roms2FVCOMNest, RomsDownloader)
    assert issubclass(Roms2FVCOMRestart, RomsDownloader)
    assert Roms2FVCOMNest._roms_dataset is RomsDownloader._roms_dataset
    assert Roms2FVCOMRestart._roms_dataset is RomsDownloader._roms_dataset


def test_metcoop_pool_is_identical_shared_pool():
    from fvtools.atm._metcoop_io import SourcePool as MetSourcePool
    assert MetSourcePool is SourcePool
