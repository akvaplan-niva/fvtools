"""Offline regression tests; no THREDDS requests are made."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'fvtools/grid/roms_grid.py'


class FakeVariable:
    def __init__(self, backend, name):
        self.backend = backend
        self.name = name
        self._array = backend.arrays[name]
        self.dtype = self._array.dtype

    @property
    def shape(self):
        if self.backend.metadata_fail_once == self.name:
            self.backend.metadata_fail_once = None
            raise RuntimeError('NetCDF: DAP failure')
        return self._array.shape

    def __getitem__(self, index):
        self.backend.reads.append(self.name)
        if self.backend.read_fail_once == self.name:
            self.backend.read_fail_once = None
            raise RuntimeError('NetCDF: DAP failure')
        if self.backend.read_fail_always == self.name:
            raise RuntimeError('NetCDF: DAP failure')
        if self.backend.fail_code == self.name:
            raise RuntimeError('Not a NetCDF network failure: a programming error')
        return self._array[index].copy()


class FakeDataset:
    def __init__(self, backend):
        self.backend = backend
        self.variables = {k: FakeVariable(backend, k) for k in backend.arrays}
        self.backend.opens += 1
        self.closed = False

    def __getitem__(self, name):
        return self.variables[name]

    def close(self):
        if not self.closed:
            self.closed = True
            self.backend.closes += 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FakeBackend:
    def __init__(self, with_bars=True):
        self.arrays = {
            name: np.full((4, 3, 2, 2), value, dtype='f4')
            for name, value in (('salt', 30.), ('temp', 8.), ('u', 2.), ('v', 4.))
        }
        self.arrays['zeta'] = np.ones((4, 2, 2), dtype='f4')
        if with_bars:
            self.arrays['ubar'] = np.full((4, 2, 2), 2.0, dtype='f4')
            self.arrays['vbar'] = np.full((4, 2, 2), 4.0, dtype='f4')
        self.opens = self.closes = 0
        self.reads = []
        self.read_fail_once = None
        self.read_fail_always = None
        self.metadata_fail_once = None
        self.fail_code = None

    def __call__(self, path):
        return FakeDataset(self)


def reader(backend):
    module_ast = ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls = next(n for n in module_ast.body if isinstance(n, ast.ClassDef)
               and n.name == 'RomsDownloader')
    sleeps = []
    env = dict(np=np, contextmanager=contextmanager, Dataset=backend,
               time_mod=SimpleNamespace(sleep=lambda seconds: sleeps.append(seconds)),
               InputError=ValueError,
               ROMSTimeStep=lambda idx: SimpleNamespace(index=idx))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(SOURCE), 'exec'), env)
    obj = env['RomsDownloader']()
    attrs = {f'{prefix}_{axis}{coord}': (0 if prefix == 'm' else 1)
             for prefix in ('m', 'x') for axis in ('r', 'u', 'v')
             for coord in ('i', 'j')}
    attrs['cropped_dsigma_u'] = np.array([0.25, 0.5, 0.25], dtype='f4')[:, None, None]
    attrs['cropped_dsigma_v'] = np.array([0.25, 0.5, 0.25], dtype='f4')[:, None, None]
    obj.N4 = SimpleNamespace(**attrs)
    return obj, sleeps


def test_dap_error_triggers_reopen_and_retry_serial():
    backend = FakeBackend()
    backend.read_fail_once = 'v'
    obj, sleeps = reader(backend)
    with obj.reuse_roms_datasets():
        timestep = obj.read_timestep(1, 'roms', max_retries=1, retry_delay=0)
        np.testing.assert_array_equal(timestep.v, backend.arrays['v'][1])
    assert (backend.opens, backend.closes, sleeps) == (2, 2, [0])


def test_dap_error_triggers_reopen_and_retry_block():
    backend = FakeBackend()
    backend.read_fail_once = 'v'
    obj, sleeps = reader(backend)
    if not hasattr(obj, 'read_timestep_block'):
        pytest.skip('Phase 6A does not include block reads')
    with obj.reuse_roms_datasets():
        steps = obj.read_timestep_block(1, 2, 'roms', max_retries=1, retry_delay=0)
    assert len(steps) == 2
    assert (backend.opens, backend.closes, sleeps) == (2, 2, [0])
    np.testing.assert_array_equal(steps[0].v, backend.arrays['v'][1])


@pytest.mark.parametrize('block', [False, True])
def test_dap_error_in_barotropic_variable_must_not_be_hidden_by_fallback(block):
    backend = FakeBackend()
    backend.read_fail_once = 'vbar'
    obj, sleeps = reader(backend)
    if block and not hasattr(obj, 'read_timestep_block'):
        pytest.skip('Phase 6A does not include block reads')
    with obj.reuse_roms_datasets():
        if block:
            result = obj.read_timestep_block(0, 2, 'roms', max_retries=1, retry_delay=0)[0]
        else:
            result = obj.read_timestep(0, 'roms', max_retries=1, retry_delay=0)
    np.testing.assert_array_equal(result.va, backend.arrays['vbar'][0])
    assert backend.opens == backend.closes == 2
    assert sleeps == [0]


@pytest.mark.parametrize('block', [False, True])
def test_missing_barotropic_velocity_is_derived_with_positive_sigma_weights(block):
    backend = FakeBackend(with_bars=False)
    obj, sleeps = reader(backend)
    if block and not hasattr(obj, 'read_timestep_block'):
        pytest.skip('Phase 6A does not include block reads')
    if block:
        result = obj.read_timestep_block(0, 2, 'roms')[0]
    else:
        result = obj.read_timestep(0, 'roms')
    np.testing.assert_allclose(result.ua, 2.0)
    np.testing.assert_allclose(result.va, 4.0)
    assert backend.opens == backend.closes == 1
    assert sleeps == []


def test_unrelated_runtime_error_is_not_retried():
    backend = FakeBackend()
    backend.fail_code = 'v'
    obj, sleeps = reader(backend)
    with obj.reuse_roms_datasets():
        with pytest.raises(RuntimeError, match='programming error'):
            obj.read_timestep(0, 'roms', max_retries=3, retry_delay=0)
    assert backend.opens == backend.closes == 1
    assert sleeps == []


def test_retry_exhaustion_keeps_original_exception():
    backend = FakeBackend()
    backend.read_fail_always = 'v'
    obj, sleeps = reader(backend)
    with obj.reuse_roms_datasets():
        with pytest.raises(RuntimeError, match='after 2 attempts') as err:
            obj.read_timestep(0, 'roms', max_retries=1, retry_delay=0)
    assert 'NetCDF: DAP failure' in str(err.value.__cause__)
    assert backend.opens == backend.closes == 2
    assert sleeps == [0]


def test_estimate_metadata_reopens_after_dap_failure():
    backend = FakeBackend()
    backend.metadata_fail_once = 'salt'
    obj, sleeps = reader(backend)
    if not hasattr(obj, 'estimate_roms_timestep_bytes'):
        pytest.skip('Phase 6A does not include metadata estimator')
    with obj.reuse_roms_datasets():
        size = obj.estimate_roms_timestep_bytes('roms', max_retries=1, retry_delay=0)
    assert size > 0
    assert (backend.opens, backend.closes, sleeps) == (2, 2, [0])


def test_transient_error_classifier_narrow():
    obj, _ = reader(FakeBackend())
    assert obj._is_retryable_roms_io_error(RuntimeError('NetCDF: DAP failure'))
    assert obj._is_retryable_roms_io_error(ConnectionError('reset'))
    assert not obj._is_retryable_roms_io_error(ValueError('invalid grid'))
    assert not obj._is_retryable_roms_io_error(RuntimeError('unsupported operation'))
