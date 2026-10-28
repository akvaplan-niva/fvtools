"""Offline tests for phase 6A ROMS serial NetCDF handle reuse.

Checks actual methods extracted from installed source, using a synthetic Dataset
backend. Never connects to THREDDS; no real external files are opened.
"""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / 'fvtools/grid/roms_grid.py'
NEST = ROOT / 'fvtools/nesting/roms_nesting_fg.py'


def _compile_reader(dataset_factory):
    root = ast.parse(GRID.read_text(encoding='utf-8'))
    node = next(n for n in root.body if isinstance(n, ast.ClassDef) and n.name == 'RomsDownloader')
    # A genuine extracted class avoids importing cartopy, OPeNDAP or other
    # grid modules, and still tests the methods that were actually patched.
    env = {
        'Dataset': dataset_factory,
        'np': np,
        'contextmanager': contextmanager,
        'time_mod': SimpleNamespace(sleep=lambda seconds: None),
        'ROMSTimeStep': lambda idx: SimpleNamespace(index=idx),
        'InputError': type('InputError', (Exception,), {}),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(GRID), 'exec'), env)
    reader = env['RomsDownloader']()
    params = {f'{name}_{xy}': value for name in ('r','u','v')
              for xy,value in (('i', 0), ('j', 0))}
    n4 = {}
    for c in ('r', 'u', 'v'):
        for prefix, value in (('m_',0),('x_',1)):
            for d in ('i','j'):
                n4[f'{prefix}{c}{d}'] = value
    reader.N4 = SimpleNamespace(**n4)
    return reader, env


class FakeDataset:
    def __init__(self, path, backend):
        self.path = str(path)
        self.backend = backend
        self.closed = False
        self.variables = {name: True for name in backend.arrays}
        backend.events.append(('open', self.path))

    def __getitem__(self, name):
        if self.closed:
            raise OSError('Dataset already closed')
        return FakeVariable(self, name)

    def close(self):
        if not self.closed:
            self.closed = True
            self.backend.events.append(('close', self.path))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeVariable:
    def __init__(self, ds, name):
        self.ds, self.name = ds, name

    def __getitem__(self, selector):
        # Verify exact indexes, including time and spatial slices.
        index = selector if isinstance(selector, tuple) else (selector,)
        selected = tuple((item.start, item.stop, item.step) if isinstance(item, slice)
                         else int(item) for item in index)
        self.ds.backend.events.append(('read', self.ds.path, self.name, selected))
        if self.ds.backend.fail_read is not None and self.ds.backend.fail_read(self.ds.path, self.name, selected):
            raise OSError('injected transient THREDDS error')
        if self.ds.backend.fail_code is not None:
            raise ValueError('injected non-retryable coding/data error')
        return self.ds.backend.arrays[self.name][selector].copy()


class Backend:
    def __init__(self):
        rng = np.random.default_rng(617)
        self.arrays = {}
        for name in ['salt', 'temp', 'u', 'v']:
            self.arrays[name] = rng.normal(size=(4, 3, 2, 2))
        for name in ['zeta', 'ubar', 'vbar']:
            self.arrays[name] = rng.normal(size=(4, 2, 2))
        self.events = []
        self.fail_read = None
        self.fail_code = None

    def __call__(self, path, *args, **kwargs):
        return FakeDataset(path, self)

    def events_of(self, kind):
        return [event for event in self.events if event[0] == kind]


def _values(steps):
    names = ('salt','temp','zeta','u','v','ua','va')
    return [{name: np.ma.getdata(getattr(t,name)).copy() for name in names} for t in steps]


def test_default_api_opens_and_closes_each_read():
    backend = Backend()
    reader, _ = _compile_reader(backend)
    reader.read_timestep(0, 'file_A')
    reader.read_timestep(1, 'file_A')
    assert backend.events_of('open') == [('open','file_A')]*2
    assert backend.events_of('close') == [('close','file_A')]*2
    assert len(backend.events_of('read')) == 14


@pytest.mark.parametrize('paths', [
    ['A','A','A'], ['A','A','B','B','A'], ['A'], []
])
def test_only_one_open_dataset_and_exact_same_reads(paths):
    baseline = Backend()
    original, _ = _compile_reader(baseline)
    expected = [original.read_timestep(i % 3, path) for i,path in enumerate(paths)]
    optimized = Backend()
    reader, _ = _compile_reader(optimized)
    with reader.reuse_roms_datasets():
        actual = [reader.read_timestep(i % 3, path) for i,path in enumerate(paths)]
        # No source connection is left alive when switching files.
        assert len(optimized.events_of('open')) - len(optimized.events_of('close')) == (1 if paths else 0)
    for a,b in zip(_values(actual), _values(expected)):
        for key in a:
            np.testing.assert_array_equal(a[key],b[key])
    assert optimized.events_of('read') == baseline.events_of('read')
    contiguous_groups = sum(i == 0 or paths[i] != paths[i-1] for i in range(len(paths)))
    assert len(optimized.events_of('open')) == contiguous_groups
    assert len(optimized.events_of('close')) == contiguous_groups


def test_broken_connection_evicted_before_retry():
    backend = Backend()
    count = [0]
    def fail_once(path, name, index):
        if name == 'temp' and index[0] == 1 and count[0] == 0:
            count[0] += 1
            return True
        return False
    backend.fail_read = fail_once
    reader, _ = _compile_reader(backend)
    with reader.reuse_roms_datasets():
        reader.read_timestep(0, 'A', max_retries=1, retry_delay=0)
        reader.read_timestep(1, 'A', max_retries=1, retry_delay=0)
        reader.read_timestep(2, 'A', max_retries=1, retry_delay=0)
    assert count == [1]
    assert backend.events_of('open') == [('open','A')]*2
    assert backend.events_of('close') == [('close','A')]*2


def test_non_retryable_error_closes_scope():
    backend = Backend()
    reader, _ = _compile_reader(backend)
    backend.fail_code = True
    with pytest.raises(ValueError, match='non-retryable'):
        with reader.reuse_roms_datasets():
            reader.read_timestep(0,'A', max_retries=5,retry_delay=0)
    assert len(backend.events_of('open')) == 1
    assert len(backend.events_of('close')) == 1
    assert len(backend.events_of('read')) == 1


def test_nested_scopes_not_allowed():
    backend = Backend()
    reader, _ = _compile_reader(backend)
    with reader.reuse_roms_datasets():
        with pytest.raises(RuntimeError,match='Nested'):
            with reader.reuse_roms_datasets():
                pass
        reader.read_timestep(0,'A')
    assert len(backend.events_of('open')) == 1
    assert len(backend.events_of('close')) == 1


def test_invalid_path_open_failure_does_not_leave_cache_dirty():
    backend = Backend()
    attempts = []
    def factory(path):
        attempts.append(path)
        if path == 'bad':
            raise OSError('unreachable')
        return backend(path)
    reader, _ = _compile_reader(factory)
    with reader.reuse_roms_datasets():
        reader.read_timestep(0,'good',max_retries=0)
        with pytest.raises(RuntimeError,match='after 2 attempts'):
            reader.read_timestep(0,'bad',max_retries=1,retry_delay=0)
        reader.read_timestep(1,'good',max_retries=0)
    assert attempts == ['good','bad','bad','good']
    assert backend.events_of('close') == [('close','good')]*2


def test_serial_dump_uses_scope_and_preserves_timestep_sequence():
    backend = Backend()
    reader, env = _compile_reader(backend)
    tree = ast.parse(NEST.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Roms2FVCOMNest')
    method = next(m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == 'dump_single')
    wrapper = ast.ClassDef(name='SerialNesting', bases=[ast.Name(id='RomsDownloader',ctx=ast.Load())],
                           keywords=[],body=[method],decorator_list=[])
    class FakeBar:
        def __init__(self, *args,**kwargs): pass
        def start(self): pass
        def update(self,n): pass
        def finish(self): pass
    output = SimpleNamespace(__enter__=None)
    class FakeOutput:
        def __init__(self,*args,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
    env['netCDF4'] = SimpleNamespace(Dataset=FakeOutput)
    env['pb'] = SimpleNamespace(ProgressBar=FakeBar,Percentage=lambda: None,
                                BouncingBar=lambda: None,ETA=lambda: None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(NEST),'exec'),env)
    serial = env['SerialNesting']()
    serial.N4 = reader.N4
    serial.path = ['A','A','B','B']
    serial.index = [0,1,2,0]
    serial.outfile = 'unused'
    counters = []
    serial._prepare_timestep = lambda counter,path,index: SimpleNamespace(
        counter=counter, zeta=serial.read_timestep(index,path).zeta)
    serial.dump_timestep_to_nest = lambda step: counters.append((step.counter,step.zeta.copy()))
    serial.dump_single()
    assert [i for i,_ in counters] == [0,1,2,3]
    assert backend.events_of('open') == [('open','A'),('open','B')]
    assert backend.events_of('close') == [('close','A'),('close','B')]
    assert len(backend.events_of('read')) == 4*7
