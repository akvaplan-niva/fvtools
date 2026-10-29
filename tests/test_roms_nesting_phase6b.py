"""Offline regression tests for opt-in ROMS contiguous-time NetCDF reads.

Read actual installed source AST; do not contact THREDDS.
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


class Backend:
    def __init__(self, *, masked=False, with_bars=True):
        rng = np.random.default_rng(621)
        self.arrays = {}
        for name in ('salt', 'temp', 'u', 'v'):
            self.arrays[name] = rng.standard_normal((8, 3, 2, 2)).astype('f4')
        for name in ('zeta', 'ubar', 'vbar'):
            self.arrays[name] = rng.standard_normal((8, 2, 2)).astype('f8')
        if not with_bars:
            del self.arrays['ubar']
            del self.arrays['vbar']
        if masked:
            for k, a in list(self.arrays.items()):
                m = np.zeros(a.shape, dtype=bool)
                m[2, ..., 1, 1] = True
                self.arrays[k] = np.ma.array(a, mask=m)
        self.events = []
        self.fail_once = None
        self.fail_count = 0

    def __call__(self, path, *args, **kwargs):
        return Dataset(self, path)

    def count(self, kind):
        return sum(v[0] == kind for v in self.events)


class Variable:
    def __init__(self, ds, name):
        self.ds = ds
        self.name = name
        self.shape = ds.backend.arrays[name].shape
        self.dtype = ds.backend.arrays[name].dtype

    def __getitem__(self, key):
        if self.ds.closed:
            raise OSError('source file closed')
        self.ds.backend.events.append(('read', self.ds.path, self.name, key))
        if self.ds.backend.fail_once == self.name and self.ds.backend.fail_count == 0:
            self.ds.backend.fail_count += 1
            raise OSError('temporary broken connection')
        return self.ds.backend.arrays[self.name][key].copy()


class Dataset:
    def __init__(self, backend, path):
        self.backend = backend
        self.path = path
        self.closed = False
        self.variables = {n: Variable(self, n) for n in backend.arrays}
        self.backend.events.append(('open', path))

    def __getitem__(self, name):
        if name not in self.variables:
            raise KeyError(name)
        return self.variables[name]

    def close(self):
        if not self.closed:
            self.closed = True
            self.backend.events.append(('close', self.path))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def make_reader(backend):
    grid_ast = ast.parse(GRID.read_text(encoding='utf-8'))
    cls = next(n for n in grid_ast.body if isinstance(n, ast.ClassDef)
               and n.name == 'RomsDownloader')
    env = {'np': np, 'contextmanager': contextmanager,
           'Dataset': backend, 'time_mod': SimpleNamespace(sleep=lambda delay: None),
           'InputError': ValueError,
           'ROMSTimeStep': lambda n: SimpleNamespace(index=n)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(GRID), 'exec'), env)
    reader = env['RomsDownloader']()
    values = {}
    for c in ('r', 'u', 'v'):
        for coord in ('i', 'j'):
            values['m_'+c+coord] = 0
            values['x_'+c+coord] = 1
    values['cropped_dsigma_u'] = np.ones((3, 2, 2), dtype='f4') / 3
    values['cropped_dsigma_v'] = np.ones((3, 2, 2), dtype='f4') / 3
    reader.N4 = SimpleNamespace(**values)
    return reader, env


def compare_step(actual, expected):
    for name in ('salt','temp','zeta','u','v','ua','va'):
        a, b = getattr(actual, name), getattr(expected, name)
        assert a.dtype == b.dtype, name
        assert a.shape == b.shape, name
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b))


@pytest.mark.parametrize('masked', [False, True])
def test_block_values_match_legacy_reading(masked):
    baseline = Backend(masked=masked)
    old, _ = make_reader(baseline)
    expected = [old.read_timestep(i, 'A') for i in (1, 2, 3)]
    new_backend = Backend(masked=masked)
    new, _ = make_reader(new_backend)
    with new.reuse_roms_datasets():
        results = new.read_timestep_block(1, 3, 'A')
    for a, b in zip(results, expected):
        compare_step(a, b)
    assert len(results) == 3
    assert baseline.count('read') == 21
    assert new_backend.count('read') == 7  # NetCDF gets per variable, not HTTP requests
    assert new_backend.count('open') == 1
    assert new_backend.count('close') == 1
    for event in (x for x in new_backend.events if x[0]=='read'):
        assert event[3][0] == slice(1,4), event


def test_barotropic_fallback_equivalent_when_ubar_vbar_missing():
    old, _ = make_reader(Backend(with_bars=False))
    expected = [old.read_timestep(i, 'A') for i in (0, 1, 2)]
    new, _ = make_reader(Backend(with_bars=False))
    with new.reuse_roms_datasets():
        actual = new.read_timestep_block(0, 3, 'A')
    for a, b in zip(actual, expected):
        compare_step(a, b)


def test_dataset_error_retries_and_invalidates_connection():
    backend = Backend()
    backend.fail_once = 'temp'
    reader, _ = make_reader(backend)
    with reader.reuse_roms_datasets():
        steps = reader.read_timestep_block(1, 2, 'A', max_retries=1, retry_delay=0)
        assert len(steps) == 2
    assert backend.fail_count == 1
    assert backend.count('open') == 2
    assert backend.count('close') == 2


def test_error_exhaustion_is_bounded():
    backend = Backend()
    backend.fail_once = 'temp'
    reader, _ = make_reader(backend)
    with reader.reuse_roms_datasets():
        with pytest.raises(RuntimeError,match='after 1 attempts'):
            reader.read_timestep_block(0,2,'A', max_retries=0)
    assert backend.count('open') == backend.count('close') == 1


@pytest.mark.parametrize('count', [0,-1,1.5,True])
def test_invalid_block_size_rejected(count):
    reader, _ = make_reader(Backend())
    with pytest.raises(ValueError, match='count'):
        reader.read_timestep_block(0,count,'A')


def test_block_does_not_permanently_cache_when_scope_is_missing():
    backend = Backend()
    reader, _ = make_reader(backend)
    reader.read_timestep_block(1,2,'A')
    reader.read_timestep_block(3,2,'A')
    assert backend.count('open') == backend.count('close') == 2


def make_serial(backend, paths, indices):
    reader, env = make_reader(backend)
    cls = next(n for n in ast.parse(NEST.read_text(encoding='utf-8')).body
               if isinstance(n, ast.ClassDef) and n.name == 'Roms2FVCOMNest')
    # The serial methods are run from the actual installed class; worker imports
    # are not needed because multiprocessing is not called.
    class FakeBar:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def update(self, *args): pass
        def finish(self): pass
    class FakeOutput:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
    env['RomsDownloader'] = type(reader)
    env['LinearInterpolation'] = object
    env['netCDF4'] = SimpleNamespace(Dataset=FakeOutput)
    env['pb'] = SimpleNamespace(ProgressBar=FakeBar, Percentage=lambda:None,
                                 BouncingBar=lambda:None, ETA=lambda:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])), str(NEST), 'exec'), env)
    s = env['Roms2FVCOMNest'].__new__(env['Roms2FVCOMNest'])
    s.N4 = reader.N4
    s.path = paths
    s.index = indices
    s.outfile = 'fake.nc'
    saved = []
    s.crop_and_transpose = lambda timestep: timestep
    s.horizontal_interpolation = lambda timestep: timestep
    s.vertical_interpolation = lambda timestep: timestep
    s.adjust_uv = lambda timestep: timestep
    s.dump_timestep_to_nest = lambda t: saved.append((t.counter, t))
    return s, saved


def test_default_serial_remains_legacy_reads():
    backend = Backend()
    serial, saved = make_serial(backend, ['A']*4,[1,2,3,4])
    serial.dump_single()
    assert [idx for idx,_ in saved] == list(range(4))
    assert backend.count('read') == 28
    assert backend.count('open') == backend.count('close') == 1


def test_serial_opt_in_groups_only_consecutive_timesteps_and_files():
    paths = ['A','A','A','B','B','A','A']
    indices = [0,1,3,0,1,5,6]
    backend = Backend()
    serial, saved = make_serial(backend,paths,indices)
    serial.dump_single(time_block_size=3,max_block_mb=100)
    assert [idx for idx,_ in saved] == list(range(7))
    assert [step.index for _,step in saved] == indices
    assert backend.count('read') == 4*7
    assert backend.count('open') == backend.count('close') == 3
    slices = [v[3][0] for v in backend.events if v[0]=='read' and v[2]=='salt']
    assert slices == [slice(0,2), 3, slice(0,2), slice(5,7)]


def test_memory_budget_falls_back_to_single_timestep_reads():
    backend = Backend()
    serial, saved = make_serial(backend,['A']*3,[0,1,2])
    serial.dump_single(time_block_size=3,max_block_mb=0.000001)
    assert len(saved) == 3
    assert backend.count('read') == 21
    assert backend.count('open') == backend.count('close') == 1


def test_invalid_serial_parameters_fail_before_any_reads():
    backend = Backend()
    serial, _ = make_serial(backend,['A','A'],[0,1])
    with pytest.raises(ValueError, match='max_block_mb'):
        serial.dump_single(time_block_size=2,max_block_mb=-1)
    assert backend.count('read') == 0


def test_source_signatures_keep_default_disabled():
    t = ast.parse(NEST.read_text(encoding='utf-8'))
    main = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name=='main')
    params = {arg.arg: val.value for arg,val in zip(main.args.args[-len(main.args.defaults):],main.args.defaults)
              if isinstance(val, ast.Constant)}
    assert params['time_block_size'] == 1
    assert params['max_block_mb'] == 128
