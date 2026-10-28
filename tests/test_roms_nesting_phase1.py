"""Phase 1 ROMS nesting tests. All data are synthetic, no THREDDS connection.

Run with: uv run python -W error::RuntimeWarning -m pytest tests/test_roms_nesting_phase1.py -v -rs
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from fvtools.grid.roms_grid import RomsDownloader
from fvtools.nesting import roms_nesting_fg as roms


def test_retry_transient_io_then_succeed(monkeypatch):
    reader = RomsDownloader()
    calls = []
    delays = []

    def read(index, path, fields, sigma):
        calls.append((index, path, sigma))
        if len(calls) < 3:
            raise OSError('Temporary connection reset')
        return 'OK'

    reader._load_roms_data = read
    monkeypatch.setattr('fvtools.grid.roms_grid.time_mod.sleep', delays.append)
    assert reader.read_timestep(2, 'file.nc', max_retries=4, retry_delay=0.5) == 'OK'
    assert len(calls) == 3
    assert delays == [0.5, 0.5]


def test_permanent_error_is_not_retried(monkeypatch):
    reader = RomsDownloader()
    calls = []

    def read(*args):
        calls.append(1)
        raise ValueError('wrong ROMS dimensions')

    reader._load_roms_data = read
    monkeypatch.setattr('fvtools.grid.roms_grid.time_mod.sleep',
                        lambda _: pytest.fail('Must not sleep on coding errors'))
    with pytest.raises(ValueError, match='wrong ROMS dimensions'):
        reader.read_timestep(2, 'file.nc')
    assert len(calls) == 1


def test_io_retry_is_bounded_and_preserves_cause(monkeypatch):
    reader = RomsDownloader()
    calls = []

    def read(*args):
        calls.append(1)
        raise OSError('not accessible')

    reader._load_roms_data = read
    monkeypatch.setattr('fvtools.grid.roms_grid.time_mod.sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='after 3 attempts') as exc:
        reader.read_timestep(0, 'missing.nc', max_retries=2, retry_delay=0)
    assert isinstance(exc.value.__cause__, OSError)
    assert len(calls) == 3


@pytest.mark.parametrize('bad', [-1, 1.5, 'many'])
def test_invalid_retry_count(bad):
    with pytest.raises(ValueError, match='max_retries'):
        RomsDownloader().read_timestep(0, 'bad.nc', max_retries=bad)


def test_filelist_sorted_overlap_and_all_files_closed(monkeypatch):
    # Day 2 provides a forecast that overlaps day 1 at t=6, but has t=18;
    # the old reverse scan accidentally removed t=12 from the day 1 file.
    sources = {
        'first.nc': np.array([0., 6., 12.]),
        'second.nc': np.array([6., 18., 24.]),
    }
    closed = []

    class FakeDataset:
        def __init__(self, path):
            self.path = path
            self.variables = {'ocean_time': SimpleNamespace(
                units='test', __getitem__=None)}

        def __enter__(self):
            self.variables = {'ocean_time': FakeVariable(sources[self.path])}
            return self

        def __exit__(self, *exc):
            closed.append(self.path)

    class FakeVariable:
        units = 'test'
        def __init__(self, values):
            self.values = values
        def __getitem__(self, index):
            return self.values[index]

    monkeypatch.setattr(roms.time_module, 'sleep', lambda n: None)
    monkeypatch.setattr(roms.netCDF4, 'Dataset', FakeDataset)
    monkeypatch.setattr(roms.netCDF4, 'num2date', lambda data, **kw: data)
    monkeypatch.setattr(roms.netCDF4, 'date2num', lambda data, **kw: data)

    class FakeROMS:
        def test_day(self, date):
            return 'first.nc' if date.day == 1 else 'second.nc'

    time, paths, indices = roms.make_fileList(datetime(2025, 1, 1), datetime(2025, 1, 2), FakeROMS())
    np.testing.assert_array_equal(time, [0., 6., 12., 18., 24.])
    assert paths == ['first.nc', 'second.nc', 'first.nc', 'second.nc', 'second.nc']
    assert indices == [0, 0, 2, 1, 2]
    assert closed == ['first.nc', 'second.nc']


def test_parallel_results_written_only_by_parent(monkeypatch):
    # Exercise the actual bounded future coordination with a thread-backed
    # executor rather than starting processes from pytest/Jupyter.
    class FakeExecutor(ThreadPoolExecutor):
        def __init__(self, *, max_workers, mp_context):
            super().__init__(max_workers=max_workers)

    class FakeBar:
        def __init__(self, **kw):
            self.steps = []
        def start(self):
            pass
        def update(self, n):
            self.steps.append(n)
        def finish(self):
            pass

    class Output:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            self.closed = True

    output = Output()
    monkeypatch.setattr(roms, 'ProcessPoolExecutor', FakeExecutor)
    monkeypatch.setattr(roms.pb, 'ProgressBar', FakeBar)
    monkeypatch.setattr(roms.netCDF4, 'Dataset', lambda *args: output)
    observed = []

    def worker(self, counter, path, idx, q=None):
        assert q is None
        return SimpleNamespace(counter=counter)

    def write(self, timestep):
        assert self.out is output
        observed.append(timestep.counter)

    monkeypatch.setattr(roms.Roms2FVCOMNest, '_worker', worker)
    monkeypatch.setattr(roms.Roms2FVCOMNest, 'dump_timestep_to_nest', write)
    instance = object.__new__(roms.Roms2FVCOMNest)
    instance.outfile = 'synthetic.nc'
    instance.path = ['a.nc'] * 13
    instance.index = list(range(13))
    instance.dump(nprocs=3)
    assert sorted(observed) == list(range(13))
    assert output.closed


def test_parallel_worker_failure_raises_and_closes_file(monkeypatch):
    class FakeExecutor(ThreadPoolExecutor):
        def __init__(self, *, max_workers, mp_context):
            super().__init__(max_workers=max_workers)
    class FakeBar:
        def __init__(self, **kw): pass
        def start(self): pass
        def update(self, n): pass
        def finish(self): pass
    class Output:
        closed = False
        def __enter__(self): return self
        def __exit__(self, *exc): self.closed = True
    output=Output()
    monkeypatch.setattr(roms, 'ProcessPoolExecutor', FakeExecutor)
    monkeypatch.setattr(roms.pb, 'ProgressBar', FakeBar)
    monkeypatch.setattr(roms.netCDF4, 'Dataset', lambda *args: output)
    def worker(self, counter, path, idx, q=None):
        raise ValueError('bad input field')
    monkeypatch.setattr(roms.Roms2FVCOMNest, '_worker', worker)
    instance=object.__new__(roms.Roms2FVCOMNest)
    instance.outfile='synthetic.nc'
    instance.path=['a.nc']*3
    instance.index=list(range(3))
    with pytest.raises(ValueError, match='bad input field'):
        instance.dump(nprocs=2)
    assert output.closed


def test_hyw_scalar_zero_fits_siglev_node_shape():
    obj=object.__new__(roms.Roms2FVCOMNest)
    obj.N4=SimpleNamespace(siglev=np.zeros((3, 4)), weight_node=np.ones(3),
                           weight_cell=np.ones(2))
    shape={
        'u': (1,3,2), 'v': (1,3,2), 'ua': (1,2), 'va': (1,2),
        'hyw': (1,4,3), 'zeta': (1,3), 'temp': (1,3,3),
        'salinity': (1,3,3), 'weight_node': (1,3), 'weight_cell': (1,2),
    }
    obj.out=SimpleNamespace(variables={name:np.full(dims,np.nan) for name,dims in shape.items()})
    timestep=SimpleNamespace(counter=0,u=np.ones((3,2)),v=np.ones((3,2)),
        ua=np.ones(2),va=np.ones(2),zeta=np.ones(3),
        temp=np.ones((3,3)),salt=np.ones((3,3)))
    obj.dump_timestep_to_nest(timestep)
    np.testing.assert_array_equal(obj.out.variables['hyw'],np.zeros((1,4,3)))


def test_netcdf_time_format_remains_float32_and_itime(monkeypatch, tmp_path):
    # Test an actual netCDF file, not only source-text assertions.
    from netCDF4 import Dataset
    m=SimpleNamespace(
        xn=np.array([[0.],[1.],[0.]]),yn=np.array([[0.],[0.],[1.]]),
        xc=np.array([.3]),yc=np.array([.3]),
        lat=np.array([60.,60.,61.]),lon=np.array([5.,6.,5.]),
        latc=np.array([60.3]),lonc=np.array([5.3]),
        h=np.ones(3)*5,hc=np.ones(1)*5,
        nv=np.array([[0,1,2]]),
        siglay=np.full((3,2),-.5),siglev=np.full((3,3),-.5),
        siglay_center=np.full((1,2),-.5),siglev_center=np.full((1,3),-.5),
    )
    dest=tmp_path/'out.nc'
    roms.create_nc_forcing_file(str(dest),m,'NKv3',np.array([61000.,61000.5]),False,'epsg:32633')
    with Dataset(dest) as ds:
        assert ds.variables['time'].dtype==np.dtype('float32')
        assert ds.variables['Itime'].dtype==np.dtype('int32')
        assert ds.variables['Itime2'].dtype==np.dtype('int32')
        assert ds.variables['hyw'].dimensions==('time','siglev','node')
        np.testing.assert_array_equal(ds.variables['time'][:], [61000.,61000.5])
