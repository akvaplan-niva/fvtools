"""Phase 4 integration tests: no downloads, no live THREDDS/FVCOM files.

Exercises exact public schemas, serial/worker stage order and transactional
output behaviour. Existing phase 1-3 tests remain the numerical references.
"""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from fvtools.prepro.nesting import roms_nesting_fg as mod


def _fake_nest():
    return SimpleNamespace(
        xn=np.array([0., 1., 0.]), yn=np.array([0., 0., 1.]),
        xc=np.array([.3333]), yc=np.array([.3333]),
        lon=np.array([5., 6., 5.]), lat=np.array([60., 60., 61.]),
        lonc=np.array([5.3333]), latc=np.array([60.3333]),
        h=np.array([10., 12., 14.]), hc=np.array([12.]),
        nv=np.array([[0, 1, 2]], dtype=np.int32),
        siglay=np.full((3, 2), -.5), siglev=np.full((3, 3), -.5),
        siglay_center=np.full((1, 2), -.5),
        siglev_center=np.full((1, 3), -.5),
    )


def test_public_netcdf_schema_retains_type_dimensions_and_time_formula(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'gmtime', lambda: (2026, 10, 28, 12, 0, 0, 2, 301, 0))
    fn = tmp_path / 'forcing.nc'
    nest = _fake_nest()
    times = np.array([61000., 61000.5, 61000.25])
    mod.create_nc_forcing_file(str(fn), nest, 'NKv3', times, False, 'epsg:32633')
    expected = {
        'time': (np.float32, ('time',)),
        'Itime': (np.int32, ('time',)),
        'Itime2': (np.int32, ('time',)),
        'lon': (np.float32, ('node',)),
        'lat': (np.float32, ('node',)),
        'x': (np.float32, ('node',)),
        'y': (np.float32, ('node',)),
        'h': (np.float32, ('node',)),
        'lonc': (np.float32, ('nele',)),
        'latc': (np.float32, ('nele',)),
        'xc': (np.float32, ('nele',)),
        'yc': (np.float32, ('nele',)),
        'h_center': (np.float32, ('nele',)),
        'nv': (np.int32, ('three', 'nele')),
        'siglay': (np.float32, ('siglay', 'node')),
        'siglev': (np.float32, ('siglev', 'node')),
        'siglay_center': (np.float32, ('siglay', 'nele')),
        'siglev_center': (np.float32, ('siglev', 'nele')),
        'weight_node': (np.float32, ('time', 'node')),
        'weight_cell': (np.float32, ('time', 'nele')),
        'zeta': (np.float32, ('time', 'node')),
        'ua': (np.float32, ('time', 'nele')),
        'va': (np.float32, ('time', 'nele')),
        'u': (np.float32, ('time', 'siglay', 'nele')),
        'v': (np.float32, ('time', 'siglay', 'nele')),
        'temp': (np.float32, ('time', 'siglay', 'node')),
        'salinity': (np.float32, ('time', 'siglay', 'node')),
        'hyw': (np.float32, ('time', 'siglev', 'node')),
    }
    with netCDF4.Dataset(fn) as nc:
        assert set(nc.variables) == set(expected)
        for name, (dtype, dimensions) in expected.items():
            var = nc.variables[name]
            assert var.dtype == np.dtype(dtype), name
            assert var.dimensions == dimensions, name
        np.testing.assert_array_equal(nc.variables['time'][:], times.astype('f4'))
        np.testing.assert_array_equal(nc.variables['Itime'][:], np.floor(times))
        np.testing.assert_array_equal(nc.variables['Itime2'][:],
            np.round((times - np.floor(times)) * 60 * 60 * 1000, decimals=0) * 24)
        np.testing.assert_array_equal(nc.variables['nv'][:], (nest.nv + 1).T)
        np.testing.assert_array_equal(nc.variables['h'][:], nest.h)
        assert nc.interpolation_projection == 'epsg:32633'
        assert nc.mother_model == 'Using data nested from NKv3'
        assert nc.created.endswith('using roms_nesting_fg.py')


def test_common_stage_order_serial_and_worker_identical(monkeypatch):
    cls = mod.Roms2FVCOMNest
    obj = object.__new__(cls)
    events = []
    step = SimpleNamespace()
    def read(self, index, path):
        events.append(('read', index, path))
        return step
    def stage(name):
        def call(self, value):
            assert value is step
            events.append(name)
            return value
        return call
    monkeypatch.setattr(cls, 'read_timestep', read)
    for name in ('crop_and_transpose', 'horizontal_interpolation',
                 'vertical_interpolation', 'adjust_uv'):
        monkeypatch.setattr(cls, name, stage(name))
    output = obj._prepare_timestep(4, 'roms.nc', 8)
    assert output is step
    assert step.counter == 4
    expected = [('read', 8, 'roms.nc'), 'crop_and_transpose',
                'horizontal_interpolation', 'vertical_interpolation', 'adjust_uv']
    assert events == expected
    events.clear()
    worker_output = obj._worker(4, 'roms.nc', 8)
    assert worker_output is step
    assert events == expected
    assert step.counter == 4
    q = SimpleNamespace(items=[])
    def put(item, timeout):
        q.items.append((item, timeout))
    q.put = put
    assert obj._worker(7, 'roms.nc', 8, q=q) == ''
    assert q.items == [(step, 10)]
    assert step.counter == 7


def test_serial_writer_uses_shared_pipeline_and_closes_dataset(monkeypatch):
    calls = []
    cls = mod.Roms2FVCOMNest
    obj = object.__new__(cls)
    obj.outfile = 'dummy.nc'
    obj.path = ['one.nc', 'two.nc', 'three.nc']
    obj.index = [3, 1, 2]
    class FakeBar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def start(self): pass
        def update(self, value): calls.append(('bar', value))
        def finish(self): pass
    class FakeNC:
        closed = False
        def __enter__(self): return self
        def __exit__(self, *exc): self.closed = True
    out = FakeNC()
    monkeypatch.setattr(mod.pb, 'ProgressBar', FakeBar)
    monkeypatch.setattr(mod.netCDF4, 'Dataset', lambda *args: out)
    def prep(self, counter, path, index):
        calls.append(('prepare', counter, path, index))
        return SimpleNamespace(counter=counter)
    def dump(self, step):
        assert self.out is out
        calls.append(('write', step.counter))
    monkeypatch.setattr(cls, '_prepare_timestep', prep)
    monkeypatch.setattr(cls, 'dump_timestep_to_nest', dump)
    obj.dump_single()
    assert out.closed
    assert [x for x in calls if x[0] == 'prepare'] == [
        ('prepare', 0, 'one.nc', 3),
        ('prepare', 1, 'two.nc', 1),
        ('prepare', 2, 'three.nc', 2),
    ]
    assert [x for x in calls if x[0] == 'write'] == [
        ('write', 0), ('write', 1), ('write', 2),
    ]


@pytest.mark.parametrize('parallel', [False, True])
def test_atomic_publication_keeps_existing_file_on_error(monkeypatch, tmp_path, parallel):
    target = tmp_path / 'nest.nc'
    target.write_bytes(b'ORIGINAL')
    history = []
    def create(path, *args):
        Path(path).write_bytes(b'PARTIAL')
        history.append('created')
    class Broken:
        def __init__(self, outfile, *args):
            self.outfile = outfile
            history.append('constructed')
        def dump_single(self):
            history.append('single')
            raise OSError('synthetic failure')
        def dump(self, nprocs=None):
            history.append('parallel')
            raise OSError('synthetic failure')
    monkeypatch.setattr(mod, 'create_nc_forcing_file', create)
    monkeypatch.setattr(mod, 'Roms2FVCOMNest', Broken)
    with pytest.raises(OSError, match='synthetic failure'):
        mod._publish_nesting_file(str(target), object(), 'NKv3', np.array([61000.]),
                                  False, 'epsg:32633', ['x.nc'], [0], object(),
                                  2 if parallel else 1)
    assert target.read_bytes() == b'ORIGINAL'
    assert history == ['created', 'constructed', 'parallel' if parallel else 'single']
    assert sorted(x.name for x in tmp_path.iterdir()) == ['nest.nc']


def test_atomic_publication_replaces_only_successful_product(monkeypatch, tmp_path):
    target = tmp_path / 'nest.nc'
    target.write_bytes(b'ORIGINAL')
    def create(path, *args):
        Path(path).write_bytes(b'COMPLETE')
    class Good:
        def __init__(self, outfile, *args):
            self.outfile = outfile
        def dump_single(self):
            with Path(self.outfile).open('ab') as f:
                f.write(b'+DATA')
        def dump(self, nprocs=None):
            self.dump_single()
    monkeypatch.setattr(mod, 'create_nc_forcing_file', create)
    monkeypatch.setattr(mod, 'Roms2FVCOMNest', Good)
    mod._publish_nesting_file(str(target), object(), 'NKv3', np.array([61000.]),
                              False, 'epsg:32633', ['x.nc'], [0], object(), 1)
    assert target.read_bytes() == b'COMPLETE+DATA'
    assert sorted(x.name for x in tmp_path.iterdir()) == ['nest.nc']


def test_interpolator_preparation_still_matches_legacy_order(monkeypatch):
    seen = []
    coarse, mother, nest = object(), SimpleNamespace(reference='epsg:32633'), SimpleNamespace(x=[1], y=[2], tri=[]) 
    class FakeN4:
        def __init__(self, roms, x, y, tri, uv):
            seen.append('init_n4')
            assert roms is coarse and x is nest.x and y is nest.y and tri is nest.tri and uv
        def nearest4(self, M):
            assert M is mother
            seen.append('nearest4')
    class FakeTopo:
        def __init__(self, roms, M, NEST, latlon):
            assert roms is coarse and M is mother and NEST is nest and not latlon
            seen.append('topo_init')
        def add_ROMS_bathymetry_to_FVCOM_and_NEST(self):
            seen.append('topo_match')
            return mother, nest
    def vertical(n4):
        assert n4.FV is nest
        seen.append('vertical')
        return n4
    monkeypatch.setattr(mod, 'N4ROMSNESTING', FakeN4)
    monkeypatch.setattr(mod, 'MatchTopo', FakeTopo)
    monkeypatch.setattr(mod.vi, 'add_vertical_interpolation2N4', vertical)
    got_n4, got_m, got_nest = mod._prepare_nesting_interpolator(coarse, mother, nest, False)
    assert isinstance(got_n4, FakeN4)
    assert got_m is mother and got_nest is nest
    assert seen == ['init_n4', 'nearest4', 'topo_init', 'topo_match', 'vertical']
