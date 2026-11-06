"""ROMS / NorKyst THREDDS fail-closed nesting regression tests (offline)."""
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import pytest

from fvtools.setup.nesting import roms


def _n4():
    ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return SimpleNamespace(rho_index=ids, u_index=ids, v_index=ids)


def _timestep(*, missing=False, outside=False):
    result = SimpleNamespace(
        salt=np.ma.array(np.ones((5, 2), dtype=float)),
        temp=np.ones((5, 2)),
        zeta=np.ones(5),
        u=np.ones((5, 2)),
        v=np.ones((5, 2)),
        ua=np.ones(5),
        va=np.ones(5),
    )
    if missing:
        result.salt[1, 0] = np.ma.masked
    if outside:
        result.salt[4, 0] = np.ma.masked
    return result


def test_incomplete_interpolation_corner_rejected_but_unused_land_ignored():
    with pytest.raises(roms.IncompleteROMSTimestep, match='salt'):
        roms._check_roms_nesting_source(_timestep(missing=True), _n4())
    roms._check_roms_nesting_source(_timestep(outside=True), _n4())


@pytest.mark.parametrize('bad', [np.nan, np.inf, np.float64(9.969209968386869e36)])
def test_missing_unmasked_fill_values_are_rejected(bad):
    field = _timestep()
    field.temp[1, 0] = bad
    with pytest.raises(roms.IncompleteROMSTimestep, match='temp'):
        roms._check_roms_nesting_source(field, _n4())


def _prepare_test_instance(monkeypatch, source):
    cls = roms.Roms2FVCOMNest
    obj = object.__new__(cls)
    obj.N4 = _n4()
    obj.incomplete_timestep_retries = 2
    obj.incomplete_timestep_retry_delay = 0
    events = []

    def reader(self, index, path):
        events.append(('read', index))
        return next(source)

    def identity(self, timestep):
        events.append('interpolate')
        return timestep

    monkeypatch.setattr(cls, 'read_timestep', reader)
    for name in ('crop_and_transpose', 'horizontal_interpolation',
                 'vertical_interpolation', 'adjust_uv'):
        monkeypatch.setattr(cls, name, identity)
    monkeypatch.setattr(roms.time_module, 'sleep', lambda secs: None)

    class Pool:
        def invalidate(self):
            events.append('invalidate')

    obj._roms_reader_cache = Pool()
    return obj, events


def test_invalid_source_restarts_entire_timestep_then_succeeds(monkeypatch):
    obj, events = _prepare_test_instance(monkeypatch, iter([
        _timestep(missing=True), _timestep(),
    ]))
    result = obj._prepare_timestep(7, 'nor.nc', 3)
    assert result.counter == 7
    assert [e for e in events if isinstance(e, tuple)] == [('read', 3), ('read', 3)]
    assert 'invalidate' in events
    # The first attempt stops before expensive horizontal interpolation.
    assert events.count('interpolate') == 5  # 1 crop on bad + 4 stages on good


def test_unrecoverable_incomplete_step_is_not_written(monkeypatch):
    obj, events = _prepare_test_instance(monkeypatch, iter([
        _timestep(missing=True) for _ in range(3)
    ]))
    with pytest.raises(RuntimeError, match='after 3 attempts') as err:
        obj._prepare_timestep(2, 'nor.nc', 3)
    assert isinstance(err.value.__cause__, roms.IncompleteROMSTimestep)
    assert len([e for e in events if isinstance(e, tuple)]) == 3
    assert events.count('invalidate') == 3
    assert events.count('interpolate') == 3  # only the source cropping ran


def test_interpolated_invalid_output_is_not_accepted(monkeypatch):
    obj, events = _prepare_test_instance(monkeypatch, iter([_timestep()]))

    def corrupt(self, timestep):
        timestep.u[0, 0] = np.nan
        return timestep

    monkeypatch.setattr(roms.Roms2FVCOMNest, 'adjust_uv', corrupt)
    with pytest.raises(ValueError, match='Invalid interpolated ROMS nesting u'):
        obj._prepare_timestep(0, 'nor.nc', 0)
    assert [e for e in events if isinstance(e, tuple)] == [('read', 0)]


def test_norkyst_availability_io_retries_instead_of_reporting_missing(monkeypatch):
    # NorKyst.test_ncfile may turn an OSError into NoAvailableData. The
    # filelist builder must retry the full probe before proceeding.
    calls = []
    class Availability:
        def test_day(self, date):
            calls.append(date)
            if len(calls) < 3:
                raise roms.NoAvailableData('temporary DAP issue')
            return 'nor.nc'

    class FakeVariable:
        units = 'seconds since 2025-01-01'
        def __getitem__(self, sel):
            return np.array([0., 3600., 7200.])
    class Dataset:
        def __init__(self, *a, **kw):
            self.variables = {'ocean_time': FakeVariable()}
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(roms.netCDF4, 'Dataset', Dataset)
    monkeypatch.setattr(roms.netCDF4, 'num2date', lambda arr, **kw: arr)
    monkeypatch.setattr(roms.netCDF4, 'date2num', lambda arr, **kw: arr)
    monkeypatch.setattr(roms.time_module, 'sleep', lambda _: None)
    times, paths, indices = roms.make_fileList(
        datetime(2025, 4, 1), datetime(2025, 4, 1), Availability())
    assert len(calls) == 3
    np.testing.assert_array_equal(times, [0., 3600., 7200.])
    assert paths == ['nor.nc'] * 3
    assert indices == [0, 1, 2]


def test_missing_day_aborts_instead_of_creating_timing_gap(monkeypatch):
    class Missing:
        def test_day(self, date):
            raise roms.NoAvailableData('file not yet available')
    monkeypatch.setattr(roms.time_module, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='no day was skipped'):
        roms.make_fileList(datetime(2025, 4, 1), datetime(2025, 4, 2), Missing())
