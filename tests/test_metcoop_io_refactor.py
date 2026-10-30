"""Offline MetCoOp regressions: bounded OPeNDAP reads and unchanged field values."""

from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.prepro.atm import read_metCoop as met
from fvtools.prepro.atm._metcoop_io import SourcePool, is_retryable_error, read_masked


FIELDS = {
    'Uwind': 'x_wind_10m',
    'Vwind': 'y_wind_10m',
    'Tair': 'air_temperature_2m',
    'Pair': 'air_pressure_at_sea_level',
    'relative_humidity': 'relative_humidity_2m',
    'caf': 'cloud_area_fraction',
    'precip': 'precipitation_amount_acc',
    'evap': 'water_evaporation_amount',
    'spechum': 'specific_humidity_2m',
    'SwD': 'integral_of_surface_net_downward_shortwave_flux_wrt_time',
    'andLw': 'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time',
}


class RecordingVariable:
    def __init__(self, data):
        self.data = data
        self.keys = []

    def __getitem__(self, key):
        self.keys.append(key)
        return self.data[key]


class RecordingDataset:
    def __init__(self, variables, path, closed):
        self.variables = variables
        self.path = path
        self._closed = closed

    def close(self):
        self._closed.append(self.path)


def make_sources(layout):
    """Generate distinct old-format, new-format, and accumulated fields."""
    mask = np.zeros((6, 7), dtype=bool)
    mask[1:3, 2:4] = True
    mask[4, 5] = True
    n4 = SimpleNamespace(fv_domain_mask=mask, fv_domain_mask_ex=mask,
                         nindex=np.array([[0]]), ncoef=np.ones((1, 1)))
    sources = {}
    if layout == 'pp':
        main, acc, rad = 'old_pp_file.nc', 'old_extracted_acc.nc', 'old_extracted_rad.nc'
    elif layout == 'full':
        main, acc, rad = 'old_pp_file.nc', 'old_full_backup.nc', 'old_subset_rad.nc'
    elif layout == 'extracted':
        main, acc, rad = 'old_extracted_main.nc', 'old_extracted_acc.nc', 'old_extracted_rad.nc'
    elif layout == 'subset':
        main, acc, rad = 'old_subset_main.nc', 'old_extracted_acc.nc', 'old_subset_rad.nc'
    else:
        main = acc = rad = 'new_det_20201030.nc'

    def shape(path, name):
        if path == main and layout in ('pp', 'full') and name in (
            'x_wind_10m', 'y_wind_10m', 'cloud_area_fraction'):
            return (5, 6, 7)
        if 'subset' in path or '_full_' in path:
            return (5, 1, 1, 6, 7)
        return (5, 1, 6, 7)

    needed = {main: ['Uwind', 'Vwind', 'Tair', 'Pair', 'relative_humidity', 'caf', 'precip']}
    if layout == 'det':
        needed[main] += ['evap', 'spechum', 'SwD', 'andLw']
    else:
        needed.setdefault(acc, []).extend(['evap', 'spechum'])
        needed.setdefault(rad, []).extend(['SwD', 'andLw'])

    for p, names in needed.items():
        v = {}
        for k in names:
            data = np.arange(np.prod(shape(p, FIELDS[k])), dtype=float).reshape(shape(p, FIELDS[k]))
            data = data + 100.0 * (list(FIELDS).index(k) + 1)
            if k in ('precip', 'evap', 'SwD', 'andLw'):
                data = data * np.arange(1, 6).reshape((5,) + (1,) * (data.ndim - 1))
            v[FIELDS[k]] = RecordingVariable(data)
        sources[p] = v
    return n4, (main, acc, rad), sources


def legacy_reference(layout, paths, sources, mask, index):
    main, acc, rad = paths
    def take(path, field, prefix):
        return sources[path][FIELDS[field]].data[(*prefix, slice(None), slice(None))][..., mask]
    if layout == 'pp' or layout == 'full':
        wind, std, cloud, two = (index,), (index, 0), (index,), ([index - 1, index], 0)
    elif layout == 'subset':
        wind = std = cloud = (index, 0, 0)
        two = ([index - 1, index], 0, 0)
    else:
        wind = std = cloud = (index, 0)
        two = ([index - 1, index], 0)
    output = {k: take(main, k, prefix) for k, prefix in (
        ('Uwind', wind), ('Vwind', wind), ('Tair', std), ('Pair', std),
        ('relative_humidity', std), ('caf', cloud), ('precip', two))}
    if layout in ('pp', 'full'):
        output['caf'] *= 0.01
    if layout == 'det':
        for k, prefix in [('evap', two), ('spechum', std), ('SwD', two), ('andLw', two)]:
            output[k] = take(main, k, prefix)
    else:
        if layout == 'full':
            accstd, acctwo, radtwo = (index, 0, 0), ([index - 1, index], 0, 0), ([index - 1, index], 0, 0)
        else:
            accstd, acctwo = (index, 0), ([index - 1, index], 0)
            radtwo = ([index - 1, index], 0) if 'extracted_' in rad else ([index - 1, index], 0, 0)
        output['evap'] = take(acc, 'evap', acctwo)
        output['spechum'] = take(acc, 'spechum', accstd)
        output['SwD'] = take(rad, 'SwD', radtwo)
        output['andLw'] = take(rad, 'andLw', radtwo)
    for name in ('SwD', 'andLw'):
        output[name] = (output[name][1] - output[name][0]) / 3600.0
    for name in ('precip', 'evap'):
        output[name] = (output[name][1] - output[name][0]) / (1000.0 * 3600.0)
    return output


@pytest.mark.parametrize('layout', ['pp', 'extracted', 'subset', 'full', 'det'])
def test_new_bounded_reads_equal_legacy_for_every_format(layout):
    n4, paths, arrays = make_sources(layout)
    closed, opened = [], []
    def opener(path, mode):
        assert mode == 'r'
        opened.append(path)
        return RecordingDataset(arrays[path], path, closed)
    expected = legacy_reference(layout, paths, arrays, n4.fv_domain_mask, 2)
    with SourcePool(opener) as datasets:
        out = met.AROMEDownloader.read_data(n4, *paths, 2, datasets=datasets)
        for name, value in expected.items():
            np.testing.assert_array_equal(getattr(out, name), value, err_msg=name)
        # A source used for multiple variable fields is opened just once.
        assert len(opened) == len(set(paths))
        for variables in arrays.values():
            for variable in variables.values():
                for key in variable.keys:
                    assert isinstance(key[-2], slice) and isinstance(key[-1], slice)
                    assert (key[-2].start, key[-2].stop) == (1, 5)
                    assert (key[-1].start, key[-1].stop) == (2, 6)
                    assert key[-2].stop - key[-2].start < 6
    assert sorted(closed) == sorted(set(paths))


def test_masked_values_and_point_order_preserved():
    grid = np.ma.array(np.arange(5 * 6 * 7).reshape(5, 6, 7).astype(float))
    grid[2, 4, 5] = np.ma.masked
    var = RecordingVariable(grid)
    ds = SimpleNamespace(variables={'field': var})
    mask = np.zeros((6, 7), dtype=bool)
    mask[1, 2] = mask[2, 3] = mask[4, 5] = True
    actual = read_masked(ds, 'field', (2,), mask)
    expected = grid[2][mask]
    assert np.ma.isMaskedArray(actual)
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_array_equal(actual.data, expected.data)


def test_source_pool_reuses_current_path_and_closes_between_forecasts():
    openings, closures = [], []
    def opener(path, mode):
        openings.append(path)
        return SimpleNamespace(close=lambda: closures.append(path))
    with SourcePool(opener) as pool:
        first = pool.get('one.nc')
        assert pool.get('one.nc') is first
        pool.get('two.nc')
        pool.close()
        assert not pool.handles
        pool.get('one.nc')
    assert openings == ['one.nc', 'two.nc', 'one.nc']
    assert closures == ['one.nc', 'two.nc', 'one.nc']


@pytest.mark.parametrize('error', [
    OSError('connection reset by peer'),
    RuntimeError('NetCDF: DAP failure'),
    RuntimeError('NetCDF: curl error'),
    RuntimeError('NetCDF: I/O failure'),
])
def test_retry_uses_roms_transient_error_messages(monkeypatch, error):
    calls = []
    class Cache:
        def close(self):
            calls.append('closed')
    def read(*args, **kwargs):
        calls.append('read')
        if calls.count('read') == 1:
            raise error
        return 'ok'
    monkeypatch.setattr(met.AROMEDownloader, 'read_data', staticmethod(read))
    monkeypatch.setattr(met.time_mod, 'sleep', lambda n: calls.append(('sleep', n)))
    assert met.AROMEDownloader._read_with_retry(None, 'file', 'a', 'b', 2,
                          datasets=Cache(), max_retries=2, retry_delay=0) == 'ok'
    assert calls == ['read', 'closed', ('sleep', 0), 'read']
    assert is_retryable_error(error)


@pytest.mark.parametrize('error', [ValueError('shape mismatch'), RuntimeError('programming error')])
def test_programming_errors_are_not_retried(monkeypatch, error):
    calls = []
    def read(*args, **kwargs):
        calls.append('read')
        raise error
    monkeypatch.setattr(met.AROMEDownloader, 'read_data', staticmethod(read))
    with pytest.raises(type(error), match=str(error)):
        met.AROMEDownloader._read_with_retry(None, 'file', 'a', 'b', 2,
                                             max_retries=5, retry_delay=0)
    assert calls == ['read']


def test_exhausted_retry_preserves_network_exception(monkeypatch):
    calls = []
    def read(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('NetCDF: DAP failure')
    monkeypatch.setattr(met.AROMEDownloader, 'read_data', staticmethod(read))
    monkeypatch.setattr(met.time_mod, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='after 3 attempts') as exc:
        met.AROMEDownloader._read_with_retry(None, 'file.nc', 'a', 'b', 2,
                                            max_retries=2, retry_delay=0)
    assert 'DAP failure' in str(exc.value.__cause__)
    assert len(calls) == 3


def test_no_full_grid_read_when_using_bounding_mask():
    arr = np.arange(5 * 6 * 7).reshape(5, 6, 7)
    var = RecordingVariable(arr)
    mask = np.zeros((6, 7), dtype=bool)
    mask[2, 3] = mask[3, 5] = True
    result = read_masked(SimpleNamespace(variables={'x': var}), 'x', (2,), mask)
    np.testing.assert_array_equal(result, arr[2][mask])
    assert var.keys == [(2, slice(2, 4), slice(3, 6))]


def test_serial_download_reuses_files_between_steps_and_closes_on_change(monkeypatch):
    sources_opened, sources_closed, written = [], [], []

    class FakeSource:
        def __init__(self, path):
            self.path = path
        def close(self):
            sources_closed.append(self.path)

    class FakeOut:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            pass

    def opener(path, mode):
        if path == 'forcing.nc':
            return FakeOut()
        assert mode == 'r'
        sources_opened.append(path)
        return FakeSource(path)

    class Bar:
        def __init__(self, **kwargs):
            self.updates = []
        def start(self):
            pass
        def update(self, i):
            self.updates.append(i)
        def finish(self):
            pass

    monkeypatch.setattr(met, 'pb', SimpleNamespace(
        Percentage=lambda: None, BouncingBar=lambda: None, ETA=lambda: None,
        ProgressBar=lambda **kwargs: Bar(**kwargs),
    ))
    monkeypatch.setattr(met.netCDF4, 'Dataset', opener)

    def read(n4, path, acc, rad, index, *, datasets):
        for p in (path, acc, rad):
            datasets.get(p)
        return SimpleNamespace(index=index)

    monkeypatch.setattr(met.AROMEDownloader, 'read_data', staticmethod(read))
    monkeypatch.setattr(met.AROMEDownloader, 'interpolate_arome_data',
                        lambda self, timestep, n4: timestep)
    monkeypatch.setattr(met.AROMEDownloader, 'write_timestep',
                        lambda self, timestep: written.append((timestep.counter, timestep.index)))

    downloader = met.AROMEDownloader.__new__(met.AROMEDownloader)
    downloader.outfile = 'forcing.nc'
    downloader.normal_paths = ['first_pp.nc', 'first_pp.nc', 'second_det.nc']
    downloader.accumulation_paths = ['acc.nc', 'acc.nc', 'second_det.nc']
    downloader.radiation_paths = ['rad.nc', 'rad.nc', 'second_det.nc']
    downloader.indices = [2, 3, 2]
    downloader.latlon = True
    downloader.N4old = object()
    downloader.N4new = object()
    downloader.dump_single()

    assert written == [(0, 2), (1, 3), (2, 2)]
    assert sources_opened == ['first_pp.nc', 'acc.nc', 'rad.nc', 'second_det.nc']
    assert sorted(sources_closed) == sorted(sources_opened)


def test_writer_exception_is_not_retried(monkeypatch):
    calls = []
    class Bar:
        def __init__(self, **kwargs): pass
        def start(self): pass
        def update(self, n): pass
        def finish(self): pass
    class Output:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(met, 'pb', SimpleNamespace(
        Percentage=lambda: None, BouncingBar=lambda: None, ETA=lambda: None,
        ProgressBar=lambda **kwargs: Bar(**kwargs),
    ))
    monkeypatch.setattr(met.netCDF4, 'Dataset', lambda path, mode: Output())
    def reader(*args, **kwargs):
        calls.append('read')
        return SimpleNamespace()
    monkeypatch.setattr(met.AROMEDownloader, '_read_with_retry', staticmethod(reader))
    monkeypatch.setattr(met.AROMEDownloader, 'interpolate_arome_data',
                        lambda self, timestep, n4: timestep)
    def broken_write(self, timestep):
        raise RuntimeError('model output bug')
    monkeypatch.setattr(met.AROMEDownloader, 'write_timestep', broken_write)
    obj = met.AROMEDownloader.__new__(met.AROMEDownloader)
    obj.outfile = 'forcing.nc'
    obj.normal_paths = ['first_pp.nc']
    obj.accumulation_paths = ['acc.nc']
    obj.radiation_paths = ['rad.nc']
    obj.indices = [2]
    obj.latlon = True
    obj.N4old = object()
    with pytest.raises(RuntimeError, match='model output bug'):
        obj.dump_single()
    assert calls == ['read']
