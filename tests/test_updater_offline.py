"""The updater accepts changed log text but never silently skips missing days."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('strict_retry_updater',ROOT/'update_roms_strict_retry.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SOURCE='''import numpy as np
import netCDF4
from fvtools.util.thredds_io import retry_read

def make_fileList(start, stop, ROMS):
    for date in [start]:
        try:
            file = ROMS.test_day(date)
        except NoAvailableData as err:
            print('Custom user message: unavailable', err)
            continue
        with netCDF4.Dataset(file) as d:
            ocean_time = d.variables['ocean_time']
            t_roms = netCDF4.num2date(ocean_time[:], units=ocean_time.units,
                                       only_use_cftime_datetimes=False)
            t_fvcom = np.atleast_1d(netCDF4.date2num(
                t_roms, units='days since 1858-11-17 00:00:00'))
    return t_fvcom

class Roms2FVCOMNest(RomsDownloader, LinearInterpolation):
    def _prepare_timestep(self, counter, path_here, index_here):
        timestep = self.read_timestep(index_here, path_here)
        timestep = self.crop_and_transpose(timestep)
        timestep = self.horizontal_interpolation(timestep)
        timestep = self.vertical_interpolation(timestep)
        timestep = self.adjust_uv(timestep)
        timestep.counter = counter
        return timestep
'''


def test_changed_missing_day_wording_works_and_is_idempotent():
    changed=mod.modify_roms(SOURCE)
    assert 'Custom user message' not in changed
    assert 'no day was skipped' in changed
    assert 'def _check_roms_nesting_source' in changed
    assert 't_fvcom = np.atleast_1d(' in changed
    compile(changed,'roms.py','exec')
    assert mod.modify_roms(changed)==changed


def test_existing_raise_preserved():
    source=SOURCE.replace("print('Custom user message: unavailable', err)\n            continue", "raise ValueError('already handled') from err")
    fixed=mod.modify_roms(source)
    assert "raise ValueError('already handled') from err" in fixed


def test_unrecognized_control_flow_fails_closed():
    source=SOURCE.replace("print('Custom user message: unavailable', err)\n            continue", "return None")
    with pytest.raises(ValueError, match='Custom missing-day control flow'):
        mod.modify_roms(source)


def test_metadata_indentation_variants_and_original_time_parsing_retained():
    changes = SOURCE.replace(
        "t_roms = netCDF4.num2date(ocean_time[:], units=ocean_time.units,",
        "t_roms = netCDF4.num2date( ocean_time[ : ], units=ocean_time.units,",
    ).replace("only_use_cftime_datetimes=False)", "only_use_cftime_datetimes = False)")
    got = mod.modify_roms(changes)
    compile(got, '<roms>', 'exec')
    assert got.count('ocean_time') >= 2
    assert 'def read_file_times():' not in got
    assert 'ocean_time[ : ]' in got
    assert 'only_use_cftime_datetimes = False' in got


def test_no_grid_file_change_needed():
    # We deliberately retry NorKyst.test_day in make_fileList and avoid modifying
    # the potentially customized roms_grid.py.
    changed = mod.modify_roms(SOURCE)
    assert 'max_retries=5' in changed
    assert 'lambda: ROMS.test_day(date)' in changed
