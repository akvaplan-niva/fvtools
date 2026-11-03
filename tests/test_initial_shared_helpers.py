"""Shared restart helpers: fields, time, atomic staging and GLORYS kernels."""
import numpy as np
import pytest
from netCDF4 import Dataset
from fvtools.setup.initial._helpers.fields import (
    RESTART_DIMS, required_fields, validate_restart_fields,
)
from fvtools.setup.initial._helpers.restart_time import restart_mjd, restart_datetime64
from fvtools.setup.initial._helpers.restart_io import atomic_update_restart
from fvtools.interpolators import glorys_fields
from fvtools.setup.nesting import cmems as nesting_cmems


def _restart(path):
    with Dataset(path, 'w') as nc:
        nc.createDimension('time', 1)
        nc.createDimension('node', 2)
        nc.createDimension('nele', 1)
        nc.createDimension('siglay', 2)
        nc.createVariable('Itime', 'i4', ('time',))[:] = [60234]
        nc.createVariable('Itime2', 'i4', ('time',))[:] = [3600001]
        nc.createVariable('zeta', 'f8', RESTART_DIMS['zeta'])[:] = 3.
        for name in ('temp', 'salinity'):
            nc.createVariable(name, 'f8', RESTART_DIMS[name])[:] = 5.


def test_field_dimensions_fixed_and_validated(tmp_path):
    path = tmp_path / 'r.nc'
    _restart(path)
    assert required_fields(False) == ('zeta', 'temp', 'salinity')
    assert required_fields(True)[3:] == ('u', 'v', 'ua', 'va')
    with Dataset(path) as nc:
        validate_restart_fields(nc, required_fields(False))
        with pytest.raises(ValueError, match='missing u'):
            validate_restart_fields(nc, required_fields(True))


def test_millisecond_mjd_is_consistent_with_datetime(tmp_path):
    path = tmp_path / 'r.nc'
    _restart(path)
    with Dataset(path) as nc:
        assert restart_datetime64(nc) == np.datetime64('1858-11-17T00:00:00.000') + np.timedelta64(60234, 'D') + np.timedelta64(3600001, 'ms')
        np.testing.assert_allclose(restart_mjd(nc), [60234 + 3600001 / 86400000])


def test_atomic_commit_and_rollback(tmp_path):
    path = tmp_path / 'r.nc'
    _restart(path)
    original = path.read_bytes()
    with pytest.raises(OSError, match='synthetic'):
        with atomic_update_restart(path) as nc:
            nc['zeta'][:] = 11
            raise OSError('synthetic error')
    assert path.read_bytes() == original
    with atomic_update_restart(path) as nc:
        nc['zeta'][:] = 12
    with Dataset(path) as nc:
        np.testing.assert_array_equal(nc['zeta'][:], 12)
    assert not list(tmp_path.glob('.restart_stage_*'))


def test_glorys_field_interpolators_shared_between_restart_and_nesting():
    assert nesting_cmems._horizontal is glorys_fields._horizontal
    assert nesting_cmems._vertical is glorys_fields._vertical
    assert nesting_cmems._open_cmems is glorys_fields._open_cmems
    assert nesting_cmems.VARIABLES is glorys_fields.VARIABLES
