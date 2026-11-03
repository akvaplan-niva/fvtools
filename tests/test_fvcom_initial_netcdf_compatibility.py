"""Offline compatibility tests for FVCOM's NetCDF attribute reader.

FVCOM's akvaplan_dev READ_ATT_TYPE handles NC_INT, but not NC_INT64.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

from fvtools.setup.initial._helpers.empty_initial import make_initial_file
from fvtools.setup.nesting import tpxo


def _minimal_grid():
    """Three nodes, one element, three sigma interfaces; no external files."""
    return SimpleNamespace(
        casename='synthetic',
        x=np.array([0., 100., 0.]),
        y=np.array([0., 0., 100.]),
        xc=np.array([100. / 3]),
        yc=np.array([100. / 3]),
        lon=np.array([5., 5.001, 5.]),
        lat=np.array([60., 60., 60.001]),
        lonc=np.array([5.00033]),
        latc=np.array([60.00033]),
        h=np.array([20., 21., 22.]),
        hc=np.array([21.]),
        tri=np.array([[0, 1, 2]], dtype=np.int32),
        obc_nodes=np.array([0, 1], dtype=np.int32),
        siglev=np.tile([0., -.5, -1.], (3, 1)),
        siglay=np.tile([-.25, -.75], (3, 1)),
        siglev_center=np.array([[0., -.5, -1.]]),
        siglay_center=np.array([[-.25, -.75]]),
    )


def test_initial_file_uses_classic_netcdf_types(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    filename = make_initial_file(_minimal_grid(), '2024-08-01-00', obc_type=3)
    with Dataset(filename) as nc:
        assert nc.data_model == 'NETCDF4_CLASSIC'
        assert nc.variables['temp'].dtype == np.dtype('float32')
        assert nc.variables['iint'].dtype == np.dtype('int32')
        np.testing.assert_array_equal(nc.variables['nv'][:], [[1], [2], [3]])
        np.testing.assert_array_equal(nc.variables['obc_nodes'][:], [1, 2])
        np.testing.assert_array_equal(nc.variables['obc_type'][:], [3, 3])


@pytest.mark.parametrize('model', ['NETCDF4', 'NETCDF4_CLASSIC'])
def test_tpxo_marker_is_nc_int32_in_both_file_models(tmp_path, model):
    filename = tmp_path / 'restart.nc'
    with Dataset(filename, 'w', format=model) as nc:
        nc.createDimension('time', 1)
        nc.createDimension('node', 2)
        nc.createVariable('zeta', 'f4', ('time', 'node'))[:] = 2.
        tpxo._write_tpxo_predictions(
            nc, {'zeta': np.array([[.25, .25]])}, ('M2',)
        )
    with Dataset(filename) as nc:
        assert nc.getncattr('tpxo_applied') == 1
        assert np.asarray(nc.getncattr('tpxo_applied')).dtype == np.dtype('int32')
        np.testing.assert_array_equal(nc.variables['zeta'][:], [[2.25, 2.25]])
        with pytest.raises(ValueError, match='already been added'):
            tpxo._assert_tpxo_not_applied(nc, filename)
