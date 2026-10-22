"""Isolated tests for fvtools.grid.netcdf_io (no FVCOM installation needed).

Run from this package folder:
    python -m unittest discover -s tests -v

These tests use a deterministic fake Dataset. They test API contracts,
not HDF5 disk throughput; run a real-file benchmark separately on Olivia.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


if importlib.util.find_spec('netCDF4') is None:
    # The testing container may not have netCDF4. Only the Dataset attribute
    # is needed here; every test patches it with FakeDataset.
    sys.modules['netCDF4'] = types.SimpleNamespace(Dataset=None)

spec = importlib.util.spec_from_file_location(
    'fvcom_netcdf_io_under_test', Path(__file__).parents[1] / 'netcdf_io.py'
)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class FakeVariable:
    def __init__(self, dimensions, values):
        self.dimensions = tuple(dimensions)
        self.values = np.ma.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.reads = []

    def __getitem__(self, selection):
        self.reads.append(selection)
        return self.values[selection]

    def chunking(self):
        return 'contiguous'


class FakeDataset:
    templates = {}
    instances = []

    def __init__(self, path, mode='r'):
        self.path = str(path)
        self.variables = {
            name: FakeVariable(v.dimensions, v.values)
            for name, v in self.templates[self.path].items()
        }
        self.closed = False
        self.instances.append(self)

    def __getitem__(self, name):
        return self.variables[name]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FakeGrid(reader.FVCOMNetCDFIO):
    def __init__(self, nodes=None, cells=None):
        self.cropped_nodes = np.array(0) if nodes is None else np.asarray(nodes)
        self.cropped_cells = np.array(0) if cells is None else np.asarray(cells)
        self.n_nodes = 12 if nodes is None else len(nodes)
        self.n_cells = 7 if cells is None else len(cells)
        self.zeta = np.zeros(self.n_nodes)
        self.interpolate_calls = 0

    @property
    def siglayz(self):
        return np.tile([-1., -5., -10.], (self.n_nodes, 1))

    @property
    def siglayz_uv(self):
        return np.tile([-1., -5., -10.], (self.n_cells, 1))

    @property
    def siglevz(self):
        return np.tile([0., -3., -7., -11.], (self.n_nodes, 1))

    def interpolate_to_z(self, data, z, depths):
        self.interpolate_calls += 1
        # Simple deterministic stub: the I/O tests verify the right depth
        # matrix and data orientation are supplied to interpolation.
        assert data.shape == depths.shape
        return (data[0] + data[1]) / 2


def make_variables():
    temperature = np.ma.array(np.arange(4 * 3 * 12, dtype='float32').reshape(4, 3, 12))
    temperature[0, 1, 1] = np.ma.masked
    u = np.arange(4 * 3 * 7, dtype='float32').reshape(4, 3, 7)
    wet_nodes = np.ones((4, 12), dtype='i1')
    wet_nodes[1, 3] = 0
    wet_cells = np.ones((4, 7), dtype='i1')
    wet_cells[0, 1] = 0
    variables = {
        'temp': FakeVariable(('time', 'siglay', 'node'), temperature),
        'salt': FakeVariable(('time', 'siglay', 'node'), temperature + 1),
        'u': FakeVariable(('time', 'siglay', 'nele'), u),
        'v': FakeVariable(('time', 'siglay', 'nele'), 2 * u),
        'zeta': FakeVariable(('time', 'node'), np.arange(48).reshape(4, 12) / 100),
        'wet_nodes': FakeVariable(('time', 'node'), wet_nodes),
        'wet_cells': FakeVariable(('time', 'nele'), wet_cells),
        'static': FakeVariable(('node',), np.arange(12)),
        'flip': FakeVariable(('node', 'siglev', 'time'),
                             np.arange(12 * 4 * 4, dtype='float32').reshape(12, 4, 4)),
    }
    return variables


class NetCDFReaderTests(unittest.TestCase):
    def setUp(self):
        FakeDataset.templates = {'a.nc': make_variables(), 'b.nc': make_variables()}
        FakeDataset.instances = []
        self.patcher = patch.object(reader, 'Dataset', FakeDataset)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_sigma_zero_and_vertical_maximum(self):
        grid = FakeGrid()
        surface = grid.load_netCDF('a.nc', 'temp', 0, sig=0)
        np.testing.assert_array_equal(surface, np.arange(12))
        maximum = grid.load_netCDF('a.nc', 'temp', 0, sig='max')
        np.testing.assert_allclose(maximum, np.arange(24, 36))

    def test_masking_nodes_and_cells(self):
        grid = FakeGrid()
        temp = grid.load_netCDF('a.nc', 'temp', 1, sig=0)
        self.assertTrue(np.isnan(temp[3]))
        u = grid.load_netCDF('a.nc', 'u', 0, sig=1)
        self.assertTrue(np.isnan(u[1]))
        unmasked = grid.load_netCDF('a.nc', 'u', 0, sig=1, mask_dry=False)
        self.assertTrue(np.isfinite(unmasked[1]))

    def test_unsorted_and_duplicate_subgrid_indices(self):
        grid = FakeGrid(nodes=[6, 1, 8, 1], cells=[4, 0, 3])
        static = grid.load_netCDF('a.nc', 'static', 0)
        np.testing.assert_array_equal(static, [6, 1, 8, 1])
        u = grid.load_netCDF('a.nc', 'u', 2, sig=2)
        np.testing.assert_array_equal(u, FakeDataset.templates['a.nc']['u'].values[2, 2, [4, 0, 3]])

    def test_z_interpolation_batch_loads_zeta_once(self):
        grid = FakeGrid(nodes=[1, 2, 5], cells=[1, 4])
        fields = grid.load_netCDF('a.nc', ['temp', 'salt'], 2, z=3)
        self.assertEqual(set(fields), {'temp', 'salt'})
        self.assertEqual(grid.interpolate_calls, 2)
        self.assertEqual(fields['temp'].shape, (3,))
        self.assertEqual(grid.zeta.shape, (3,))
        ds = FakeDataset.instances[-1]
        self.assertEqual(len(ds.variables['zeta'].reads), 1)
        self.assertEqual(len(ds.variables['wet_nodes'].reads), 1)

    def test_alternate_dimension_order(self):
        grid = FakeGrid(nodes=[7, 2, 6])
        data = grid.load_netCDF('a.nc', 'flip', 1, z=3)
        self.assertEqual(data.shape, (3,))
        self.assertEqual(grid.interpolate_calls, 1)

    def test_reused_handle_and_cleanup(self):
        grid = FakeGrid()
        with grid.netcdf_session():
            grid.load_netCDF('a.nc', 'temp', 0, sig=0)
            grid.load_netCDF('a.nc', 'temp', 1, sig=0)
            self.assertEqual(len(FakeDataset.instances), 1)
            self.assertFalse(FakeDataset.instances[0].closed)
            grid.load_netCDF('b.nc', 'temp', 1, sig=0)
            self.assertTrue(FakeDataset.instances[0].closed)
            self.assertEqual(len(FakeDataset.instances), 2)
        self.assertTrue(FakeDataset.instances[-1].closed)

    def test_session_closes_on_exception(self):
        grid = FakeGrid()
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with grid.netcdf_session():
                grid.load_netCDF('a.nc', 'temp', 0)
                raise RuntimeError('deliberate failure')
        self.assertTrue(FakeDataset.instances[0].closed)

    def test_validation(self):
        grid = FakeGrid()
        with self.assertRaises(ValueError):
            grid.load_netCDF('a.nc', 'temp', 0, sig=0, z=3)
        with self.assertRaises(KeyError):
            grid.load_netCDF('a.nc', 'missing', 0)
        with self.assertRaises(ValueError):
            grid.load_netCDF('a.nc', [], 0)


if __name__ == '__main__':
    unittest.main()
