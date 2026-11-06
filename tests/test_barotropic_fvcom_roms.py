"""Barotropic consistency across initial and nesting writers (offline)."""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import pytest

from fvtools.interpolators.barotropic import correct_fvcom_barotropic

INTERFACES = np.array([[0., -.2, -1.]])

def mean(u):
    w = -np.diff(INTERFACES, axis=1).T
    return np.sum(np.asarray(u) * w, axis=0) / np.sum(w, axis=0)


def test_sigma_weighted_consistency_and_shear():
    u, v = np.array([[1.], [9.]]), np.array([[7.], [3.]])
    ua, va = np.array([4.]), np.array([-2.])
    uc, vc = correct_fvcom_barotropic(u, v, ua, va, INTERFACES)
    np.testing.assert_allclose(mean(uc), ua)
    np.testing.assert_allclose(mean(vc), va)
    np.testing.assert_allclose(np.diff(uc, axis=0), np.diff(u, axis=0))
    np.testing.assert_allclose(np.diff(vc, axis=0), np.diff(v, axis=0))


def test_roms_initial_writer_matches_barotropic(monkeypatch):
    from fvtools.setup.initial import roms
    arrays = dict(
        siglev_center=INTERFACES.T.copy(),
        u=np.zeros((1, 2, 1)), v=np.zeros((1, 2, 1)),
        ua=np.zeros((1, 1)), va=np.zeros((1, 1)),
    )
    @contextmanager
    def fake_transaction(_):
        yield SimpleNamespace(variables=arrays)
    monkeypatch.setattr(roms, 'atomic_update_restart', fake_transaction)
    writer = object.__new__(roms.Roms2FVCOMRestart)
    writer.restartfile = 'not-used.nc'
    step = SimpleNamespace(u=np.array([[1.], [9.]]),
                           v=np.array([[7.], [3.]]),
                           ua=np.array([4.]), va=np.array([-2.]))
    writer.write_to_restart(step, variables=['u', 'v', 'ua', 'va'])
    np.testing.assert_allclose(mean(arrays['u'][0]), arrays['ua'][0])
    np.testing.assert_allclose(mean(arrays['v'][0]), arrays['va'][0])
    np.testing.assert_allclose(np.diff(arrays['u'][0], axis=0), [[8.]])


def test_roms_nesting_writer_preserves_barotropic_and_shear():
    from fvtools.setup.nesting import roms
    arrays = dict(
        siglev_center=INTERFACES.T.copy(),
        u=np.zeros((1, 2, 1)), v=np.zeros((1, 2, 1)),
        ua=np.zeros((1, 1)), va=np.zeros((1, 1)),
        zeta=np.zeros((1, 1)), temp=np.zeros((1, 2, 1)),
        salinity=np.zeros((1, 2, 1)), hyw=np.zeros((1, 3, 1)),
        weight_node=np.zeros((1, 1)), weight_cell=np.zeros((1, 1)),
    )
    writer = object.__new__(roms.Roms2FVCOMNest)
    writer.out = SimpleNamespace(variables=arrays)
    writer.N4 = SimpleNamespace(weight_node=np.array([1.]),
                                weight_cell=np.array([1.]))
    step = SimpleNamespace(counter=0, u=np.array([[1.], [9.]]),
                           v=np.array([[7.], [3.]]),
                           ua=np.array([4.]), va=np.array([-2.]),
                           zeta=np.array([0.]), temp=np.zeros((2, 1)),
                           salt=np.zeros((2, 1)))
    writer.dump_timestep_to_nest(step)
    np.testing.assert_allclose(mean(arrays['u'][0]), arrays['ua'][0])
    np.testing.assert_allclose(mean(arrays['v'][0]), arrays['va'][0])
    np.testing.assert_allclose(np.diff(arrays['u'][0], axis=0), [[8.]])


@pytest.mark.parametrize('vertical', [False, True])
def test_fvcom_nesting_with_and_without_vertical_interpolation(monkeypatch, vertical):
    from fvtools.setup.nesting import fvcom

    source = {
        'time': np.array([60000.]), 'Itime': np.array([60000]),
        'Itime2': np.array([0]), 'zeta': np.array([[0.]]),
        'ua': np.array([[4.]]), 'va': np.array([[-2.]]),
        'u': np.array([[[1.], [9.]]]), 'v': np.array([[[7.], [3.]]]),
        'temp': np.array([[[10.], [11.]]]),
        'salinity': np.array([[[32.], [33.]]]),
    }
    output = {
        'time': np.zeros(1), 'Itime': np.zeros(1), 'Itime2': np.zeros(1),
        'zeta': np.zeros((1, 1)), 'ua': np.zeros((1, 1)),
        'va': np.zeros((1, 1)), 'u': np.zeros((1, 2, 1)),
        'v': np.zeros((1, 2, 1)), 'temp': np.zeros((1, 2, 1)),
        'salinity': np.zeros((1, 2, 1)),
    }

    class DummyDataset:
        def __init__(self, path, mode='r'):
            self.variables = source if path == 'mother.nc' else output
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __getitem__(self, name):
            return self.variables[name]

    ngrd = SimpleNamespace(
        nid=np.array([0]), cid=np.array([0]),
        siglev_center=INTERFACES.copy(),
        siglayz_mother=np.array([[-.1, -.6]]),
        siglayz=np.array([[-.1, -.6]]),
        siglayz_uv_mother=np.array([[-.1, -.6]]),
        siglayz_uv=np.array([[-.1, -.6]]),
    )
    fl = SimpleNamespace(path=['mother.nc'], index=[0],
                         datetime=['2026-11-03'], time=[60000.])
    monkeypatch.setattr(fvcom, 'Filelist', lambda *args, **kwargs: fl)
    monkeypatch.setattr(fvcom, 'NEST_grid', lambda path: ngrd)
    monkeypatch.setattr(fvcom, 'check_grid', lambda *args: None)
    monkeypatch.setattr(fvcom, 'create_nestingfile', lambda *args: None)
    monkeypatch.setattr(fvcom, 'Dataset', DummyDataset)
    if vertical:
        def identity_weights(src, dest):
            i = np.array([[0], [1]])
            return i, i, np.ones((2, 1)), np.zeros((2, 1))
        monkeypatch.setattr(fvcom, 'calc_interp_matrices', identity_weights)

    fvcom.main(ngrd='unused.npy', fileList='unused.txt',
               output_file='nest.nc', vertical_interpolation=vertical)
    np.testing.assert_allclose(mean(output['u'][0]), output['ua'][0])
    np.testing.assert_allclose(mean(output['v'][0]), output['va'][0])
    np.testing.assert_allclose(np.diff(output['u'][0], axis=0), [[8.]])
