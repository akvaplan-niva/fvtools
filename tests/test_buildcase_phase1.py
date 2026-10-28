"""Regression tests for BuildCase phase 1: unchanged notebook and numerical API.

No external topography, mapping service, Jupyter kernel or production files needed.
"""

from types import SimpleNamespace
import inspect

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fvtools.prepro import BuildCase as bc_module
from fvtools.prepro._buildcase.depth import DepthHandler as ExtractedDepthHandler
from fvtools.prepro._buildcase.mesh_validation import validate_boundary_triangles
from fvtools.prepro._buildcase.plotting import BathymetryPlotter


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_public_jupyter_class_and_depthhandler_reexports():
    assert bc_module.BuildCase.__module__ == 'fvtools.prepro.BuildCase'
    assert bc_module.DepthHandler is ExtractedDepthHandler
    assert issubclass(bc_module.BuildCase, ExtractedDepthHandler)
    assert bc_module.BuildCase.show_depth is BathymetryPlotter.show_depth
    assert hasattr(bc_module, 'main')
    assert hasattr(bc_module, 'RemapError')
    assert 'dmfile' in inspect.signature(bc_module.BuildCase).parameters
    assert 'sigma_file' in inspect.signature(bc_module.BuildCase).parameters
    assert 'casename' in inspect.signature(bc_module.BuildCase).parameters


def _fake_grid_methods(monkeypatch, valid=True):
    """Stub grid/IO dependencies, but run the *actual* BuildCase.__init__."""
    cls = bc_module.BuildCase
    events = []

    def setup(self):
        events.append('read_mesh')
        self.x = np.array([0., 1., 1., 0.])
        self.y = np.array([0., 0., 1., 1.])
        self.lon = self.x.copy()
        self.lat = self.y.copy()
        self.tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.obc_nodes = np.array([], dtype=np.int64)

    def proj(self):
        events.append('get_proj')
        return lambda x, y, **kw: (x, y)

    def reproject(self, reference):
        events.append(('reproject', reference))

    def valid_triangles(self, x, y, tri):
        events.append('validate')
        return np.array([0, 1] if valid else [0], dtype=np.int64)

    def subgrid(self, cells):
        events.append('subgrid')
        return SimpleNamespace(
            x=self.x.copy(), y=self.y.copy(),
            lon=self.lon.copy(), lat=self.lat.copy(),
            tri=self.tri[np.asarray(cells)].copy(),
            obc_nodes=self.obc_nodes.copy())

    monkeypatch.setattr(cls, '__str__', lambda self: 'Synthetic test grid')
    monkeypatch.setattr(cls, '_add_grid_parameters_2dm', setup)
    monkeypatch.setattr(cls, '_get_proj', proj)
    monkeypatch.setattr(cls, '_project_xy', lambda self: events.append('project_xy'))
    monkeypatch.setattr(cls, 're_project', reproject)
    monkeypatch.setattr(cls, '_return_valid_triangles', valid_triangles)
    monkeypatch.setattr(cls, 'subgrid', subgrid)
    monkeypatch.setattr(cls, 'plot_grid', lambda self, **kw: events.append('plot_grid'))
    monkeypatch.setattr(cls, 'plot_obc', lambda self, **kw: events.append('plot_obc'))
    monkeypatch.setattr(cls, 'main', lambda self: events.append('run_main'))
    return events


def test_normal_notebook_constructor_auto_runs_main(monkeypatch):
    events = _fake_grid_methods(monkeypatch, valid=True)
    exp = bc_module.BuildCase(
        dmfile='PO56_v8.2dm',
        depth_file='/nird/datalake/NS9067K/store/fvcom-setup/topo/trondelag2026_topo.npy',
        sigma_file='input/PO5-6_sigma.dat',
        casename='PO5-6',
    )
    assert exp.casename == 'PO5-6'
    assert exp.sigmafile == 'input/PO5-6_sigma.dat'
    assert events.count('run_main') == 1
    assert events.index('validate') < events.index(('reproject', 'epsg:32633'))
    assert events.index(('reproject', 'epsg:32633')) < events.index('run_main')
    assert 'subgrid' not in events


def test_corrected_mesh_stops_before_main_and_manual_resume(monkeypatch):
    events = _fake_grid_methods(monkeypatch, valid=False)
    exp = bc_module.BuildCase(dmfile='PO56_v8.2dm', depth_file='topo.npy',
                              sigma_file='input/PO5-6_sigma.dat', casename='PO5-6')
    assert 'subgrid' in events
    assert 'plot_grid' in events
    assert 'plot_obc' in events
    assert 'run_main' not in events
    assert len(exp.tri) == 1
    exp.main()
    assert events.count('run_main') == 1


def test_main_pipeline_preserves_original_step_order():
    class Recorder:
        latlon = False
        sigmafile = 'input/sigma.dat'
        h_raw = np.array([2., 3., 4.])
        def __init__(self):
            self.calls = []
        def __getattr__(self, name):
            if name in ('setup_metrics','make_depth','smooth_topo','show_depth',
                        'read_sigma','get_cfl','add_dict_info','write_grd',
                        'write_obc','write_sponge','write_cor','to_npy'):
                def record(*args, **kwargs):
                    self.calls.append((name, args, kwargs))
                return record
            raise AttributeError(name)
    obj = Recorder()
    bc_module.BuildCase.main(obj)
    assert [call[0] for call in obj.calls] == [
        'setup_metrics','make_depth','smooth_topo','show_depth',
        'read_sigma','get_cfl','add_dict_info','write_grd',
        'write_obc','write_sponge','write_cor','to_npy',
    ]
    np.testing.assert_array_equal(obj.h, obj.h_raw)
    assert obj.calls[7][2] == {'latlon': False}
    assert obj.calls[10][2] == {'latlon': False}


def test_depth_filter_keeps_original_weighting_and_no_input_mutation():
    # A leaf node with two neighbours; formula must remain (1-S)*h + S*mean(neighbours).
    x = np.array([0., 1., 2.])
    y = np.zeros(3)
    h = np.array([10., 20., 40.])
    nbsn = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    ntsn = np.array([2, 2, 2], dtype=np.int64)
    original = h.copy()
    result = ExtractedDepthHandler.laplacian_filter(x,y,h,nbsn,ntsn,.2)
    np.testing.assert_allclose(result, [.8*10+.2*30, .8*20+.2*25, .8*40+.2*15])
    np.testing.assert_array_equal(h, original)


def test_mellor_ezer_oey_keeps_depth_sum_and_target_rx0():
    h = np.array([80., 10.])
    nbsn = np.array([[1], [0]], dtype=np.int64)
    ntsn = np.array([1, 1], dtype=np.int64)
    bath, rmax, corrected = ExtractedDepthHandler.mellor_ezer_oey(h, ntsn, nbsn, rx0max=.2)
    assert corrected >= 1
    np.testing.assert_allclose(bath.sum(), h.sum())
    assert rmax > .2
    np.testing.assert_array_equal(h, [80., 10.])


def test_plotting_shows_shallow_grid_in_three_panels():
    obj = SimpleNamespace(
        x=np.array([0., 1., 1., 0.]), y=np.array([0., 0., 1., 1.]),
        tri=np.array([[0, 1, 2], [0, 2, 3]]),
        h_raw=np.array([5., 6., 7., 8.]),
        h=np.array([5., 6., 7., 8.]),
    )
    before = set(plt.get_fignums())
    assert BathymetryPlotter.show_depth(obj) is None
    newer = set(plt.get_fignums()) - before
    assert len(newer) == 1
    assert len(plt.figure(next(iter(newer))).axes) == 6


def test_mesh_validator_does_not_modify_valid_grid():
    class Grid:
        x = np.array([0.,1.,0.])
        y = np.array([0.,0.,1.])
        tri = np.array([[0,1,2]], dtype=np.int64)
        xc = np.array([1/3])
        def _return_valid_triangles(self,*args):
            return np.array([0])
    obj=Grid()
    original=obj.tri.copy()
    assert validate_boundary_triangles(obj) is False
    np.testing.assert_array_equal(obj.tri, original)
