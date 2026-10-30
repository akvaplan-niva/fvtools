"""BuildCase phase 2 regressions: public API, default sigma and all grid neighbors.

The tests are synthetic; they never open a bathymetry file or write FVCOM outputs.
"""
from types import SimpleNamespace

import numpy as np

from fvtools.prepro import BuildCase as bc
from fvtools.prepro._buildcase.depth import DepthHandler


def _stub_constructor(monkeypatch):
    """Run the actual BuildCase constructor, replacing only unrelated grid IO."""
    seen = []
    cls = bc.BuildCase

    def setup(self):
        self.x = np.array([0., 1., 1., 0.])
        self.y = np.array([0., 0., 1., 1.])
        self.lon = self.x.copy()
        self.lat = self.y.copy()
        self.tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp)
        self.obc_nodes = np.array([], dtype=np.intp)

    monkeypatch.setattr(cls, '__str__', lambda self: 'Synthetic grid')
    monkeypatch.setattr(cls, '_add_grid_parameters_2dm', setup)
    monkeypatch.setattr(cls, '_get_proj', lambda self: lambda x, y, **kw: (x, y))
    monkeypatch.setattr(cls, '_project_xy', lambda self: None)
    monkeypatch.setattr(cls, 're_project', lambda self, reference: seen.append(('reproject', reference)))
    monkeypatch.setattr(cls, 'main', lambda self: seen.append(('main', self.sigmafile)))
    monkeypatch.setattr(bc, 'validate_boundary_triangles', lambda self: False)
    return seen


def test_direct_jupyter_constructor_default_sigma_and_auto_run(monkeypatch):
    seen = _stub_constructor(monkeypatch)
    obj = bc.BuildCase(dmfile='PO56_v8.2dm', depth_file='topo.npy', casename='PO5-6')
    assert obj.sigmafile == './input/PO5-6_sigma.dat'
    assert seen == [('reproject', 'epsg:32633'), ('main', obj.sigmafile)]


def test_direct_constructor_explicit_sigma_unchanged(monkeypatch):
    seen = _stub_constructor(monkeypatch)
    obj = bc.BuildCase(dmfile='PO56_v8.2dm', depth_file='topo.npy', casename='PO5-6',
                       sigma_file='input/PO5-6_sigma.dat')
    assert obj.sigmafile == 'input/PO5-6_sigma.dat'
    assert seen == [('reproject', 'epsg:32633'), ('main', obj.sigmafile)]


def test_direct_constructor_accepts_flat_bathymetry(monkeypatch):
    """The DepthHandler already documents and supports numeric depth input."""
    _stub_constructor(monkeypatch)
    obj = bc.BuildCase(dmfile='mesh.2dm', depth_file=25., casename='flat')
    assert obj.depth_file == 25.


def test_module_main_forwards_every_explicit_setting_without_eval(monkeypatch):
    seen = []
    def construct(**kwargs):
        seen.append(kwargs)
        return object()
    monkeypatch.setattr(bc, 'BuildCase', construct)
    args = dict(dmfile='mesh.2dm', depth_file='topo.npy', casename='S',
                dm_projection='epsg:32632', depth_projection='epsg:32633',
                target_projection='epsg:32631', sigma_file='input/S_sigma.dat',
                rx0max=.12, SmoothFactor=.15, min_depth=7., sponge_radius=8000,
                sponge_factor=.001, latlon=True)
    result = bc.main(**args)
    assert result is not None
    assert seen == [args]


def test_module_main_computes_default_sigma_file(monkeypatch):
    calls=[]
    monkeypatch.setattr(bc, 'BuildCase', lambda **kwargs: calls.append(kwargs))
    bc.main(dmfile='mesh.2dm', depth_file='topo.npy', casename='S')
    assert calls[0]['sigma_file'] == './input/S_sigma.dat'


def _fan_mesh(n):
    """Planar fan with n external nodes, center has n neighbors."""
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.r_[0., np.cos(theta)]
    y = np.r_[0., np.sin(theta)]
    tri = np.array([[0, 1+i, 1+((i+1) % n)] for i in range(n)], dtype=np.intp)
    return x, y, tri


def test_dynamic_neighbor_capacity_exceeds_old_twelve_limit():
    x, y, tri = _fan_mesh(16)
    obj = SimpleNamespace(x=x, y=y, tri=tri)
    DepthHandler.setup_metrics(obj)
    assert obj.nbsn.shape == (len(x), 16)
    assert obj.ntsn[0] == 16
    assert set(obj.nbsn[0, :obj.ntsn[0]].tolist()) == set(range(1, 17))
    assert np.all(obj.ntsn[1:] == 3)
    # Stability algorithms still consume only the filled columns.
    depths = np.arange(1., len(x) + 1.)
    smoothed = DepthHandler.laplacian_filter(x, y, depths, obj.nbsn, obj.ntsn, .2)
    np.testing.assert_allclose(smoothed[0], .8*depths[0] + .2*np.mean(depths[1:]))


def test_dynamic_neighbors_have_no_arbitrary_minimum():
    x, y, tri = _fan_mesh(5)
    obj = SimpleNamespace(x=x, y=y, tri=tri)
    DepthHandler.setup_metrics(obj)
    assert obj.nbsn.shape == (6, 5)
