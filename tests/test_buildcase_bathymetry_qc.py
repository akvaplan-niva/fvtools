"""BuildCase QC regressions: no changes to finite bathymetry or notebook API."""
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pytest

from fvtools.prepro._buildcase.depth import DepthHandler
from fvtools.prepro._buildcase.plotting import BathymetryPlotter


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close('all')


def _mesh(*, shallow=False, flat=False, missing=False):
    raw = np.array([10., 20., 30., 40.]) if shallow else np.array([110., 120., 130., 140.])
    smooth = raw - np.array([1., 2., 1., 2.])
    if flat:
        raw = np.full(4, 15.)
        smooth = raw.copy()
    return SimpleNamespace(
        x=np.array([0., 1., 1., 0.]),
        y=np.array([0., 0., 1., 1.]),
        tri=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        h_raw=raw, h=smooth,
        bathymetry_missing_mask=np.array([False, bool(missing), False, False]),
    )


def test_shallow_depth_always_produces_the_three_panels():
    obj = _mesh(shallow=True)
    before = set(plt.get_fignums())
    assert BathymetryPlotter.show_depth(obj) is None
    newer = set(plt.get_fignums()) - before
    assert len(newer) == 1
    fig = plt.figure(next(iter(newer)))
    assert len(fig.axes) == 6  # Three maps and three colorbars.
    assert [fig.axes[k].get_title() for k in (0, 1, 2)] == [
        'raw bathymetry', 'smoothed bathymetry', 'raw - smoothed bathymetry']


def test_constant_shallow_bathymetry_is_plotable():
    obj = _mesh(flat=True)
    BathymetryPlotter.show_depth(obj)
    fig = plt.gcf()
    assert len(fig.axes) == 6
    assert 'max |depth change|: 0.00 m' in fig._suptitle.get_text()


def test_missing_nodes_marked_on_raw_panel_without_changing_depth():
    obj = _mesh(missing=True)
    original, smoothed = obj.h_raw.copy(), obj.h.copy()
    BathymetryPlotter.show_depth(obj)
    fig = plt.gcf()
    assert '1/4 nodes' in fig._suptitle.get_text()
    plotted = [coll for coll in fig.axes[0].collections if isinstance(coll, PathCollection)]
    assert plotted
    assert any(len(p.get_offsets()) == 1 for p in plotted)
    np.testing.assert_array_equal(obj.h_raw, original)
    np.testing.assert_array_equal(obj.h, smoothed)


def test_invalid_mask_shape_is_reported():
    obj = _mesh()
    obj.bathymetry_missing_mask = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError, match='mask'):
        BathymetryPlotter.show_depth(obj)


def test_missing_mask_is_optional_for_older_grid_objects():
    obj = _mesh()
    del obj.bathymetry_missing_mask
    BathymetryPlotter.show_depth(obj)
    assert len(plt.gcf().axes) == 6


def test_missing_before_clipping_is_reported_and_numerics_are_unchanged(capsys):
    class Text:
        depth_file = 'topo.txt'
        x = np.zeros(5)
        min_depth = 4.
        def load_textbath(self, filename):
            return 'loaded'
        def crop(self, source):
            return source
        def interpolate_to_mesh(self, data):
            return np.array([2., np.nan, 8., 4., np.inf])
    obj = Text()
    DepthHandler.make_depth(obj)
    np.testing.assert_array_equal(obj.h_raw, [4., 4., 8., 4., 4.])
    np.testing.assert_array_equal(obj.bathymetry_missing_mask,
                                  [False, True, False, False, True])
    assert obj.bathymetry_qc['missing_nodes'] == 2
    assert obj.bathymetry_qc['missing_percent'] == pytest.approx(40.)
    assert obj.bathymetry_qc['at_min_depth_nodes'] == 4
    assert '2/5' in capsys.readouterr().out


def test_real_minimum_depth_is_not_misreported_as_missing():
    class Text:
        depth_file = 'topo.txt'
        x = np.zeros(3)
        min_depth = 4.
        def load_textbath(self, filename): return None
        def crop(self, data): return None
        def interpolate_to_mesh(self, data): return np.array([4., 2., 8.])
    obj = Text()
    DepthHandler.make_depth(obj)
    np.testing.assert_array_equal(obj.bathymetry_missing_mask, [False, False, False])
    assert obj.bathymetry_qc['missing_nodes'] == 0
    assert obj.bathymetry_qc['at_min_depth_nodes'] == 2
    np.testing.assert_array_equal(obj.h_raw, [4., 4., 8.])


def test_constant_depth_below_min_remains_unchanged():
    obj = SimpleNamespace(depth_file=2., x=np.zeros(4), min_depth=5.)
    DepthHandler.make_depth(obj)
    np.testing.assert_array_equal(obj.h_raw, [2.] * 4)
    assert not obj.bathymetry_missing_mask.any()


def test_interpolation_rejects_wrong_output_size():
    class Text:
        depth_file = 'topo.txt'
        x = np.zeros(4)
        min_depth = 4.
        def load_textbath(self, filename): return None
        def crop(self, data): return None
        def interpolate_to_mesh(self, data): return np.ones(3)
    with pytest.raises(ValueError, match='one value per grid node'):
        DepthHandler.make_depth(Text())
