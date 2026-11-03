"""Offline validity-mask regression: no THREDDS connections."""
from types import SimpleNamespace
import numpy as np
import pytest
from fvtools.grid.roms.coastal_mask import probe_hindcast_wet_crop, repair_wet_corners


class Var:
    def __init__(self, values):
        self.values = np.ma.asarray(values)
    def __getitem__(self, key):
        return self.values[key]


def make_data():
    fields = {'zeta': (2, 4, 4), 'ubar_eastward': (2, 4, 4),
              'vbar_northward': (2, 4, 4), 'salinity': (2, 3, 4, 4),
              'temperature': (2, 3, 4, 4), 'u_eastward': (2, 3, 4, 4),
              'v_northward': (2, 3, 4, 4)}
    return SimpleNamespace(variables={key: Var(np.ones(shape)) for key, shape in fields.items()})


def test_effective_mask_checks_data_not_just_sea_mask():
    data = make_data()
    # This point is STATICALLY wet, but ubar is missing at every timestep.
    x = np.ma.array(np.ones((2, 4, 4)), mask=False)
    x.mask[0, 0, 0] = True
    data.variables['ubar_eastward'] = Var(x)
    window = (slice(0, 3), slice(0, 3))
    good = probe_hindcast_wet_crop(data, window, np.zeros((3, 3), bool), np.ones((3, 3)))
    assert not good[0, 0]
    assert good[1, 1]


def test_repair_renormalizes_and_replaces_invalid_source_indices():
    index = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    w = np.array([[.1, .2, .3, .4], [.1, .2, .3, .4]])
    valid = np.array([False, True, True, True])
    i, c, count = repair_wet_corners(index, w, valid, coordinate='uv')
    assert count == 2
    assert not np.any(i == 0)
    assert np.allclose(c, [[0, 2/9, 3/9, 4/9]] * 2)
    assert np.allclose(c.sum(axis=1), 1.)
    assert np.allclose(w, [[.1, .2, .3, .4]]*2)  # never mutate upstream arrays


def test_never_extrapolates_from_all_dry_cell_or_zero_weight():
    index = np.array([[0, 1, 2, 3]])
    w = np.array([[.25, .25, .25, .25]])
    with pytest.raises(ValueError, match='no valid'):
        repair_wet_corners(index, w, [False]*4, coordinate='uv')
    with pytest.raises(ValueError, match='zero-weight'):
        repair_wet_corners(index, [[1., 0., 0., 0.]], [False, True, True, True], coordinate='uv')


def test_uncorrupted_cells_are_unchanged():
    i = np.array([[0,1,2,3]])
    w = np.array([[.2,.3,.1,.4]])
    ii, ww, n = repair_wet_corners(i,w,[True]*4,coordinate='rho')
    assert n == 0
    assert ii is i and ww is w
