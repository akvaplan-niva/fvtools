"""Independent, synthetic regression tests for CMEMS/TPXO computational helpers."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'fvtools/setup/nesting'

def load(name):
    spec = spec_from_file_location(name, ROOT / f'{name}.py')
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cm = load('cmems')
tpx = load('tpxo')


def test_bottom_fill_preserves_wet_levels():
    a = np.array([[1., 9.], [2., 10.], [np.nan, 11.], [np.nan, np.nan]])
    np.testing.assert_array_equal(cm._fill_below_bottom(a),
                                  [[1, 9], [2, 10], [2, 11], [2, 11]])
    with pytest.raises(ValueError, match='interior missing'):
        cm._fill_below_bottom([[1.], [np.nan], [3.]])
    with pytest.raises(ValueError, match='surface'):
        cm._fill_below_bottom([[np.nan], [2.]])


def test_cmems_keeps_native_time_samples():
    times = np.array(['2026-01-01T00', '2026-01-02T00',
                      '2026-01-04T00'], dtype='datetime64[h]')
    indices = cm._native_time_indices(times,
        start_time='2026-01-01T12', stop_time='2026-01-04T00')
    np.testing.assert_array_equal(indices, [1, 2])
    assert cm._time_bracket(times, '2026-01-01T12') == (0, 1, 0.5)
    assert len(cm._output_times(times, interval_hours=1)) == 73
    with pytest.raises(ValueError, match='coverage'):
        cm._native_time_indices(times, stop_time='2026-01-05')


def test_vertical_bilinear_formula():
    # One target point halfway between source levels 10 and 20 m.
    p = np.array([[2., 8.]])
    weights = (np.array([[0]]), np.array([[1]]),
               np.array([[.5]]), np.array([[.5]]))
    np.testing.assert_allclose(cm._vertical(p, weights), [[5.]])


def test_complex_phase_wrap_is_correct():
    lon = np.array([0.,1.]); lat = np.array([0.,1.])
    phase1,phase2 = np.deg2rad([179.,-179.])
    z = np.array([[[np.exp(1j*phase1), np.exp(1j*phase2)],
                   [np.exp(1j*phase1), np.exp(1j*phase2)]]])
    mid = tpx._interpolate_complex(lon,lat,z,[.5],[.5])[0,0]
    assert mid.real < -.99
    assert abs(mid.imag) < 1e-8


def test_tpxo_units_and_components():
    assert tpx._unit_factor('cm') == .01
    assert tpx._unit_factor('cm2/s',transport=True) == 1e-4
    assert tpx._constituents(np.array([[b'M',b'2',b' '],[b'S',b'2',b' ']])) == ['M2','S2']
    assert tpx._unit_factor('meter^2/s', transport=True) == 1.0
    assert tpx._unit_factor('meters^2/s', transport=True) == 1.0
    with pytest.raises(ValueError,match='Unknown'):
        tpx._unit_factor('unknown')
