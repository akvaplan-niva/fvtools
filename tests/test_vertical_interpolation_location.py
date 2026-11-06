"""Regression tests for the relocation of the vertical interpolation module."""
from __future__ import annotations

from pathlib import Path
import importlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_vertical_interpolation_lives_in_interpolators():
    assert (ROOT / 'fvtools/interpolators/vertical_interpolation.py').is_file()
    assert not (ROOT / 'fvtools/setup/nesting' / 'vertical_interpolation.py').exists()
    module = importlib.import_module('fvtools.interpolators.vertical_interpolation')
    assert callable(module.calc_interp_matrices)
    assert callable(module.add_vertical_interpolation2N4)
    assert callable(module.numba_absdiff)
    assert callable(module.find_inds_and_weights)


def test_vertical_interpolation_exact_linear_depth_weights():
    module = importlib.import_module('fvtools.interpolators.vertical_interpolation')
    source = np.array([[0., 0.], [5., 10.], [10., 20.]])
    target = np.array([[2.5, 5.], [7.5, 15.]])
    i, j, a, b = module.calc_interp_matrices(source, target)
    cols = np.arange(source.shape[1])[None, :]
    actual = source[i, cols] * a + source[j, cols] * b
    np.testing.assert_allclose(actual, target, atol=1e-12, rtol=0)


def test_no_internal_code_imports_old_vertical_module():
    # Construct legacy spelling without writing it as a literal, so that
    # the static stale-reference test does not match this test itself.
    old = 'fvtools.setup.nesting.' + 'vertical_interpolation'
    old_path = 'setup/nesting/' + 'vertical_interpolation.py'
    problems = []
    for base in ('fvtools', 'tests', 'examples'):
        directory = ROOT / base
        if not directory.is_dir():
            continue
        for path in directory.rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            text = path.read_text(encoding='utf-8-sig')
            if old in text or old_path in text:
                problems.append(str(path.relative_to(ROOT)))
            if path.parts[-3:-1] == ('setup', 'nesting') and \
                    'from .vertical_interpolation import' in text:
                problems.append(str(path.relative_to(ROOT)) + ' (relative import)')
    assert not problems, 'Outdated vertical interpolation references: ' + ', '.join(problems)
