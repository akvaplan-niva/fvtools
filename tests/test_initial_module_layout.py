"""Verify the new restart module API and that old import paths are gone."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'fvtools' / 'setup' / 'initial'
NEW_MODULES = ('roms', 'cmems', 'fvcom', 'tpxo')
OLD_MODULES = ('interpol_roms_restart', 'interpol_glorys_restart',
               'interpol_restart', 'interpol_tpxo_restart')


@pytest.mark.parametrize('name', NEW_MODULES)
def test_restart_module_is_in_initial(name):
    assert (PACKAGE / f'{name}.py').is_file()
    assert importlib.util.find_spec(f'fvtools.setup.initial.{name}') is not None


@pytest.mark.parametrize('name', OLD_MODULES)
def test_old_module_is_removed(name):
    assert not (ROOT / 'fvtools' / 'setup' / f'{name}.py').exists()
    assert importlib.util.find_spec(f'fvtools.setup.{name}') is None


def test_original_initial_helpers_were_moved_beneath_initial():
    helpers = PACKAGE / '_helpers'
    assert (helpers / '__init__.py').is_file()
    assert (helpers / 'empty_initial.py').is_file()
    assert (helpers / 'plot_restart.py').is_file()
    assert not (PACKAGE / 'empty_initial.py').exists()
    assert not (PACKAGE / 'plot_restart.py').exists()


def test_no_old_restart_import_paths_within_package():
    forbidden = {
        'fvtools.setup.interpol_restart',
        'fvtools.setup.interpol_roms_restart',
        'fvtools.setup.interpol_glorys_restart',
        'fvtools.setup.interpol_tpxo_restart',
        'fvtools.setup.initial.empty_initial',
        'fvtools.setup.initial.plot_restart',
    }
    for path in (ROOT / 'fvtools').rglob('*.py'):
        try:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        except (SyntaxError, UnicodeError):
            # Some archived, unrelated modules still contain Python 2 syntax.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module] if node.module else []
            else:
                continue
            assert not forbidden.intersection(names), (path, node.lineno, names)
