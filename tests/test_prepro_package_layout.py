"""Regression tests for the six preprocessing package relocations."""
import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKGS = ('rivers', 'bathymetry', 'fabm', 'gridding', 'nesting', 'atm')


@pytest.mark.parametrize('name', PKGS)
def test_prepro_package_exists_and_old_top_level_is_gone(name):
    target = ROOT / 'fvtools' / 'prepro' / name
    assert (target / '__init__.py').is_file(), f'Missing {target}/__init__.py'
    assert not (ROOT / 'fvtools' / name).exists(), f'Old package still exists: {name}'
    assert importlib.util.find_spec(f'fvtools.prepro.{name}') is not None


def test_no_imports_reference_removed_package_locations():
    prefixes = tuple('fvtools.' + p for p in PKGS)
    for file in (ROOT / 'fvtools').rglob('*.py'):
        if '__pycache__' in file.parts:
            continue
        try:
            tree = ast.parse(file.read_text(encoding='utf-8-sig'))
        except (SyntaxError, UnicodeError):
            # Older, unmodified Python 2 scripts must not cause collection errors.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = ([node.module] if node.module else [])
                if node.module == 'fvtools':
                    names += ['fvtools.' + x.name for x in node.names]
            else:
                continue
            for imported in names:
                assert not any(imported == prefix or imported.startswith(prefix + '.')
                               for prefix in prefixes), (file, node.lineno, imported)
