"""Regression check for the renamed fvtools.setup namespace."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('rivers', 'bathymetry', 'fabm', 'gridding', 'nesting', 'atm', 'initial')


def test_setup_replaces_prepro():
    assert (ROOT / 'fvtools' / 'setup' / '__init__.py').is_file()
    assert not (ROOT / 'fvtools' / 'prepro').exists()
    assert importlib.util.find_spec('fvtools.setup') is not None
    assert importlib.util.find_spec('fvtools.prepro') is None


@pytest.mark.parametrize('name', PACKAGES)
def test_moved_package_is_available(name):
    assert (ROOT / 'fvtools' / 'setup' / name / '__init__.py').is_file()
    assert importlib.util.find_spec(f'fvtools.setup.{name}') is not None
    assert not (ROOT / 'fvtools' / name).exists()
