"""Regression checks for renamed nesting modules.

No need to load numerical/data dependencies just to check module discovery.
"""
from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
NEST = ROOT / 'fvtools/setup/nesting'


def test_new_nesting_modules_are_present():
    for name in ('cmems', 'tpxo', 'roms', 'fvcom'):
        assert (NEST / f'{name}.py').is_file()
        assert importlib.util.find_spec(f'fvtools.setup.nesting.{name}') is not None


def test_retired_modules_are_gone():
    for name in ('cmems_nesting_fg', 'tpxo_nesting_fg',
                 'roms_nesting_fg', 'fvcom2fvcom_nesting'):
        assert not (NEST / f'{name}.py').exists()
