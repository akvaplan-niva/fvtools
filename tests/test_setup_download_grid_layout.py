"""Regression checks for nesting mesh utilities and external-download modules."""
from pathlib import Path
from importlib.util import find_spec

ROOT = Path(__file__).resolve().parents[1] / "fvtools" / "setup"


def test_download_locations():
    for module in ("fvtools.setup.download.cmems", "fvtools.setup.download.era5",
                   "fvtools.setup.atm.era5_processing"):
        assert find_spec(module) is not None
    assert not (ROOT / "nesting" / "cmems_download.py").exists()
    assert not (ROOT / "atm" / "download_era5.py").exists()


def test_nesting_grid_locations():
    for name in ("get_ngrd", "get_fvcom_ngrd", "get_roms_ngrd", "matchtopo", "obcgridding"):
        assert find_spec("fvtools.setup.nesting.grid." + name) is not None
        assert not (ROOT / "nesting" / (name + ".py")).exists()
