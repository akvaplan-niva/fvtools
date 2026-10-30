# BuildRivers without grid_metrics (roms-phase6a)

Unzip in the root of `fvtools`. The previously removed `fvtools/grid/fvcom/grid_metrics.py` should remain deleted.

```powershell
git apply --check .\fix_buildrivers_no_grid_metrics.patch
git apply .\fix_buildrivers_no_grid_metrics.patch
uv run --no-sync python -m pytest tests/test_buildrivers_no_grid_metrics.py tests/test_buildrivers_refactor.py tests/test_buildrivers_forcing_phase2b.py -q
uv run --no-sync python -m pytest tests -q
git status --short
```

`FVCOM_rivers` now uses the existing `M.boundary_nodes` and `M.nodestrings`, matching the original `get_nbe` node classification. Cell-based river logic is untouched. Only these two files are altered by the patch.
