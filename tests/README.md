# Tests in fvtools

The suite is primarily a **numerical and behavioral regression suite**, not an
assertion that each implementation must live in one particular module.

## Keep

- Mesh and FVCOM topology, index ordering, sigma levels, NetCDF file formats.
- ROMS horizontal and vertical interpolation and correct node/cell staggering.
- OBC, river temperature and discharge, bathymetry, and section results.
- Numba kernel equivalence, independently checked geometry, nopython compilation.
- Actual KDTree nearest-neighbour distances and index behavior.
- Stable public API contracts and backward-compatible legacy names where necessary.
- Real Python 3.15 imports/JIT tests; Pandas 3 microsecond/nanosecond regression.
- Error handling, retries, safe file publication, THREDDS access efficiency.

## Remove

One-off checks of old code locations, specific mixin inheritance order,
installation patch metadata, and helper tests of the test harness itself.
Importantly, the test checking **all Numba kernels are accounted for** remains.

Avoid adding tests that simply assert exact strings in `pyproject.toml`,
`env.yml`, or an old patch's internal class hierarchy unless this is an explicit
public contract. `scripts/audit_dependencies.py` remains a manual audit tool.

## Running on Windows using uv

```powershell
# Collection alone (no test execution)
uv run --no-sync python -m pytest tests --collect-only -q

# A targeted compatibility smoke after Python/Numba changes
uv run --no-sync python -m pytest tests/test_python315_compat.py tests/test_all_numba_kernels.py tests/test_kdtree_backend.py -q

# Before committing a functional change
uv run --no-sync python -m pytest tests -q
```

On Olivia, use the equivalent `python -m pytest tests -q` in the existing
mamba/hpc-container-wrapper environment after verifying the environment.
No tests should require a live THREDDS server to pass. Reference mesh fixtures
may be skipped when optional external files are unavailable.

Do not compare test counts across runs without accounting for parametrization,
optional reference data, and version-dependent skips. Test coverage of model
results matters more than a fixed number of tests.
