# fvtools regression tests

These tests are intended to be run from the **`diverse_smaating`** branch.
They use the public FVCOM APIs (`FVCOM_grid`, `read_sms_mesh`, `load_netCDF`),
so reorganizing implementation modules should not require rewriting the tests.

## Quick tests

Install the project's normal runtime dependencies from `env.yml`, and ensure
`pytest` is available in the environment. From the repository root, run:

```bash
python -m pytest tests -q
```

The fast tests build a four-node grid and a synthetic NetCDF file, so they
run without external geographic datasets, network access, or an HPC system.

## Optional tests on a real mesh

The two original reference files are not committed to Git because the `.npy`
file alone is about 31 MB. Put the files in `tests/reference/`:

- `v2_with_nodestrings_with_nest.2dm`
- `M.npy`

Or provide their existing full paths:

```bash
export FVTOOLS_TEST_2DM=/path/to/v2_with_nodestrings_with_nest.2dm
export FVTOOLS_TEST_M_NPY=/path/to/M.npy
python -m pytest tests/test_reference_mesh.py -q
```

Tests that need those files are skipped if either is absent. `M.npy` contains
a Python object array, which requires pickle-based loading; **only load a
trusted file**.

The real fixtures currently contain 48,803 nodes, 93,524 triangular elements,
and three nodestrings (25, 57, and 17 nodes). They test the exact ordering of
triangles, coordinates, boundaries, and vertical sigma-coordinate invariants.

## Scope

- `.2dm` SMS grid parsing and ordered open-boundary nodestrings
- Recreating FVCOM_grid from a saved `M.npy`
- Distinguishing node-based `temp`/`pv` from element-based `u`/`v`
- Wet/dry masks and repeatable NetCDF loading across timesteps
- Avoiding accidental simultaneous `sig` and `z` inputs

The suite deliberately does not modify `env.yml` or `environment.yaml`.
