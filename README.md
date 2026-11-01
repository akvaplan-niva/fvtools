# Rename `fvtools.setup` to `fvtools.setup`

Designed for the `roms-phase6a` branch of `akvaplan-niva/fvtools` as of 2026-11-01.

## What it changes

- Rename `fvtools/setup/` to `fvtools/setup/`, preserving all tracked and untracked files under the package.
- Rewrite references in Python source, Python notebooks, examples, tests and documentation, including qualified module names (`fvtools.setup`), path components (`fvtools/setup`), and standalone strings (`'setup'`). Unrelated identifiers such as `preprocessing` and the nested `fvtools/setup/fabm/setup/` remain unchanged.
- Rename `tests/test_prepro_package_layout.py` to `tests/test_setup_package_layout.py`.
- Update the existing import-integrity test to reject `fvtools.setup` imports explicitly.
- Add `tests/test_setup_namespace.py` checking the new package and the six subpackages.
- Remove **only generated Python/Numba caches** (`.pyc`, `.nbc`, `.nbi`) inside the relocated package, so compiled kernels don't attempt to import `fvtools.setup` after the rename.
- No backwards-compatibility alias from `fvtools.setup` to `fvtools.setup` is introduced.

## Run from the fvtools root directory

Copy `rename_prepro_to_setup.py` to the repository root, then run in PowerShell:

```powershell
uv run --no-sync python .\rename_prepro_to_setup.py --check
uv run --no-sync python .\rename_prepro_to_setup.py --apply
uv run --no-sync python -m pytest tests/test_setup_namespace.py tests/test_setup_package_layout.py tests/test_fvtools_import_integrity.py -q
uv run --no-sync python -m pytest tests -q
git diff --check
git status --short
```

First execution of any Numba-accelerated function might be slower because the cache is being rebuilt. The migration is repeatable; `--check` reports changes without modifying files. The script refuses to move if `fvtools/setup` already exists alongside `fvtools/setup`, or if the destination layout test already exists.

### Source usage

Before:

```python
from fvtools.setup import BuildRivers
from fvtools.setup.nesting import roms_nesting_fg
```

After:

```python
from fvtools.setup import BuildRivers
from fvtools.setup.nesting import roms_nesting_fg
```

**Breaking change:** imports in other repositories, scripts, external Jupyter notebooks, installed cron/Slurm jobs, and any absolute paths outside the `fvtools` checkout will also need updating. This script does **not** search beyond the repository root.

The package `fvtools/setup/fabm/setup/` is retained: it has a distinct purpose and its own namespace.

## Validation

The migration was tested on a synthetic repository fixture including six packages, nested `fabm/setup`, imports, a Jupyter notebook, a stale Numba cache, a local untracked source file, syntax checks, collision refusal, and idempotence. The complete upstream `fvtools` test suite has **not** been run in the user's environment.
