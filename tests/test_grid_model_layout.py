"""Structural regression tests after moving model-specific grid dependencies."""

import ast
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / 'fvtools' / 'grid'


def test_public_grid_modules_stay_at_root():
    for name in ('fvcom_grd', 'arome_grid', 'glorys_grid', 'roms_grid'):
        assert (GRID / f'{name}.py').is_file()
        assert importlib.import_module(f'fvtools.grid.{name}')


def test_model_helpers_are_grouped():
    for rel in (
        'fvcom/depth.py', 'fvcom/netcdf_io.py', 'fvcom/tge.py',
        'fvcom/grid_metrics.py', 'fvcom/_components/grid_loader.py',
        'fvcom/_components/export.py', 'roms/cropper_roms.py',
        'common/base.py',
    ):
        assert (GRID / rel).is_file(), rel
    for legacy in ('depth.py', 'netcdf_io.py', 'tge.py', 'grid_metrics.py',
                   'cropper_roms.py', 'base.py'):
        assert not (GRID / legacy).exists(), legacy
    assert not list((GRID / '_components').glob('*.py'))


def test_public_classes_keep_shared_dependencies():
    from fvtools.grid.arome_grid import AROMEbase
    from fvtools.grid.glorys_grid import GLORYS
    from fvtools.grid.roms_grid import ROMSbase
    from fvtools.grid.common.base import BaseGrid
    from fvtools.grid.roms.cropper_roms import ROMSCropper
    from fvtools.grid.fvcom_grd import FVCOM_grid
    from fvtools.grid.fvcom._components.grid_loader import GridLoader

    assert issubclass(AROMEbase, BaseGrid)
    assert issubclass(GLORYS, BaseGrid)
    assert issubclass(ROMSbase, ROMSCropper)
    assert issubclass(FVCOM_grid, GridLoader)


def _import_nodes(path):
    """Yield imports, including from unrelated legacy Python 2 scripts.

    A few archived fvtools scripts still use Python 2 syntax (e.g. ``print x``).
    Their syntax is outside the scope of this grid-layout regression test.
    When whole-file parsing fails, parse individual import lines so old grid
    imports in those scripts are still checked rather than silently skipped.
    """
    source = path.read_text(encoding='utf-8-sig')
    if 'fvtools.grid' not in source:
        return

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        for lineno, line in enumerate(source.splitlines(), 1):
            statement = line.strip()
            if not statement.startswith(('import ', 'from ')):
                continue
            try:
                statements = ast.parse(statement).body
            except SyntaxError:
                # A multiline legacy import cannot be parsed on its own.
                # Refuse to silently ignore it if it references fvtools.grid.
                if 'fvtools.grid' in statement:
                    raise AssertionError(
                        f'Cannot check grid import in {path}:{lineno}: {statement}'
                    )
                continue
            for node in statements:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    yield lineno, node
    else:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                yield node.lineno, node


def test_internal_code_has_no_old_imports():
    # These must be the ORIGINAL paths, not the destinations of the move.
    # The original migration script inadvertently rewrote this tuple.
    old_modules = (
        'fvtools.grid._components', 'fvtools.grid.depth',
        'fvtools.grid.netcdf_io', 'fvtools.grid.tge',
        'fvtools.grid.grid_metrics', 'fvtools.grid.cropper_roms',
        'fvtools.grid.base',
    )
    for path in (ROOT / 'fvtools').rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        for lineno, node in _import_nodes(path):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or '']
                if node.module == 'fvtools.grid':
                    names += ['fvtools.grid.' + item.name for item in node.names]
            else:
                continue
            assert not any(
                name == old or name.startswith(old + '.')
                for name in names for old in old_modules
            ), (path, lineno, names)
