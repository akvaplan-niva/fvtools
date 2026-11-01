"""Static FVTools import checks, independent of one-off migration scripts.

Read Python source rather than importing it. Tokenization also supports archived
Python 2 files that cannot be parsed by the Python 3 AST parser.
"""
from __future__ import annotations

from importlib.util import resolve_name
from io import StringIO
from pathlib import Path
import tokenize


ROOT = Path(__file__).resolve().parents[1]
MOVED_PACKAGES = ('rivers', 'bathymetry', 'fabm', 'gridding', 'nesting', 'atm')
OBSOLETE_MODULES = (
    *(f'fvtools.{package}' for package in MOVED_PACKAGES),
    *(package for package in MOVED_PACKAGES),
    'grid', 'fvcom_grd', 'fvcom_plot', 'nesting', 'pre_pro',
    'fvtools.pre_pro',
    'fvtools.prepro',
    'fvtools.grid.grid_metrics',
    'fvtools.grid.fvcom.grid_metrics',
)


def _is_obsolete(module: str) -> bool:
    return any(module == old or module.startswith(old + '.')
               for old in OBSOLETE_MODULES)


def _iter_imports(source: str):
    """Yield (module, line number) for real import statements, not text.

    This uses the same token-based approach as the original migration audit:
    strings, comments, and Python 2 print statements are all safe to scan.
    """
    tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    ignored = {tokenize.COMMENT, tokenize.ENCODING, tokenize.INDENT,
               tokenize.DEDENT, tokenize.NL}
    at_stmt_start = True
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == tokenize.NEWLINE or (
                token.type == tokenize.OP and token.string == ';'):
            at_stmt_start = True
            i += 1
            continue
        if token.type in ignored:
            i += 1
            continue
        if token.type == tokenize.OP and token.string == ':':
            # Handle one-line suites: if condition: import something
            at_stmt_start = True
            i += 1
            continue
        if not (token.type == tokenize.NAME
                and token.string in ('import', 'from') and at_stmt_start):
            at_stmt_start = False
            i += 1
            continue
        is_from = token.string == 'from'
        i += 1
        while i < len(tokens):
            if tokens[i].type in (tokenize.NL, tokenize.COMMENT):
                i += 1
                continue
            if tokens[i].type == tokenize.OP and tokens[i].string in ('(', ')', '\\'):
                i += 1
                continue
            if (tokens[i].type not in (tokenize.NAME, tokenize.OP)
                    or tokens[i].string in (',', ';')):
                break
            start = i
            parts = []
            while i < len(tokens) and (
                    tokens[i].type == tokenize.NAME or
                    (tokens[i].type == tokenize.OP and tokens[i].string == '.')):
                if tokens[i].string in ('import', 'as'):
                    break
                parts.append(tokens[i].string)
                i += 1
            module = ''.join(parts)
            if not module:
                break
            yield module, tokens[start].start[0]
            if is_from:
                break
            if i < len(tokens) and tokens[i].string == 'as':
                i += 2  # skip alias and continue after it
            if i < len(tokens) and tokens[i].string == ',':
                i += 1
                continue
            break
        at_stmt_start = False


def _local_modules(root: Path) -> set[str]:
    """Collect importable FVTools modules, including namespace packages."""
    modules = {'fvtools'}
    for path in (root / 'fvtools').rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        parts = path.relative_to(root).with_suffix('').parts
        if parts[-1] == '__init__':
            parts = parts[:-1]
        modules.update('.'.join(parts[:i]) for i in range(1, len(parts) + 1))
    return modules


def _audit_source(source: str, file: Path, root: Path,
                  modules: set[str]) -> list[str]:
    rel = file.relative_to(root)
    parent_parts = list(rel.with_suffix('').parts)
    if parent_parts[-1] != '__init__':
        parent_parts.pop()
    package = '.'.join(parent_parts)
    issues = []
    for imported, line in _iter_imports(source):
        name = imported
        if name.startswith('.'):
            try:
                name = resolve_name(name, package)
            except ImportError:
                issues.append(f'{rel}:{line}: relative import escapes package: {imported}')
                continue
        if _is_obsolete(name):
            issues.append(f'{rel}:{line}: obsolete import path: {imported}')
        elif name == 'fvtools' or name.startswith('fvtools.'):
            if name not in modules:
                issues.append(f'{rel}:{line}: missing internal module: {imported}')
    return issues


def test_import_scanner_ignores_comments_and_docstrings():
    source = ('"""import gridding.domain"""\n'
              '# from fvtools.nesting import matchtopo\n'
              'import fvtools.grid.fvcom_grd as grid\n'
              'from fvtools.setup.nesting import vertical_interpolation\n')
    assert [name for name, _ in _iter_imports(source)] == [
        'fvtools.grid.fvcom_grd', 'fvtools.setup.nesting'
    ]


def test_import_scanner_handles_legacy_python2_syntax():
    source = 'print file\nimport gridding.domain as domain\n'
    assert [name for name, _ in _iter_imports(source)] == ['gridding.domain']


def test_audit_reports_obsolete_and_missing_modules():
    source = ('from fvtools.nesting import matchtopo\n'
              'from fvtools.grid.does_not_exist import Example\n'
              'from fvtools.setup import BuildRivers\n')
    issues = _audit_source(
        source, ROOT / 'fvtools/setup/example.py', ROOT,
        {'fvtools', 'fvtools.grid', 'fvtools.setup'},
    )
    assert len(issues) == 2, issues
    assert any('obsolete import path' in item for item in issues)
    assert any('missing internal module' in item for item in issues)


def test_all_fvtools_imports_resolve_to_local_modules():
    assert (ROOT / 'fvtools').is_dir(), 'FVTools package directory is missing'
    modules = _local_modules(ROOT)
    issues = []
    for file in sorted((ROOT / 'fvtools').rglob('*.py')):
        if '__pycache__' in file.parts:
            continue
        source = file.read_text(encoding='utf-8-sig')
        issues.extend(_audit_source(source, file, ROOT, modules))
    assert not issues, '\n'.join(issues)
