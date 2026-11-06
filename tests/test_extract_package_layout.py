"""Checks the extraction-package migration without importing heavy scripts."""
import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTRACT = ROOT / 'fvtools' / 'extract'
UTIL = ROOT / 'fvtools' / 'util'
EXTRACTION_MODULES = (
    'fvcom_data_extract',
    'fvcom_data_extract_TS',
    'fvcom_data_extract_TS_layers',
    'fvcom_data_extract_station',
    'fvcom_data_extract_uv',
    'fvcom_data_extract_uv_layers',
    'fvcom_eager_subsetting_cell',
    'fvcom_eager_subsetting_node',
)


def test_package_and_filelist_layout():
    assert not (ROOT / 'fvtools' / 'postpro').exists()
    assert (EXTRACT / '__init__.py').is_file()
    assert not (EXTRACT / 'fvcom_make_file_list.py').exists()
    assert (UTIL / 'fvcom_make_file_list.py').is_file()
    assert importlib.util.find_spec('fvtools.extract') is not None
    assert importlib.util.find_spec('fvtools.util.fvcom_make_file_list') is not None


@pytest.mark.parametrize('module', EXTRACTION_MODULES)
def test_all_extract_modules_retained(module):
    assert (EXTRACT / (module + '.py')).is_file()
    assert importlib.util.find_spec('fvtools.extract.' + module) is not None


def test_no_stale_imports_in_fvtools():
    """Reject stale absolute imports, and old directory references in code."""
    for file in (ROOT / 'fvtools').rglob('*.py'):
        try:
            tree = ast.parse(file.read_text(encoding='utf-8-sig'))
        except (SyntaxError, UnicodeError):
            continue  # Some unrelated archived scripts use Python 2 syntax.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or '']
                if node.module == 'fvtools':
                    names.extend('fvtools.' + alias.name for alias in node.names)
                if node.module == 'fvtools.extract':
                    assert all(alias.name != 'fvcom_make_file_list' for alias in node.names), file
            else:
                continue
            assert not any(name == 'fvtools.postpro' or
                           name.startswith('fvtools.postpro.') for name in names), file


def test_readme_filelist_example_points_to_new_module():
    source = (ROOT / 'README.md').read_text(encoding='utf-8-sig')
    assert 'python -m fvtools.util.fvcom_make_file_list' in source
    assert 'python -m fvtools.postpro.fvcom_make_file_list' not in source


def test_filelist_api_preserved():
    from fvtools.util import fvcom_make_file_list as fl
    for name in ('main', 'make_fileList', 'write_fileList',
                 'read_result_directories', 'parse_command_line'):
        assert callable(getattr(fl, name))
