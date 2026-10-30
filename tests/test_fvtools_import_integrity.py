"""Ensure imports remain valid after the fvtools/prepro package migration.

Uses lexical scanning, so archived scripts with Python 2 syntax are included.
No production code is imported or executed during the scan.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location('fix_fvtools_imports', ROOT / 'fix_fvtools_imports.py')
MOD = module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_rewriter_ignores_comments_and_docstrings():
    old = '''"""\nimport gridding.domain as domain\n"""\n# import grid.fvcom_grd as grid\nimport grid.fvcom_grd as fvg\nfrom fvcom_grd import FVCOM_grid\nimport fvcom_grd\n'''
    new, edits = MOD.rewrite_source(old, Path('prepro/fabm/fabm.py'))
    assert '"""\nimport gridding.domain as domain\n"""' in new
    assert '# import grid.fvcom_grd as grid' in new
    assert 'import fvtools.grid.fvcom_grd as fvg' in new
    assert 'from fvtools.grid.fvcom_grd import FVCOM_grid' in new
    assert 'import fvtools.grid.fvcom_grd as fvcom_grd' in new
    assert len(edits) == 3


def test_rewriter_handles_python2_source():
    old = 'print file\nimport gridding.domain as domain\nfrom grid.fvcom_grd import FVCOM_grid\n'
    new, _ = MOD.rewrite_source(old, Path('meshingtemplate/make_mesh.py'))
    assert new == ('print file\nimport fvtools.prepro.gridding.domain as domain\n'
                   'from fvtools.grid.fvcom_grd import FVCOM_grid\n')


def test_rewriter_retains_valid_imports():
    old = ('from ..util.time import epoch_pandas\n'
           'from fvtools.prepro.nesting import matchtopo\n'
           'import fvtools.grid.fvcom.tge as tge\n')
    new, edits = MOD.rewrite_source(old, Path('injection/makedata.py'))
    assert old == new
    assert edits == []


def test_rewriter_repairs_legacy_matplotlib_find():
    old = ('import numpy as np\nimport matplotlib.mlab as mlab\n'
           'from IPython.core.debugger import Tracer\n'
           'x = mlab.find(grd.x > 1)\n')
    new, _ = MOD.rewrite_source(old, Path('postpro/fvcom_data_extract.py'))
    assert 'matplotlib.mlab' not in new
    assert 'from IPython.core.debugger' not in new
    assert 'x = np.flatnonzero(grd.x > 1)' in new


def test_all_fvtools_imports_resolve_to_local_modules():
    modules = MOD._module_names(ROOT)
    issues = []
    for file in sorted((ROOT / 'fvtools').rglob('*.py')):
        if '__pycache__' in file.parts:
            continue
        source = file.read_text(encoding='utf-8-sig')
        issues.extend(MOD.audit_file(source, file, ROOT, modules))
    assert not issues, '\n'.join(issues)


def test_old_nesting_relative_to_top_level_is_rewritten():
    old = ('from nesting.vertical_interpolation import calc_interp_matrices\n'
           'import grid.fvcom_grd as fvg\n'
           'from fvtools.nesting import matchtopo\n')
    new, events = MOD.rewrite_source(old, Path('prepro/nesting/fvcom2fvcom.py'))
    assert new == ('from fvtools.prepro.nesting.vertical_interpolation import calc_interp_matrices\n'
                   'import fvtools.grid.fvcom_grd as fvg\n'
                   'from fvtools.prepro.nesting import matchtopo\n')
    assert len(events) == 3


def test_refuse_unaliased_import_when_binding_would_change():
    import pytest
    with pytest.raises(ValueError, match='unaliased'):
        MOD.rewrite_source('import gridding.coast\n', Path('meshingtemplate/example.py'))
