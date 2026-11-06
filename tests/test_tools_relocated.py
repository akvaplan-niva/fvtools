"""Regression tests for the relocation of shared FVCOM helpers to util."""

from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from fvtools.util import tools


def test_public_tools_are_importable_from_util():
    assert tools is import_module('fvtools.util.tools')
    for name in (
        'Filelist', 'num2date', 'date2num', 'get_cstr', 'coast_segments',
        'make_interpolation_matrices', 'mask_land_tris',
    ):
        assert callable(getattr(tools, name))


def test_time_conversion_unchanged():
    reference = datetime(1858, 11, 17, tzinfo=timezone.utc)
    one_day_later = datetime(1858, 11, 18, tzinfo=timezone.utc)
    np.testing.assert_array_equal(tools.date2num([reference, one_day_later]), [0.0, 1.0])
    np.testing.assert_array_equal(tools.num2date(time=np.array([0.0, 1.0])),
                                  [reference, one_day_later])


def test_filelist_still_reads_original_format(tmp_path: Path):
    filelist = tmp_path / 'fileList.txt'
    filelist.write_text('0.0\tfile_a.nc\t0\n1.0\tfile_a.nc\t1\n', encoding='utf-8')
    fl = tools.Filelist(str(filelist))
    np.testing.assert_array_equal(fl.time, [0.0, 1.0])
    assert fl.path == ['file_a.nc', 'file_a.nc']
    assert fl.index == [0, 1]


def test_old_tools_path_is_not_part_of_project():
    import fvtools.grid
    from importlib.util import find_spec

    # Assemble the old module name so the migration scanner does not treat
    # this expected assertion as a stale import.
    old = 'fvtools.grid.' + 'tools'
    assert find_spec(old) is None
