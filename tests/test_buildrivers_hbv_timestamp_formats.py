"""Regression tests for the native HBV YYYYMMDD/HHMM timestamps."""
from datetime import datetime, timezone
import numpy as np
import pytest

from fvtools.prepro import BuildRivers as br


def write_var(root, area, stamps, values=None):
    if values is None:
        values = np.arange(1, len(stamps) + 1, dtype=float)
    path = root / f'hbv_{area:08d}.var'
    path.write_text(''.join(f'{stamp} {value}\n' for stamp, value in zip(stamps, values)), encoding='utf-8')
    return path


@pytest.mark.parametrize(('stamp', 'expected'), [
    ('19900101/1200', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),
    ('19900101/1230', datetime(1990, 1, 1, 12, 30, tzinfo=timezone.utc)),
    ('19900101/0015', datetime(1990, 1, 1, 0, 15, tzinfo=timezone.utc)),
    ('19900101_1200', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),
    ('199001011200', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),
    ('19900101/12', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),
    ('19900101_12', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),
    ('1990010112', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),
])
def test_hbv_timestamp_forms(tmp_path, stamp, expected):
    write_var(tmp_path, 91, [stamp])
    runoff = br.HBVRunoff(tmp_path)
    assert runoff.dates.tolist() == [expected]
    np.testing.assert_array_equal(runoff.transport.loc[91], [1.0])


def test_hbv_mixed_format_files_share_identical_time_axis(tmp_path):
    write_var(tmp_path, 91, ['19900101/0000', '19900101/1200', '19900102/0000'])
    write_var(tmp_path, 92, ['1990010100', '1990010112', '1990010200'], [10, 20, 30])
    result = br.HBVRunoff(tmp_path)
    assert result.dates.tolist() == [
        datetime(1990, 1, 1, 0, tzinfo=timezone.utc),
        datetime(1990, 1, 1, 12, tzinfo=timezone.utc),
        datetime(1990, 1, 2, 0, tzinfo=timezone.utc),
    ]
    np.testing.assert_array_equal(result.transport.loc[92], [10., 20., 30.])


@pytest.mark.parametrize('stamp', [
    '19900101/2460', '19900101/2400', '19900230/1200',
    '19900101/12:00', '19900101/abc0',
])
def test_invalid_hbv_timestamp_is_rejected_with_filename(tmp_path, stamp):
    path = write_var(tmp_path, 91, [stamp])
    with pytest.raises(ValueError, match=path.name):
        br.HBVRunoff(tmp_path)


def test_real_world_hbv_times_and_exclusive_time_crop(tmp_path):
    write_var(tmp_path, 91, ['20231231/1200', '20240101/1200', '20240102/1200'],
              [3.5, 4.5, 5.5])
    result = br.HBVRunoff(
        tmp_path,
        start=datetime(2023, 12, 31, 12, tzinfo=timezone.utc),
        stop=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
    )
    assert result.dates.tolist() == [datetime(2024, 1, 1, 12, tzinfo=timezone.utc)]
    np.testing.assert_array_equal(result.transport.loc[91], [4.5])
