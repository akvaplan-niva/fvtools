#!/usr/bin/env python3
"""Fix YYYYMMDD/HHMM timestamps in BuildRivers Phase 2A.

Place this script in the repository root and run:
    uv run python fix_buildrivers_hbv_timestamps.py

Only patches the HBV date parser; leaves temperature and forcing untouched.
"""
from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'fvtools/prepro/_buildrivers/sources.py'
TEST = ROOT / 'tests/test_buildrivers_hbv_timestamp_formats.py'
OLD = "                # Both YYYYMMDDHH and YYYYMMDD_HH (single separator) occur.\n                if re.fullmatch(r'\\d{10}', stamp):\n                    packed = stamp\n                elif re.fullmatch(r'\\d{8}\\D\\d{2}', stamp):\n                    packed = stamp[:8] + stamp[-2:]\n                else:\n                    raise ValueError(f'Invalid HBV timestamp {stamp!r} in {path.name}')\n                dates.append(datetime.strptime(packed, '%Y%m%d%H').replace(\n                    tzinfo=timezone.utc))\n"
NEW = "                # Accept both original hourly HBV timestamps and HHMM formats,\n                # including YYYYMMDD/HHMM used by the Norwegian HBV .var files.\n                match = re.fullmatch(r'(\\d{8})[/_]?(\\d{2})(\\d{2})?', stamp)\n                if match is None:\n                    raise ValueError(f'Invalid HBV timestamp {stamp!r} in {path.name}')\n                day, hour, minute = match.groups()\n                packed = day + hour + (minute or '')\n                fmt = '%Y%m%d%H%M' if minute is not None else '%Y%m%d%H'\n                try:\n                    parsed = datetime.strptime(packed, fmt)\n                except ValueError as error:\n                    raise ValueError(\n                        f'Invalid HBV timestamp {stamp!r} in {path.name}'\n                    ) from error\n                dates.append(parsed.replace(tzinfo=timezone.utc))\n"
TEST_CONTENT = '"""Regression tests for the native HBV YYYYMMDD/HHMM timestamps."""\nfrom datetime import datetime, timezone\nimport numpy as np\nimport pytest\n\nfrom fvtools.prepro import BuildRivers as br\n\n\ndef write_var(root, area, stamps, values=None):\n    if values is None:\n        values = np.arange(1, len(stamps) + 1, dtype=float)\n    path = root / f\'hbv_{area:08d}.var\'\n    path.write_text(\'\'.join(f\'{stamp} {value}\\n\' for stamp, value in zip(stamps, values)), encoding=\'utf-8\')\n    return path\n\n\n@pytest.mark.parametrize((\'stamp\', \'expected\'), [\n    (\'19900101/1200\', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),\n    (\'19900101/1230\', datetime(1990, 1, 1, 12, 30, tzinfo=timezone.utc)),\n    (\'19900101/0015\', datetime(1990, 1, 1, 0, 15, tzinfo=timezone.utc)),\n    (\'19900101_1200\', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),\n    (\'199001011200\', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),\n    (\'19900101/12\', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),\n    (\'19900101_12\', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),\n    (\'1990010112\', datetime(1990, 1, 1, 12, tzinfo=timezone.utc)),\n])\ndef test_hbv_timestamp_forms(tmp_path, stamp, expected):\n    write_var(tmp_path, 91, [stamp])\n    runoff = br.HBVRunoff(tmp_path)\n    assert runoff.dates.tolist() == [expected]\n    np.testing.assert_array_equal(runoff.transport.loc[91], [1.0])\n\n\ndef test_hbv_mixed_format_files_share_identical_time_axis(tmp_path):\n    write_var(tmp_path, 91, [\'19900101/0000\', \'19900101/1200\', \'19900102/0000\'])\n    write_var(tmp_path, 92, [\'1990010100\', \'1990010112\', \'1990010200\'], [10, 20, 30])\n    result = br.HBVRunoff(tmp_path)\n    assert result.dates.tolist() == [\n        datetime(1990, 1, 1, 0, tzinfo=timezone.utc),\n        datetime(1990, 1, 1, 12, tzinfo=timezone.utc),\n        datetime(1990, 1, 2, 0, tzinfo=timezone.utc),\n    ]\n    np.testing.assert_array_equal(result.transport.loc[92], [10., 20., 30.])\n\n\n@pytest.mark.parametrize(\'stamp\', [\n    \'19900101/2460\', \'19900101/2400\', \'19900230/1200\',\n    \'19900101/12:00\', \'19900101/abc0\',\n])\ndef test_invalid_hbv_timestamp_is_rejected_with_filename(tmp_path, stamp):\n    path = write_var(tmp_path, 91, [stamp])\n    with pytest.raises(ValueError, match=path.name):\n        br.HBVRunoff(tmp_path)\n\n\ndef test_real_world_hbv_times_and_exclusive_time_crop(tmp_path):\n    write_var(tmp_path, 91, [\'20231231/1200\', \'20240101/1200\', \'20240102/1200\'],\n              [3.5, 4.5, 5.5])\n    result = br.HBVRunoff(\n        tmp_path,\n        start=datetime(2023, 12, 31, 12, tzinfo=timezone.utc),\n        stop=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),\n    )\n    assert result.dates.tolist() == [datetime(2024, 1, 1, 12, tzinfo=timezone.utc)]\n    np.testing.assert_array_equal(result.transport.loc[91], [4.5])\n'


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False,
                                     dir=path.parent, prefix='.br-hbv-', newline=None) as stream:
        stream.write(content)
        tmp = Path(stream.name)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f'Missing {SOURCE}: BuildRivers Phase 2A must already be installed')
    current = SOURCE.read_text(encoding='utf-8')
    if current.count(OLD) == 1:
        revised = current.replace(OLD, NEW, 1)
        changed = True
    elif current.count(NEW) == 1:
        revised = current
        changed = False
    else:
        raise SystemExit('Unrecognized HBV date parser; refusing to modify any files')
    ast.parse(revised, filename=str(SOURCE))

    if TEST.exists() and TEST.read_text(encoding='utf-8') != TEST_CONTENT:
        raise SystemExit(f'{TEST} exists with different contents; no files changed')

    if changed:
        write_atomic(SOURCE, revised)
        print(f'Updated {SOURCE}')
    else:
        print('HBV date parser already updated')
    if not TEST.exists():
        write_atomic(TEST, TEST_CONTENT)
        print(f'Created {TEST}')
    print('Run: uv run python -W error::RuntimeWarning -m pytest tests -v -rs')


if __name__ == '__main__':
    main()
