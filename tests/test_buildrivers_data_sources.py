"""Behavioral tests for BuildRivers data-source processing (no NIRD access needed)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from fvtools.prepro import BuildRivers as br


UTC = timezone.utc


def ts(day, hour=0):
    return datetime(2024, 1, day, hour, tzinfo=UTC)


def write_hbv(folder, watershed, pairs, *, filename=None):
    path = folder / (filename or f'HBV_{watershed:03d}.var')
    path.write_text(''.join(f'{date} {runoff}\n' for date, runoff in pairs), encoding='utf-8')
    return path


def hbv_series():
    return [(f'2024010{i}00', float(10 * i)) for i in range(1, 6)]


def write_temp_db(tmp_path, observations=None):
    root = tmp_path / 'temp'
    root.mkdir()
    pd.DataFrame(np.vstack((np.arange(1, 32, dtype=float),
                                np.arange(101, 132, dtype=float))),
                 index=[91, 92], columns=range(1, 32)).to_csv(
                     root / 'temperature_climatology.csv')
    if observations is not None:
        obs_dir = root / 'temperature_observations'
        obs_dir.mkdir()
        for name, frame in observations.items():
            frame.to_csv(obs_dir / (name + '.csv'), index=False)
    return root


def obs_frame(days, values=None, quality=None):
    days = list(days)
    return pd.DataFrame({
        'time': [ts(d).isoformat() for d in days],
        'quality': ([2] * len(days)) if quality is None else quality,
        'filtered_temperature': ([float(d) for d in days] if values is None else values),
    })


def clim_row(size=10, value=-10.):
    return pd.DataFrame(np.full((1, size), value), index=['91-001'])


def test_hbv_identifies_ids_independent_of_filename_order(tmp_path):
    pairs = hbv_series()
    write_hbv(tmp_path, 183, [(date, 999.) for date, _ in pairs], filename='A_183.var')
    write_hbv(tmp_path, 91, [(date, v + 2.) for date, v in pairs], filename='B_091.var')
    write_hbv(tmp_path, 182, pairs, filename='Z_182.var')
    (tmp_path / 'README.txt').write_text('Ignored', encoding='utf-8')
    runoff = br.HBVRunoff(tmp_path, start=ts(1), stop=ts(5))
    assert sorted(runoff.transport.index.tolist()) == [91, 182, 183]
    np.testing.assert_allclose(runoff.transport.loc[182], [20., 30., 40.])
    np.testing.assert_allclose(runoff.transport.loc[183], [7.5, 11.25, 15.])
    np.testing.assert_allclose(runoff.transport.loc[91], [22., 32., 42.])
    assert runoff.dates.tolist() == [ts(2), ts(3), ts(4)]


def test_hbv_183_requires_182_if_present(tmp_path):
    write_hbv(tmp_path, 183, hbv_series())
    with pytest.raises(ValueError, match='requires watershed 182'):
        br.HBVRunoff(tmp_path)


def test_hbv_single_watershed_and_no_crop(tmp_path):
    write_hbv(tmp_path, 91, hbv_series())
    runoff = br.HBVRunoff(tmp_path)
    np.testing.assert_array_equal(runoff.transport.loc[91], [10., 20., 30., 40., 50.])
    assert len(runoff.dates) == 5


def test_hbv_supports_hour_and_separated_timestamps(tmp_path):
    write_hbv(tmp_path, 91, [('20240101_00', 1.), ('20240101_12', 2.), ('20240102_00', 3.)])
    runoff = br.HBVRunoff(tmp_path)
    assert runoff.dates.tolist() == [ts(1), ts(1, 12), ts(2)]


@pytest.mark.parametrize('bad', ['2024', '2024010199', '20240101AA'])
def test_hbv_invalid_timestamp_rejected(tmp_path, bad):
    write_hbv(tmp_path, 91, [(bad, 1.)])
    with pytest.raises(ValueError):
        br.HBVRunoff(tmp_path)


def test_hbv_mismatching_time_series_fails(tmp_path):
    write_hbv(tmp_path, 91, hbv_series())
    write_hbv(tmp_path, 92, hbv_series()[:-1])
    with pytest.raises(ValueError, match='dates do not match'):
        br.HBVRunoff(tmp_path)


def test_hbv_unsorted_times_fail(tmp_path):
    write_hbv(tmp_path, 91, [('2024010200', 2.), ('2024010100', 1.)])
    with pytest.raises(ValueError, match='strictly increasing'):
        br.HBVRunoff(tmp_path)


def test_hbv_duplicate_watershed_id_fails(tmp_path):
    pairs = hbv_series()
    write_hbv(tmp_path, 91, pairs, filename='A_091.var')
    write_hbv(tmp_path, 91, pairs, filename='B_091.var')
    with pytest.raises(ValueError, match='Duplicate HBV watershed'):
        br.HBVRunoff(tmp_path)


def test_hbv_no_var_files_fails(tmp_path):
    with pytest.raises(ValueError, match='No HBV .var files'):
        br.HBVRunoff(tmp_path)


def test_hbv_non_numeric_runoff_fails(tmp_path):
    write_hbv(tmp_path, 91, [('2024010100', 'not-a-number')])
    with pytest.raises(ValueError, match='Invalid HBV runoff'):
        br.HBVRunoff(tmp_path)


def test_hbv_rejects_uncovered_period(tmp_path):
    write_hbv(tmp_path, 91, hbv_series())
    with pytest.raises(ValueError, match='starts after'):
        br.HBVRunoff(tmp_path, start=ts(1)-timedelta(days=1))
    with pytest.raises(ValueError, match='ends before'):
        br.HBVRunoff(tmp_path, stop=ts(5)+timedelta(days=1))


def test_temperature_climatology_and_utc_timestamps(tmp_path):
    root = write_temp_db(tmp_path)
    temp = br.RiverTemperatures(root, ts(1), ts(3))
    assert temp.timestamp.tolist() == [ts(i).timestamp() for i in range(1, 4)]
    assert temp.day_of_year.tolist() == [1, 2, 3]
    # Climatology can be loaded without a directory of station observations.
    assert not temp.observations
    assert temp.vassdrag_temperature_at_time.shape == (2, 3)


def test_temperature_observations_replace_climatology_at_utc_seconds(tmp_path):
    # 10 in-window daily observations plus endpoints avoiding extrapolation.
    frame = obs_frame(range(1, 11), values=[2. * d + 1. for d in range(1, 11)])
    root = write_temp_db(tmp_path, {'91-001': frame})
    obj = br.RiverTemperatures(root, ts(1), ts(10))
    result = obj.insert_observed_temperature(clim_row(), [91])
    np.testing.assert_allclose(result.loc['91-001'], [2 * d + 1. for d in range(1, 11)])


def test_temperature_observations_are_sorted_and_deduplicated(tmp_path):
    frame = obs_frame(range(1, 11))
    duplicate = obs_frame([5], values=[100.])
    frame = pd.concat([frame, duplicate], ignore_index=True)
    frame = frame.iloc[[2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10]]
    root = write_temp_db(tmp_path, {'91-001': frame})
    obj = br.RiverTemperatures(root, ts(1), ts(10))
    got = obj.insert_observed_temperature(clim_row(), [91])
    assert np.isfinite(got.to_numpy()).all()
    assert got.loc['91-001', 0] == 1.
    # sort_values + drop_duplicates(keep='last') selects duplicate 100 at day 5
    assert got.loc['91-001', 4] == 100.


def test_temperature_bad_quality_and_empty_stations_keep_climatology(tmp_path):
    frame = obs_frame(range(1, 11), quality=[1] * 10)
    root = write_temp_db(tmp_path, {'91-001': frame, '91-002': pd.DataFrame(columns=frame.columns)})
    obj = br.RiverTemperatures(root, ts(1), ts(10))
    unchanged = clim_row()
    got = obj.insert_observed_temperature(unchanged, [91])
    np.testing.assert_array_equal(got, np.full((1, 10), -10.))


def test_temperature_incomplete_coverage_keeps_climatology(tmp_path):
    frame = obs_frame([1, 2, 3, 4, 5, 6, 7, 10])
    root = write_temp_db(tmp_path, {'91-001': frame})
    obj = br.RiverTemperatures(root, ts(1), ts(10))
    got = obj.insert_observed_temperature(clim_row(), [91])
    np.testing.assert_array_equal(got, np.full((1, 10), -10.))


def test_temperature_no_extrapolation_for_uncovered_ends(tmp_path):
    frame = obs_frame(range(2, 10))
    root = write_temp_db(tmp_path, {'91-001': frame})
    obj = br.RiverTemperatures(root, ts(1), ts(10))
    got = obj.insert_observed_temperature(clim_row(), [91])
    np.testing.assert_array_equal(got, np.full((1, 10), -10.))


def test_temperature_missing_columns_and_unselected_ids_do_not_replace(tmp_path):
    frame = obs_frame(range(1, 11))
    root = write_temp_db(tmp_path, {'91-001': frame.drop(columns='filtered_temperature'), '92-002': frame})
    obj = br.RiverTemperatures(root, ts(1), ts(10))
    got = obj.insert_observed_temperature(clim_row(), [91])
    np.testing.assert_array_equal(got, np.full((1, 10), -10.))


def test_temperature_observation_missing_river_row_does_not_crash(tmp_path):
    root = write_temp_db(tmp_path, {'91-999': obs_frame(range(1, 11))})
    obj = br.RiverTemperatures(root, ts(1), ts(10))
    got = obj.insert_observed_temperature(clim_row(), [91])
    np.testing.assert_array_equal(got, np.full((1, 10), -10.))
