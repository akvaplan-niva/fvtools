"""The gap filler must leave the hours that have data unchanged.

FVCOM nesting files store ``time`` as float32 days since 1858-11-17. Near
2024 one float32 step is 2**-8 day (about 5.6 minutes), so hourly times are
off by up to about 3 minutes; ``Itime``/``Itime2`` are exact. If the filler
interpolates from the float32 times onto the exact new times, temperature
and salinity change at hours where nothing was missing.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
from netCDF4 import Dataset

import fvtools.setup.nesting.fill_nesting_gaps as fg

FVCOM_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)
START = datetime(2024, 11, 1, tzinfo=timezone.utc)


def _write_nest_times(path, times):
    """Write the time variables of a nesting file, as FVCOM stores them."""
    days = np.array([(t - FVCOM_EPOCH).total_seconds() / 86400 for t in times])
    itime = np.floor(days).astype(np.int32)
    itime2 = np.round((days - itime) * 86_400_000).astype(np.int32)
    with Dataset(path, "w") as d:
        d.createDimension("time", None)
        d.createVariable("time", "f4", ("time",))[:] = days
        d.createVariable("Itime", "i4", ("time",))[:] = itime
        d.createVariable("Itime2", "i4", ("time",))[:] = itime2


def _gap_filler(tmp_path, hours):
    path = tmp_path / "nest.nc"
    _write_nest_times(path, [START + timedelta(hours=h) for h in hours])
    new_timeseries, inds_copy, inds_overlap_orig, inds_fill = fg.get_new_timeseries(str(path))
    filler = fg.FillGaps(str(path), None, new_timeseries,
                         inds_copy, inds_overlap_orig, inds_fill, {}, {})
    return filler, inds_copy, inds_overlap_orig, inds_fill


def test_original_time_is_exact(tmp_path):
    hours = [h for h in range(48) if h not in (20, 21, 22)]
    filler, inds_copy, inds_overlap_orig, inds_fill = _gap_filler(tmp_path, hours)

    assert len(inds_fill) == 3
    np.testing.assert_allclose(filler.original_time[inds_overlap_orig],
                               filler.new_time[inds_copy], rtol=0, atol=1e-9)


def test_existing_temperature_and_salinity_are_kept(tmp_path):
    hours = [h for h in range(10 * 24) if h not in range(100, 106)]
    filler, inds_copy, inds_overlap_orig, inds_fill = _gap_filler(tmp_path, hours)

    rng = np.random.default_rng(0)
    n, layers = len(hours), 3
    temp = 8 + rng.standard_normal((n, layers))
    salt = 35 + 0.1 * rng.standard_normal((n, layers))
    m2_period_days = 12.4206012 / 24
    zeta = 0.5 * np.cos(2 * np.pi * filler.original_time / m2_period_days)

    node = fg.NodeData(
        original_time=filler.original_time,
        new_time=filler.new_time,
        zeta=zeta,
        temp=temp,
        salinity=salt,
        weight_node=np.ones(n),
        idx=0,
        lat=61.0,
        copy_inds=inds_copy,
        inds_overlap_orig=inds_overlap_orig,
        replace_inds=inds_fill,
        solver_settings={"epoch": "1858-11-17", "verbose": False, "constit": ["M2"]},
        reconstruct_settings={"epoch": "1858-11-17", "verbose": False},
    )
    node.fill_gaps()

    np.testing.assert_allclose(node.temp[inds_copy], temp[inds_overlap_orig], rtol=0, atol=1e-12)
    np.testing.assert_allclose(node.salinity[inds_copy], salt[inds_overlap_orig], rtol=0, atol=1e-12)
    assert np.isfinite(node.temp[inds_fill]).all()
    assert np.isfinite(node.salinity[inds_fill]).all()
