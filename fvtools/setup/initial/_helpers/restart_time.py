"""Millisecond-accurate FVCOM restart time, in UTC MJD or datetime64."""
from __future__ import annotations

import numpy as np
from netCDF4 import num2date

MJD_EPOCH = np.datetime64('1858-11-17T00:00:00', 'ms')
MSECS_PER_DAY = 86_400_000


def restart_day_msec(nc):
    """Read the first Itime/Itime2 record; do not round via float32 time."""
    if ('Itime' not in nc.variables or 'Itime2' not in nc.variables
            or len(nc['Itime']) != 1 or len(nc['Itime2']) != 1):
        raise ValueError('TPXO restart requires one Itime/Itime2 timestamp')
    day, msec = int(nc['Itime'][0]), int(nc['Itime2'][0])
    if not 0 <= msec < MSECS_PER_DAY:
        raise ValueError('Invalid restart Itime2; expected milliseconds in one day')
    return day, msec


def restart_mjd(nc):
    """MJD floating-point array of length 1 (for UTide)."""
    day, msec = restart_day_msec(nc)
    return np.array([day + msec / MSECS_PER_DAY], dtype=np.float64)


def restart_datetime64(nc, *, allow_time_fallback=True):
    """One restart timestamp as datetime64[ms], preferring exact integers."""
    if 'Itime' in nc.variables and 'Itime2' in nc.variables:
        day, msec = restart_day_msec(nc)
        return MJD_EPOCH + np.timedelta64(day, 'D') + np.timedelta64(msec, 'ms')
    if not allow_time_fallback or 'time' not in nc.variables or not len(nc['time']):
        raise ValueError('FVCOM restart has no valid time coordinate')
    value = num2date(nc['time'][0], units=nc['time'].units,
                     only_use_cftime_datetimes=False)
    return np.datetime64(value.strftime('%Y-%m-%dT%H:%M:%S'), 'ms')
