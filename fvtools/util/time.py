from netCDF4 import num2date as n2d
from netCDF4 import date2num as d2n
from datetime import datetime
import pandas as pd


epoch_pandas = pd.Timestamp('1858-11-17 00:00:00')
epoch_datetime = datetime(1858, 11, 17, 0, 0, 0)

def date2num(y, mo, d, h=0, mi = 0, s = 0):
    d = datetime(y,mo,d,h,mi,s)
    return d2n(d, units = 'days since 1858-11-17 00:00:00')

def num2date(dnum):
    return n2d(dnum, units = 'days since 1858-11-17 00:00:00')

def timestamps_from_days_since_epoch(days):
    """
    Convert from iterable of days since epoch to a list of pandas Timestamps
    """
    return [pd.Timestamp(epoch_pandas) + pd.Timedelta(days=d) for d in days]
