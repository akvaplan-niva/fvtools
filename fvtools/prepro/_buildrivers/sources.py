"""HBV runoff, river-outlet positions and river temperatures."""
import os
from datetime import datetime, timedelta, timezone
from functools import cached_property

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from pyproj import Proj
from scipy import interpolate


class HBVRunoff:
    def __init__(self, hbvfolder, start=None, stop=None):
        """Read HBV .var files, keyed by watershed ID rather than file order.

        All files must describe the same strictly increasing UTC time axis.
        The historical correction for watershed 183 is 6/16 of watershed
        182, regardless of the order in which the files were listed.
        """
        import re
        from pathlib import Path

        folder = Path(hbvfolder)
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() == '.var')
        if not files:
            raise ValueError(f'No HBV .var files found in {folder}')

        series = {}
        common_dates = None
        for path in files:
            match = re.search(r'(\d{3})$', path.stem)
            if match is None:
                raise ValueError(f'Cannot determine watershed ID from {path.name}')
            watershed = int(match.group(1))
            if watershed in series:
                raise ValueError(f'Duplicate HBV watershed ID {watershed}')

            values = pd.read_csv(path, header=None, sep=r'\s+', dtype={0: str})
            if values.shape[1] != 2 or values.empty:
                raise ValueError(f'{path.name} must have timestamp and runoff columns')

            dates = []
            for raw in values.iloc[:, 0]:
                stamp = str(raw).strip()
                # Both YYYYMMDDHH and YYYYMMDD_HH (single separator) occur.
                if re.fullmatch(r'\d{10}', stamp):
                    packed = stamp
                elif re.fullmatch(r'\d{8}\D\d{2}', stamp):
                    packed = stamp[:8] + stamp[-2:]
                else:
                    raise ValueError(f'Invalid HBV timestamp {stamp!r} in {path.name}')
                dates.append(datetime.strptime(packed, '%Y%m%d%H').replace(
                    tzinfo=timezone.utc))
            dates = np.asarray(dates, dtype=object)
            if any(left >= right for left, right in zip(dates[:-1], dates[1:])):
                raise ValueError(f'HBV timestamps must be strictly increasing in {path.name}')
            if common_dates is None:
                common_dates = dates
            elif not np.array_equal(dates, common_dates):
                raise ValueError(f'HBV dates do not match across files: {path.name}')

            runoff = pd.to_numeric(values.iloc[:, 1], errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(runoff).all():
                raise ValueError(f'Invalid HBV runoff values in {path.name}')
            series[watershed] = runoff

        # Preserve the HBV-specific adjustment by IDs, not arbitrary list indexes.
        if 183 in series:
            if 182 not in series:
                raise ValueError('HBV watershed 183 requires watershed 182 for its 6/16 correction')
            series[183] = (6.0 / 16.0) * series[182]

        if start is not None and start < common_dates[0]:
            raise ValueError(f'HBV runoff starts after requested start {start}')
        if stop is not None and stop > common_dates[-1]:
            raise ValueError(f'HBV runoff ends before requested stop {stop}')
        if start is not None and stop is not None and start > stop:
            raise ValueError('HBV start must not be later than stop')

        keep = np.ones(len(common_dates), dtype=bool)
        if start is not None:
            keep &= common_dates > start
        if stop is not None:
            keep &= common_dates < stop
        self.dates = common_dates[keep]
        self.transport = pd.DataFrame.from_dict(series, orient='index').loc[:, keep]

class RiverPositions:
    def __init__(self, path, vassdrag, epsg_code):
        '''
        Connect river outlets identified by fvtools.rivers.NVEforcing.elvis_riveroutlets to the FVCOM mesh
        - path:      e.g. riverdata/river_positions.csv
        - vassdrag:  list of vassdrags covered by the model
        - epsg_code: EPSG code of the projection the river positions is stored in. Should always be in UTM33 (epsg:32633) in Norway.
        '''
        # Set georeference, and prepare a Proj object for the river data
        self.reference = epsg_code
        self.Proj = Proj(epsg_code)

        # Read the river database. Each edited database carries its own
        # immutable original outlet coordinates.
        rivers = pd.read_csv(path).set_index('rivers')
        original_columns = ('x_original', 'y_original')
        missing_columns = [col for col in original_columns if col not in rivers]

        if missing_columns:
            # Only the national, unedited database may omit original coordinates.
            # An old case-specific file must be migrated once instead of silently
            # treating already-moved outlets as their original locations.
            if os.path.basename(os.fspath(path)) != 'river_positions.csv':
                raise ValueError(
                    f'{path} is missing {missing_columns}. '
                    'Use a case database containing x_original and y_original.'
                )
            rivers['x_original'] = rivers['x_outlet']
            rivers['y_original'] = rivers['y_outlet']

        if rivers.index.has_duplicates:
            raise ValueError(f'Duplicate river IDs found in {path}')
        if rivers[['x_original', 'y_original']].isna().any().any():
            raise ValueError(f'Missing original river coordinates in {path}')

        # Filter while retaining both the current and the original coordinates.
        self.rivers = rivers.loc[rivers.vassdragsomraade.isin(vassdrag)].copy()

    @cached_property
    def crs(self):
        return ccrs.epsg(int(self.reference.split(':')[-1]))

    @property
    def x(self):
        return self.rivers.x_outlet.values 

    @x.setter
    def x(self, val):
         self.rivers.x_outlet = val

    @property
    def y(self):
        return self.rivers.y_outlet.values

    @y.setter
    def y(self, val):
        self.rivers.y_outlet = val

    # Derive lon, lat from x,y
    @property
    def lonlat(self):
        _lon, _lat = self.Proj(self.x, self.y, inverse = True)
        return [_lon, _lat]
    
    @property
    def lon(self):
        return self.lonlat[0]
    
    @property
    def lat(self):
        return self.lonlat[1]

    @property
    def names(self):
        '''
        Large rivers have names (not necessarilly unique), small rivers just have their IDs
        '''
        names = []
        for r in self.rivers.iterrows():
            if type(r[1].elvenavn) == str:
                names.append(f'{r[0]} - {r[1].elvenavn}')
            else:
                names.append(r[0])
        return names

    def crop(self, indices):
        '''
        Remove rivers that are not requested from the rivers dataframe
        '''
        self.rivers = self.rivers.iloc[indices]

class RiverTemperatures:
    '''
    Prepares climatological temperature at each watershed area for the FVCOM experiment
    - Also provides preliminary 
    '''
    def __init__(self, riverpath, start, stop):
        """Load temperature climatology and optional observed river temperatures."""
        from pathlib import Path

        self.start, self.stop = start, stop
        root = Path(riverpath)
        self.climatology = pd.read_csv(root / 'temperature_climatology.csv', index_col=0)
        self.climatology.columns = self.climatology.columns.astype(int)
        self.observations = {}
        observation_folder = root / 'temperature_observations'
        if observation_folder.is_dir():
            for path in sorted(observation_folder.glob('*.csv')):
                try:
                    observation = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    continue
                # Malformed or empty station files cannot replace climatology.
                if 'time' not in observation:
                    continue
                observation['time'] = pd.to_datetime(
                    observation['time'], errors='coerce', utc=True)
                self.observations[path.stem] = observation

    @property
    def time(self):
        '''
        Days we will force the model
        '''
        return [self.start + timedelta(days = n) for n in range((self.stop - self.start).days + 1)]

    @property
    def timestamp(self):
        """Unix seconds for the forcing time axis, suitable for interpolation."""
        return np.array([time.timestamp() for time in self.time], dtype=float)
    
    @property
    def day_of_year(self):
        '''
        Which day of the year each time is
        '''
        return np.array([t.timetuple().tm_yday for t in self.time])

    @property
    def vassdrag_temperature_at_time(self):
        '''
        Temperatures on the day we force the model
        '''
        temp = self.climatology.T
        return temp.loc[self.day_of_year].T

    def make_individual_river_climatology(self, Rivers):
        '''
        Connect individual rivers to the climatoligical temperature on the days we need data to force FVCOM
        '''
        river_temp = self.vassdrag_temperature_at_time.loc[Rivers.rivers.vassdragsomraade]
        river_temp['elv-ID'] = Rivers.rivers.index
        return river_temp.set_index('elv-ID')

    def insert_observed_temperature(self, river_temp, vassdrag):
        """Use quality-controlled station data only when coverage is adequate.

        Interpolation uses UTC Unix seconds. Insufficient or malformed station
        data leave the corresponding climatology untouched.
        """
        target_time = np.asarray(self.timestamp, dtype=float)
        if not len(target_time):
            return river_temp

        selected = set(np.asarray(vassdrag).tolist())
        required = {'time', 'quality', 'filtered_temperature'}
        for key, original in self.observations.items():
            try:
                watershed = int(key.split('-', 1)[0])
            except ValueError:
                continue
            if watershed not in selected or key not in river_temp.index:
                continue
            if not required.issubset(original.columns):
                print(f'  - Station {key} is missing required columns')
                continue

            station = original.copy()
            station['time'] = pd.to_datetime(
                station['time'], errors='coerce', utc=True)
            station['quality'] = pd.to_numeric(station['quality'], errors='coerce')
            station['filtered_temperature'] = pd.to_numeric(
                station['filtered_temperature'], errors='coerce')
            station = station.loc[
                (station['quality'] > 1)
                & station['time'].notna()
                & np.isfinite(station['filtered_temperature'])
            ].sort_values('time', kind='stable').drop_duplicates('time', keep='last')
            if len(station) < 2:
                print(f'  - Station {key} has insufficient valid observations')
                continue

            # No extrapolation; stations must span the requested forcing axis.
            first, last = station['time'].iloc[0], station['time'].iloc[-1]
            model_start = pd.Timestamp(self.time[0])
            model_stop = pd.Timestamp(self.time[-1])
            if model_start.tzinfo is None:
                model_start = model_start.tz_localize('UTC')
            if model_stop.tzinfo is None:
                model_stop = model_stop.tz_localize('UTC')
            if first > model_start or last < model_stop:
                print(f'  - Station {key} did not cover the simulation period')
                continue

            valid_days = station.loc[
                (station['time'] >= model_start) &
                (station['time'] <= model_stop), 'time'
            ].dt.floor('D').nunique()
            if valid_days / len(self.time) < 0.9:
                print(f'  - Station {key} had too many gaps in the simulation period')
                continue

            sample_time = station['time'].astype('int64').to_numpy(dtype=float) / 1e9
            sample_temp = station['filtered_temperature'].to_numpy(dtype=float)
            # Explicit seconds, not pd.to_datetime(seconds) interpreted as ns.
            out = np.interp(target_time, sample_time, sample_temp)
            print(f'  - Replacing climatology with observations at {key}')
            river_temp.loc[key] = out
        return river_temp
