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
    def __init__(self, hbvfolder, start = None, stop = None):
        '''
        Read all hbv output files in the hbvfolder directory
        - Assumes that all .var files in the hbv folder are runoff data.
        - crop to time between start and stop
        '''
        # Load data for the full country
        data = []
        for i, file in enumerate(os.listdir(hbvfolder)):
            # Figure out which number this vassdragsområde is
            vassdragsomraade = int(file.split('.var')[0][-3:])
        
            # Load the data
            _data = pd.read_csv(f'{hbvfolder}{file}', header = None, sep=r'\s+')
        
            # Load the dates, will have the same coverage in all files
            if i == 0:
                dates = [datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[9:11]), tzinfo = timezone.utc) for d in _data[0]]
        
            data.append(_data.rename(columns = {1: vassdragsomraade}).drop(columns=0))
        
        # Re-scale vassdragsområde 183 (python index 182), as suggested by James Sample at NIVA
        data[182] = 6/16 * data[181]
        self.transport = pd.concat(data, axis = 1).T
        self.dates = np.array(dates)

        assert any(self.dates <= start), f'the database does not cover dates prior to {self.dates[0]}'
        assert any(self.dates >= stop),  f'the database does not cover dates after {self.dates[-1]}'
            
        
        # Crop to requested time-span
        crop = np.logical_and(self.dates > start, self.dates < stop)

        self.transport = self.transport.loc[:, crop]
        self.dates = self.dates[crop]

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
        '''
        Read observed river temperatures and river climatologies
        '''
        self.start, self.stop = start, stop
        self.climatology = pd.read_csv(f'{riverpath}/temperature_climatology.csv', index_col = 0)
        self.climatology.columns = self.climatology.columns.astype(int)
        self.observations = {}
        observation_folder = f'{riverpath}/temperature_observations/'
        
        for f in os.listdir(observation_folder):
            if '.csv' in f:
                _tmp = pd.read_csv(f'{observation_folder}{f}')
                _tmp['time'] = pd.to_datetime(_tmp.time)
                self.observations[f.split('.csv')[0]] = _tmp

    @property
    def time(self):
        '''
        Days we will force the model
        '''
        return [self.start + timedelta(days = n) for n in range((self.stop - self.start).days + 1)]

    @property
    def timestamp(self):
        return pd.to_datetime([t.timestamp() for t in self.time])
    
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
        '''
        Use the observed temperature instead of the climatology where we have it
        '''
        # Identify which temperature observation are within the model domain
        within_domain = [key for key in self.observations.keys() if int(key.split('-')[0]) in vassdrag]
        
        # Interpolate observations to the forcing time
        if any(within_domain):
            for key in within_domain:
                observation = self.observations[key]
                
                # Only use quality controlled data
                observation = observation[observation.quality > 1]
        
                # Check if the observation covers the modelled period, crop the array is so
                if observation.time.iloc[0] < self.time[0] and observation.time.iloc[-1] > self.time[-1]:
                    observation = observation[np.logical_and(observation.time >= self.start - timedelta(days = 7), observation.time <= self.stop + timedelta(days = 7))]
                else:
                    print(f'  - Station {key} did not cover the simulation period')
                    continue
        
                # Require that the coverage of temperature data (daily) is at least 90% of the model period (to avoid making up too much data)
                if len(observation)/len(self.time) < 0.9:
                    print(f'  - Station {key} had too many gaps in the simulation period')
                    continue
                    
                # Interpolate the observed temperature to the forcing data
                print(f'  - Replacing climatology with observations at {key}')
                f = interpolate.interp1d([t.timestamp() for t in observation.time], observation.filtered_temperature)
                river_temp.loc[key] = f(self.timestamp)
        return river_temp
