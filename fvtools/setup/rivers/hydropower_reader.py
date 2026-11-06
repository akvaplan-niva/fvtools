# Hydropower-reader
import numpy as np
import pandas as pd
import os
from datetime import datetime
from functools import cached_property

def main(vannkraft_folder = '/work/hes/Vannkraft', plant_name = 'Evanger', 
         slukevne = None):
    '''
    Read power production timeseries, convert to runoff estimate
    - 
    '''
    # Find all relevant csv files
    vannkraft_files  = get_csv_files(vannkraft_folder)
    vannkraft_files.sort() # sort in ascending order

    # Read data from these csv filesgit 
    power_production = PowerReader(vannkraft_files)

    # Extract a specific station
    this_station = power_production.plant(plant_name) # will return all generators in one pandas series

    # Initialize a Power Station instance to deal with the irregular time-stamping
    Evanger = PowerStation(this_station)

class PowerReader:
    '''
    Read all power production datasets
    - basically just does operations on a pandas series object
    '''
    def __init__(self, 
                 vannkraft_files,
                 area_code = '10YNO', 
                 production_type = 'Hydro Water Reservoir'):
        '''
        csv_files:  list of files to be read
        plant_name: name of production plant to 
        '''
        # Load production data from the csv files
        # ----
        self._load_csv_files(vannkraft_files, area_code, production_type)
        self._sort_by_datetime()
        self.production = self.production.filter(columns=['ActualGenerationOutput', 'PowerSystemResourceName', 'InstalledGenCapacity'])

    def _load_csv_files(self, vannkraft_files, area_code, production_type):
        '''
        load production for plants in Norway
        '''
        self.production = pd.Series(dtype='object')
        # Load data, remove plants outside of Norway, and plants that are not of the desired production_type
        # ----
        print('Loading:\n----')
        for i, file in enumerate(vannkraft_files):
            print(f'- {file}')
            tmp = pd.read_csv(file, delimiter = '\t') # load file
            tmp = tmp[tmp.AreaCode.str.contains(area_code)] # remove plants outside of Norway

            # remove non-"production_type" plants, concatenate norwegian plants to one single dataseries
            self.production = pd.concat([self.production,
                                         tmp[tmp.ProductionType.str.contains(production_type)]
                                         ],
                                         axis = 0)

    def _sort_by_datetime(self):
        '''
        Convert DateTime string into datetime64, sort the data by datetime and use datetime for indexing
        '''
        self.production['DateTime'] = pd.to_datetime(self.production['DateTime'])
        self.production = self.production.sort_values(by='DateTime')
        self.production = self.production.set_index('DateTime')

    def plant(self, plant_name):
        '''
        get production from this plant name (as sum of all generators)
        '''
        tmp = self.production[self.production.PowerSystemResourceName.str.contains(plant_name)]
        return tmp

class PowerStation:
    '''
    Handles data from a single station
    '''
    def __init__(self, this_station):
        '''
        Restructure input array so that it gives a single timeseries
        '''
        self.input = this_station
        self._add_shared_start_stop()

    @property
    def generator_names(self):
        return self.input.PowerSystemResourceName.unique()

    @property
    def generator_capacity(self):
        for name in self.generator_names:
            stations = self.input.PowerSystemResourceName
            self._cap.append(self.input.InstalledGenCapacity[self.input.PowerSystemResourceName.str.contains(name)].max())
        return self._cap

    @property
    def installed_capacity(self):
        return np.sum(self.generator_capacity)

    @cached_property
    def production(self):
        '''
        Power production at each generator interpolated to an evenly spaced timeseries covering the same time interval
        '''
        data = {}
        for name in self.generator_names:
            data[name] = self.input[self.input.PowerSystemResourceName==name].ActualGenerationOutput.resample('H').interpolate().fillna(method='bfill').fillna(method='ffill')
        return pd.DataFrame(data)

    
    def _add_shared_start_stop(self):
        '''
        Add start- and stop time to generators so that all start/stop at the same time.
        Done to prepare the DataFrame for resampling and interpolation.
        '''
        min_time = self.input.index.min()
        max_time = self.input.index.max()

        def _write_new_row(new_time, row):
            row.name = new_time
            row.ActualGenerationOutput = np.nan
            return row

        for generator in self.generator_names:
            if min_time < self.input[self.input.PowerSystemResourceName==generator].index.min():
                new_row = _write_new_row(min_time, self.input[self.input.PowerSystemResourceName==generator].iloc[0])
                self.input = self.input.append(new_row)

            if max_time > self.input[self.input.PowerSystemResourceName==generator].index.max():
                new_row = _write_new_row(min_time, self.input[self.input.PowerSystemResourceName==generator].iloc[0])
                self.input = self.input.append(new_row)

        # Sort at the end
        self.input = self.input.sort_index()


def get_csv_files(vannkraft_folder):
    '''
    Quite simply finds all files with "Generation" in- and ".csv" appended to their name
    '''
    files = os.listdir(vannkraft_folder)
    return [f'{vannkraft_folder}/{file}' for file in files if file.find('Generation')>-1 and file.find('.csv')>-1]