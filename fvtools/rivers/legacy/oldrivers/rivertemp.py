import os
import chardet
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fvtools.grid.tools import num2date, date2num
from scipy import interpolate
from scipy.signal import filtfilt
from datetime import datetime, timedelta, timezone

'''
This river tempreature class was written to deal with temperature data that NVE sent us in .csv formatted files.

We have since moved to downloading the temperatures ourselves through API calls to hydapi, and processing the data using pandas.
'''

class RiverTemperatures:
    """
    Scans the folder contatining river temperatures.
    - Compiles a yearly "typical" river temperature file
    - Looks for specific timesteps in specific vassdrag to get as in-situ temperatures as possible
    - Most of the data-processing here can most likely be replaced using pandas
    """
    def __init__(self, info, vassdrag, casename, start_date):
        """
        What temperature should the rivers in the domain have?
        """
        self.casename = casename
        self.info = info
        self.model_vassdrag = vassdrag
        self.vassdrag = []
        _start      = start_date.split('-') # start date as numbers
        _start_date = datetime(int(_start[0]), int(_start[1]), int(_start[2]), tzinfo = timezone.utc) - timedelta(days = 60)
        self.min_date = date2num([_start_date])[0]
        if info['compile river']:
            self.compile_temperature() # For large models spanning many vassdrags
        else:
            self.read_temperature()    # For small models nested FVCOM2FVCOM

    def read_temperature(self):
        """
        Read pre-compiled temperature file
        """
        riverfile = self.info['rivertemp']
        if riverfile.split('.')[-1] == 'npy':
            data = np.load(riverfile, allow_pickle = True)
            print(f'- {riverfile}')

        else:
            files     = os.listdir(self.info['rivertemp'])
            riverfile = [f for f in files if f.split('.')[-1] == 'npy']
            if len(riverfile) == 1:
                data = np.load(self.info['rivertemp']+riverfile[0], allow_pickle = True)
                print(f'- {self.info["rivertemp"]}{riverfile[0]}')

            else:
                raise ValueError(f"{self.info['rivertemp']} did not lead to a numpy file or to a folder with only one numpy file in it")

        self.average_temp = data.item()['average temp']
        self.river_temp = data.item()['temp']
        self.river_time = data.item()['time']
        self.vassdrag   = data.item()['vassdrag']

    def compile_temperature(self):
        """
        This routine checks the rivertemperature folder, and compiles a new RiverTemperature file
        """
        # Find the csv files containing river temperatures
        # ----
        folder_files = os.listdir(self.info['rivertemp'])
        all_files = [f for f in folder_files if f.split('.')[-1] == 'csv']

        # Remove files that are not part of our vassdrag
        # ----
        self.files = [f for f in all_files if int(f.split('.')[0]) in self.model_vassdrag]
        if not any(self.files):
            raise ValueError('None of the river temperatures originate from measurements in the model domain!')

        data, mintime, maxtime = self.read_river_files()
        self.river_time = np.arange(np.ceil(mintime), np.floor(maxtime), 1/24) # interpolate to hourly values
        self.raw_temp   = np.nan*np.ones((len(self.river_time),len(self.files)))
        data = self.remove_jumps(data)
        data = self.insert_yearly_statistics(data)
        self.river_to_rivertime(data)
        self.impose_lower_cutoff()
        self.filter_river_temperatures()
        self.river_temp = self.filtered_temp
        dates = num2date(self.river_time)
        self.set_average_temp(dates)

        # Store river data to a .npy file and return to main
        # ----
        data = {}
        data['average temp'] = self.average_temp
        data['temp'] = self.river_temp
        data['time'] = self.river_time
        data['vassdrag'] = self.vassdrag
        print(f" - Store compiled temperature file to: {self.info['rivertemp']}{self.casename}_temperatures.npy")
        np.save(f"{self.info['rivertemp']}/{self.casename}_temperatures.npy", data)

    def read_river_files(self):
        '''
        Read data from the raw river temperature files
        '''
        # Define preliminary min- and max time for file
        # ----
        mintime = 10**9; maxtime = 0
        data    = []
        plt.figure()
        for _file in self.files:
            print(f'  - {_file}')
            _data = self.read_vassdrag_temperatures(_file)
            data.append(_data)
            mintime = min(mintime, min(_data['time']))
            maxtime = max(maxtime, max(_data['time']))
            plt.plot(_data['datetime'], _data['temp'], label = _file)

        plt.title('All raw temperatures')
        plt.legend()
        return data, mintime, maxtime

    def read_vassdrag_temperatures(self, _file):
        """
        Read river temperature excel files (see if some of the other stuff can be done in pandas here...)
        """
        out = {}
        out['id'] = _file.split('_')[0]

        with open(self.info['rivertemp']+_file, 'rb') as _f:
            result = chardet.detect(_f.read())

        # read the file
        data = pd.read_csv(self.info['rivertemp']+_file,
                           skiprows = 1, delimiter = ';',encoding=result['encoding']).to_numpy()

        # convert to datetime-format
        time = data[:,0]
        date = []
        temp = data[:,1]
        missing = np.where(temp < -100)
        temp[missing] = np.nan
        nan_ind = []
        for i, tid in enumerate(time):
            if tid is np.nan:
                nan_ind.append(i)
                continue
            try:
                year    = int(tid.split('-')[0])
                month   = int(tid.split('-')[1])
                day     = int(tid.split('-')[2].split(' ')[0])
                hour    = int(tid.split(' ')[1].split(':')[0])
                minutes = int(tid.split(' ')[1].split(':')[1])
                date.append(datetime(year, month, day, hour, minutes, tzinfo = timezone.utc))
            except:
                day     = int(tid.split('.')[0])
                month   = int(tid.split('.')[1])
                year    = int(tid.split('.')[2].split(' ')[0])
                hour    = int(tid.split(' ')[1].split(':')[0])
                minutes = int(tid.split(' ')[1].split(':')[1])
                date.append(datetime(year, month, day, hour, minutes, tzinfo = timezone.utc))

        if any(nan_ind):
            temp = np.delete(temp, nan_ind)

        # Remove obvious spikes
        temp         = np.array(temp, dtype = float)
        tolerance_p  = np.nanmean(temp) + 2.25*np.nanstd(temp)
        tolerance_m  = np.nanmean(temp) - 2.25*np.nanstd(temp)
        inds_p       = np.where(temp>tolerance_p)[0]
        inds_m       = np.where(temp<tolerance_m)[0]
        temp[inds_p] = np.nan
        temp[inds_m] = np.nan

        # store temperatures, date and vassdrag
        out['temp']  = temp
        out['time']  = netCDF4.date2num(date, units = 'days since 1858-11-17 00:00:00')
        out['datetime'] = date
        out['Vdrag'] = int(out['id'].split('.')[0])
        return out

    def remove_jumps(self, data):
        '''
        The data can contain suddent jumps, that's indicative of bad data so we remove them.
        '''
        for _data in data:
            diff = np.diff(_data['temp'])
            std  = np.nanstd(diff)
            threshold = 2.25*std # Basically assuming that large chunks of the data is noise
            jump = False
            for i in range(len(_data['temp'])-1):
                if np.isnan(_data['temp'][i]):
                    jump = False
                    continue
                if not jump:
                    i_old = i
                if np.abs(_data['temp'][i+1]-_data['temp'][i_old]) > threshold:
                    _data['temp'][i+1] = np.nan
                    jump = True
                else:
                    jump = False
        return data

    def insert_yearly_statistics(self, data):
        '''
        Much of the data is patchy, replace gaps with historical average of data for that day
        '''
        for _data in data:
            # Get time as datetime object and daynumber of year for each timestamp
            time      = num2date(_data['time'])
            day_num   = np.array([t.timetuple().tm_yday for t in time])

            # Create temperature statistics of given date
            temp_stat = np.nan*np.ones((max(day_num),))
            days      = np.arange(1,max(day_num)+1)
            for day in days:
                inds = np.where(day_num == day)[0]                 # find all measured years with this day
                temp_stat[day-1] = np.nanmean(_data['temp'][inds]) # python index starts at 0, day index start at 1
            temp_full = temp_stat[day_num-1] # Create an array of statistic temperature covering the model period

            # replace missing temperatures with temp_full
            nans      = np.isnan(_data['temp'])
            _data['temp'][nans] = temp_full[nans]
            zero_rivs = np.where(_data['temp']<0)[0]
            _data['temp'][zero_rivs] = 0
            _data['year_temp'] = temp_stat
        return data

    def river_to_rivertime(self, data):
        '''
        Interpolate raw river data to the time indices we will be forcing in FVCOM
        '''
        self.river_temp = np.nan*np.ones((len(self.river_time),len(self.files)))
        river_dates     = num2date(self.river_time)
        day_num         = np.array([t.timetuple().tm_yday for t in river_dates])
        self.year_temp  = np.nan*np.ones((max(day_num),len(self.files)))

        for i, _data in enumerate(data):
            time = _data['time']
            temp = _data['temp']
            f    = interpolate.interp1d(time, temp, bounds_error = False)
            self.river_temp[:,i] = f(self.river_time)
            self.vassdrag.append(_data['Vdrag'])
            nan_inds  = np.where(np.isnan(self.river_temp[:,i]))[0]
            nan_dates = day_num[nan_inds]
            _data['year_temp'] = np.append(_data['year_temp'], _data['year_temp'][-1])
            self.river_temp[nan_inds,i] = _data['year_temp'][nan_dates-1]

    def impose_lower_cutoff(self):
        '''
        Removes very old temperature now that we have done the yearly statistics
        '''
        too_early = np.where(self.river_time<self.min_date)[0][-1]
        self.river_temp = self.river_temp[too_early:]
        self.river_time = self.river_time[too_early:]

    def filter_river_temperatures(self):
        print(' - Filter temperature to reduce noise')
        n = 80
        b = [1.0/n] * n
        a = 1
        dates = num2date(self.river_time)

        self.filtered_temp = np.nan*np.ones((len(self.river_time),len(self.files)))
        inds  = np.arange(len(self.river_time))
        for i, temp in enumerate(self.river_temp.T):
            not_nans   = ~np.isnan(temp)
            if any(np.isnan(temp)):
                first      = min(inds[not_nans])
                last       = max(inds[not_nans])
                temp       = temp[first:last]
                nans, x    = self.nan_helper(temp)
                temp[nans] = np.interp(x(nans), x(~nans), temp[~nans])
                yy = filtfilt(b,a,temp)
                self.filtered_temp[first:last, i] = yy
            else:
                yy = filtfilt(b,a,temp)
                self.filtered_temp[:, i] = yy

            # Force user to do a quick QC of each river temperature csv
            plt.figure()
            plt.plot(dates, self.river_temp[:,i], c = 'r', label = 'no filter applied')
            plt.plot(dates, self.filtered_temp[:,i], c = 'k', label = 'low pass filtered')
            plt.title(f'River temperature at: {self.files[i].split(".csv")[0]}')
            plt.legend()

    def set_average_temp(self, dates):
        '''
        The average temperature will be used by rivers we don't have data from (typically small ones).

        We just fit a curve that doesn't get as warm as the warmest rivers, but much warmer than the coldest ones.
        '''
        self.average_temp  = np.nanmean(self.river_temp, axis = 1)
        std_pr_day         = np.nanstd(self.river_temp, axis = 1)

        # Seasonal std
        summer             = np.arange(100,250)
        daynr = [date.timetuple().tm_yday for date in dates]
        std   = [(std if day in summer else -std) for std, day in zip(std_pr_day, daynr)]

        # Smooth transition
        v     = np.ones((450,))
        std   = np.convolve(std, v, 'same')/len(v)
        self.average_temp += std

        # Remove negative values for numerical stability
        lt_zero = np.where(self.average_temp < 0)[0]
        self.average_temp[lt_zero] = 0

    def nan_helper(self,data):
        return np.isnan(data), lambda z: z.nonzero()[0]