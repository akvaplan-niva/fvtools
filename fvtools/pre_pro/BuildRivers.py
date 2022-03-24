"""
--------------------------------------------------------------------------------------------------------
                            Development project - Status: Testing (beta)
--------------------------------------------------------------------------------------------------------
BuildCase - Creates all river files needed to initialize an FVCOM experiment
          - Future improvements should include stuff like user interfaces (graphical or command line)
            leading the user through any potential FVCOM setup. (connecting all the setup tools in one
            user friendly interface)
--------------------------------------------------------------------------------------------------------
"""
import sys
import os
import chardet
import numpy as np
import fvtools.grid.grid_metrics as gm
import netCDF4
import matplotlib.pyplot as plt
import pandas as pd
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.tools import num2date, date2num
from scipy.io import loadmat
from scipy.spatial import cKDTree as KDTree
from scipy import interpolate
from scipy.signal import filtfilt
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")

global version
version = 1.3

def main(start, stop, vassdrag, mesh_dict = 'M.npy', info = None, temp = False):
    """
    BuildRiver use data from the NVE and feeds all the mapped rivers leading to the ocean

    Parameters:
    ----
    start:     yyyy-mm-dd
    stop:      yyyy-mm-dd
    vassdrag:  [233, 234] etc, any array containing all ids (integers) will do.
    temp:      Swith to tell if you want to use an existing river_temperatures.npy file, or compile a new one.
               ----
                Ideal use of this is to compile a new file (temp = None) for mother-FVCOM models, and temp = 'PO10_temperatures.npy' for
                the smaller models you later nest into the mother run.

               temp = None
                - If you are running a large scale (nested to eg. NorShelf), you should compile a new *_temperatures.npy file
                  by setting temp = True.
                  After this, it will return a "casename"_temperatures.npy file to the RiverTemperatures folder
                  --> On Stokes: /data/FVCOM/Setup_Files/Rivers/Raw_Temperatures/
                  --> On Betzy:  /cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/Raw_Temperatures/

               temp = '/data/FVCOM/Setup_Files/Rivers/Raw_Temperatures/PO10_temperatures.npy' (for example)
                - Giving the temperature string will let BuildRivers know that this is the pre-compiled
                  river temperatures file you want to use.
                  --> This option was included specifically for fvcom2fvcom nested runs.

    Optional:
    ----
    mesh_dict: (M.npy by default)
    info:      Dict where all paths are stored. Change the basic settings by giving it as
               an input: info. Get the basic settings by calling BuildRivers.get_input(),
               which can then be edited and passed back to main if other paths etc. are
               needed in the given experiment.

    hes@akvaplan.niva.no
    """
    if temp is False:
        raise InputError('You must decide whether to compile a new *_temperatures.npy or use an existing one. See docstring for instructions.')

    M = FVCOM_grid(mesh_dict)

    print('----------------------------------------------------------------------------')
    print('                       BuildRivers: ' + M.info['casename'])
    print('----------------------------------------------------------------------------')

    if info is None:
        info = get_input()

    if temp is not None:
        # use a pre-compiled temperatures file
        info['rivertemp']     = temp
        info['compile river'] = False

    # Load river information
    print('Loading:')
    print('Large river positions')
    Large   = LargeRivers(info)

    print('\nSmall river positions')
    Small   = SmallRivers(info)

    print('\nRunoff data')
    Runoff  = RiverRunoff(info)

    print('\nRiver temperature from '+info['rivertemp'])
    Temp    = RiverTemperatures(info, vassdrag, M.info['casename'], start)

    # Initialize output object
    print('\nFit the input to the model:')
    Forcing = FVCOM_rivers(info, M, vassdrag)

    # Remove vassdrags that are not part of our domain
    print('- Connect rivers to nedborfelt')
    Large.connect_nedborsfelt(vassdrag)
    Small.connect_nedborsfelt(vassdrag)

    # Crop the large and small datasets to our vassdrag
    print('- Crop rivers to the vassdrag')
    Large.crop_to_vassdrag()
    Small.crop_to_vassdrag()

    # Remove the rivers that are too far away from land, and too close to the obc
    print('- Crop river to a distance from the obc')
    Small = Forcing.crop_river_to_obc(Small)
    Lagre = Forcing.crop_river_to_obc(Large)

    # Add temperatures to the small and large rivers
    print('- Add temperatures to the rivers')
    Small.add_temperature(Temp)
    Large.add_temperature(Temp)

    # --> Any other tracer that is released via the river runoff can be added here.

    # Re-distribute the runoff according to catchment area
    print('- Adjust runoff over rivers according to catchment area')
    Forcing.redistribute_runoff(Small, Large, Runoff)

    # Combine small and large rivers to the same list
    print('- Combine large and small rivers')
    Forcing.combine_small_and_large(Large, Small)

    # Trim the forcing to fit the desired start and stop time
    print('- Trim the forcing to fit with start and stop time')
    Forcing.make_time(start, stop, Runoff, Temp)

    # Find edge to connect outflow to. Distribute across rivers.
    print('- Connect the rivers to the FVCOM mesh')
    Forcing.connect_to_mesh()

    # Show the variables
    print('\nFinished, plotting the forcing')
    show_forcing(Forcing)

    # Write to netCDF, write RiverNamelist.nml
    Forcing.dump()
    Forcing.write_namelist()

def get_input():
    """
    Pre-defined paths are stored here. They are distributed to other parts of the code via main.

    iloc:        Determine if the input is given as a flux at edge or at the node
    whichrivers: 'all', 'small' , 'large'
    dRmax:       Distance from boundary without rivers
    Isplit:      Baroclinic split
    tideamp:     Tidal amplityde
    plot:        Show the results on a map
    rivertemp:   River temperature folder
    vassdrag:    File containing runoff data from vassdrag
    LargeRivers: File containing info about the large rivers in Norway
    SmallRivers: File containing info about the small rivers in Norway
    minrcoef:    Tunable parameter to determine the maximum volume of a CV we will let a river fill over a timestep
    """
    if os.getcwd().split('/')[1] == 'cluster':
        river_data_path = '/cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files'
    elif os.getcwd().split('/')[1] == 'work':
        river_data_path = '/data/FVCOM/Setup_Files'
    elif os.getcwd().split('/')[1] == 'home':
        river_data_path = '/data/FVCOM/Setup_Files'
    else:
        raise ValueError('Are you running BuildRivers on a new cluster? Could not find river_data_path.')

    info = {'iloc': 'edge',
            'whichrivers': 'all',
            'dRmax': 5000,
            'Isplit': 8,
            'tideamp': 1,
            'plot': True,
            'compile river': True,
            'rivertemp': river_data_path+'/Rivers/Raw_Temperatures/',
            'vassdrag': river_data_path+'/Rivers/riverdata_2018-2020.dat',
            'LargeRivers': river_data_path+'/Rivers/RiverData/LargeRivers_030221.mat',
            'SmallRivers': river_data_path+'/Rivers/SmallRivers_wElvID',
            'minrcoef': 0.3}
    return info

# -----------------------------------------------------------------------------------------------------
#                                        Input handling
# -----------------------------------------------------------------------------------------------------
class RiverTemperatures():
    """
    Scans the folder contatining river temperatures.
    - Compiles a yearly "typical" river temperature file
    - Looks for specific timesteps in specific vassdrag to get as in-situ temperatures as possible
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
        _start_date = datetime(int(_start[0]), int(_start[1]), int(_start[2])) - timedelta(days = 60)
        self.min_date = date2num([_start_date])[0]
        if info['compile river']:
            self.compile_temperature() # For large models spanning many vassdrags
        else:
            self.read_temperature() # For small models nested FVCOM2FVCOM

    def read_temperature(self):
        """
        Read pre-compiled temperature file
        """
        riverfile = self.info['rivertemp']
        if riverfile.split('.')[-1] == 'npy':
            data = np.load(riverfile, allow_pickle = True)
            print('- '+riverfile)
        else:
            files     = os.listdir(self.info['rivertemp'])
            riverfile = [f for f in files if f.split('.')[-1] == 'npy']
            if len(riverfile) == 1:
                data = np.load(self.info['rivertemp']+riverfile[0], allow_pickle = True)
                print('- '+self.info['rivertemp']+riverfile[0])
            else:
                raise ValueError(self.info['rivertemp'] + ' did not lead to a numpy\n '+\
                                 'file or to a folder with only one numpy file in it. Redefine your input')

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

        # Read the files
        # ----
        data, mintime, maxtime = self.read_river_files()

        # Prepare to interpolate the the river data every hour
        # ----
        self.river_time = np.arange(np.ceil(mintime), np.floor(maxtime), 1/24)

        # Loop to smooth river temperatures
        # ----
        self.raw_temp = np.nan*np.ones((len(self.river_time),len(self.files)))

        # The data can include some significant jumps (indicative of bad data), remove them
        # ----
        data = self.remove_jumps(data)

        # Fill missing days with average value from other years
        # ----
        data = self.insert_yearly_statistics(data)

        # Interpolate river temperatures to rivertime
        # ----
        self.river_to_rivertime(data)

        # Remove data from before lower cutoff data
        # ----
        self.impose_lower_cutoff()

        # Reduce noise in signal using a filter.
        # ----
        self.filter_river_temperatures()

        # Update the river temperature
        # ----
        self.river_temp = self.filtered_temp

        # Convert river time to normal dates
        # ----
        dates = num2date(self.river_time)

        # Assume that small rivers come from small lakes etc, and therefore choose the warmest temp as "average"
        # ----
        self.set_average_temp(dates)

        # Store river data and return to main
        # ----
        data = {}
        data['average temp'] = self.average_temp
        data['temp'] = self.river_temp
        data['time'] = self.river_time
        data['vassdrag'] = self.vassdrag
        print(f" - Store compiled temperature file in: {self.info['rivertemp']}/{self.casename}_temperatures.npy")
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
            print('  - '+_file)
            _data = self.read_vassdrag_temperatures(_file)
            data.append(_data)
            mintime = min(mintime, min(_data['time']))
            maxtime = max(maxtime, max(_data['time']))
            plt.plot(_data['datetime'], _data['temp'], label = _file)

        plt.title('All raw temperatures')
        plt.legend()
        return data, mintime, maxtime

    def remove_jumps(self, data):
        '''
        The data can contain suddent jumps, that's indicative of bad data so we remove them.
        '''
        # Remove significant jumps
        # ----
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
            # Get time as datetime object
            time      = num2date(_data['time'])

            # Find daynumber of year
            day_num   = np.array([t.timetuple().tm_yday for t in time])

            # Create temperature statistics of given date
            temp_stat = np.nan*np.ones((max(day_num),))
            days      = np.arange(1,max(day_num)+1)
            for day in days:
                inds = np.where(day_num == day)[0]                 # find all measured years with this day
                temp_stat[day-1] = np.nanmean(_data['temp'][inds]) # python index starts at 0, day index start at 1

            # Create an array covering the model period
            temp_full = temp_stat[day_num-1]

            # replace missing temperatures with temp_full
            nans      = np.isnan(_data['temp'])
            _data['temp'][nans] = temp_full[nans]

            # Force rivers to be >= 0
            zero_rivs = np.where(_data['temp']<0)[0]
            _data['temp'][zero_rivs] = 0

            # Store the yearly statistics
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
        # ----
        summer             = np.arange(100,250)

        # Loop over all days
        # ----
        daynr = [date.timetuple().tm_yday for date in dates]
        std   = [(std if day in summer else -std) for std, day in zip(std_pr_day, daynr)]

        # Smooth transition
        # ----
        v     = np.ones((450,))
        std   = np.convolve(std, v, 'smooth')/len(v)
        self.average_temp += std

        # Remove negative values to achieve numerical stability
        # ----
        lt_zero = np.where(self.average_temp < 0)[0]
        self.average_temp[lt_zero] = 0

    def nan_helper(self,data):
        return np.isnan(data), lambda z: z.nonzero()[0]

    def read_vassdrag_temperatures(self, _file):
        """
        Read river temperature excel files
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
                date.append(datetime(year, month, day, hour, minutes))
            except:
                day    = int(tid.split('.')[0])
                month   = int(tid.split('.')[1])
                year     = int(tid.split('.')[2].split(' ')[0])
                hour    = int(tid.split(' ')[1].split(':')[0])
                minutes = int(tid.split(' ')[1].split(':')[1])
                date.append(datetime(year, month, day, hour, minutes))

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

class LargeRivers():
    """
    Loads river data, at the moment just from mat files, but in the future?
    """
    def __init__(self, info):
        """
        Import large rivers
        """
        self.pathToRiver = info['LargeRivers']
        print('- '+self.pathToRiver)
        if self.pathToRiver[-3:] == 'mat':
            self.add_parameters(['areal','landareal','name','nedborfelt','totalareal','Vl','x','y'])
        else:
            raise NameError('.'+self.pathToRiver.split('.')[-1]+' files are not supported')

    def add_parameters(self, names):
        '''Read grid attributes from mfile and add them to FVCOM_grid object'''
        rivers = loadmat(self.pathToRiver)

        if type(names) is str:
            names=[names]

        for name in names:
            setattr(self, name, rivers[name])

    def connect_nedborsfelt(self, vassdrag_tuple):
        """
        Big rivers (nedbørsfelt til hav)
        """
        self.rivers_in_vassdrag  = np.array([ind for ind, i in enumerate(self.Vl) if i in vassdrag_tuple]).astype(int)

    def crop_to_vassdrag(self):
        """
        Removes:
        - Rivers outside of the chosen vassdrag
        - Rivers too close to the OBC
        """
        self = crop_object(self, self.rivers_in_vassdrag)

    def add_temperature(self, Temp):
        """
        Connect specific rivres to vassdrag, and mean river to the rest of the domain

        Future:
        - Investigate whether rivers can obtain temperatures as a function of distance
          from nearest temperature measurement
        """
        self.river_temp = np.zeros((len(Temp.average_temp), len(self.Vl)))
        for i, vassdrag in enumerate(self.Vl):
            if vassdrag in Temp.vassdrag:
                river = np.where(np.array(Temp.vassdrag) == vassdrag)[0][0]

                # Ps. this method will be flawed if more than 1 temperature measurement in vassdrag
                self.river_temp[:,i] = Temp.river_temp[:,river]

            else:
                self.river_temp[:,i] = Temp.average_temp


        self.river_time = Temp.river_time


    def get_area_fraction(self):
        """
        To get a reasonable estimate of the total runoff that goes through the main river
        """
        self.Vfrac = self.areal/self.landareal

    def move_river(self, Forcing):
        """
        Some times, NVE will have placed rivers at inconvenient places
        An example is "Namsen" in Trøndelag. This class lets you adjust such rivers.
        """
        i = 1
        #river = self.info['move']
        #plt.figure()
        #plt.scatter(self.x_land, self.y_land)
        #plt.scatter(self.x, self.y)
        #plt.close()

class SmallRivers():
    """
    Handle data from small rivers
    """
    def __init__(self, info):
        """
        import small rivers
        """
        self.pathToRiver = info['SmallRivers']
        print('- '+self.pathToRiver)
        self.add_parameters(['riv_ids','Vs','x2','y2'])
        self.__dict__['x'] = self.__dict__.pop('x2')
        self.__dict__['y'] = self.__dict__.pop('y2')

    def add_parameters(self, names):
        rivers = loadmat(self.pathToRiver)

        if type(names) is str:
            names=[names]

        for name in names:
            setattr(self, name, rivers[name])

    def crop_to_vassdrag(self):
        """
        Removes:
        - Rivers outside of the chosen vassdrag
        - Rivers too close to the OBC
        """
        self = crop_object(self, self.rivers_in_vassdrag)

    def add_temperature(self, Temp):
        """
        Set all small rivers to equal the average-temperature.
        We may look into finding better ways to connect small rivers to temperatures
        in the future.
        """
        self.river_temp = np.zeros((len(Temp.average_temp), len(self.Vs)))
        for i, vassdrag in enumerate(self.Vs):
            self.river_temp[:,i] = Temp.average_temp

        self.river_time = Temp.river_time


    def connect_nedborsfelt(self, vassdrag_tuple):
        """
        Big rivers (nedbørsfelt til hav)
        """
        self.rivers_in_vassdrag = np.array([ind for ind, i in enumerate(self.Vs) if i in vassdrag_tuple]).astype(int)

class RiverRunoff():
    """
    Load river runoff files, prepare to be used by the routine
    """
    def __init__(self, info):
        """
        riverdata.dat files contain all the necessary information about river runoff from
        "Vassdrag". See atlas.nve.no for more information about the whats and wheres of Vassdrag.
        """
        self.pathToRiver = info['vassdrag']
        self.info = info
        print('- '+self.pathToRiver)
        Q = self.load_data()

        # get the time format as numbers
        self.convert_dates(Q)

        # load the transport pr. "vassdragsområde"
        self.transport = Q[:,3:]

    def load_data(self):
        Q = np.loadtxt(self.info['vassdrag'])
        return Q

    def convert_dates(self,Q):
        self.dates = []
        for i in range(Q.shape[0]):
            self.dates.append(datetime(int(Q[i,0]), int(Q[i,1]), int(Q[i,2])))

class FVCOM_rivers():
    """
    Class storing data that eventually ends up as FVCOM forcing
    """
    def __init__(self, info, M, vassdrag):
        """
        Initialize?
        """
        self.info = info
        self.M = M
        self.vassdrag = vassdrag
        self.nodes, self.cells = gm.get_nbe(self.M)

        if self.info['iloc'] == 'edge':
            self.x_land  = self.M.xc[self.cells['boundary'][np.where(self.cells['id']==1)[0]]]
            self.y_land  = self.M.yc[self.cells['boundary'][np.where(self.cells['id']==1)[0]]]
            self.x_obc   = self.M.xc[self.cells['boundary'][np.where(self.cells['id']==2)[0]]]
            self.y_obc   = self.M.yc[self.cells['boundary'][np.where(self.cells['id']==2)[0]]]

        elif self.info['iloc'] == 'node':
            self.x_land  = self.M.x[self.nodes['boundary'][np.where(self.nodes['id']==1)[0]]]
            self.y_land  = self.M.y[self.nodes['boundary'][np.where(self.nodes['id']==1)[0]]]
            self.x_obc   = self.M.x[self.nodes['boundary'][np.where(self.nodes['id']==2)[0]]]
            self.y_obc   = self.M.y[self.nodes['boundary'][np.where(self.nodes['id']==2)[0]]]

        else:
            raise NameError(self.info['iloc'] + ' is not supported. Choose "edge" or "node" instead')

        print('- set-up KDTrees')
        print('  - for land')
        self.land_tree   = KDTree(np.array([self.x_land, self.y_land]).transpose())
        print('  - for obc')
        self.obc_tree    = KDTree(np.array([self.x_obc,  self.y_obc]).transpose())

        print('  - for the mesh')
        if self.info['iloc'] == 'edge':
            self.mesh_tree   = KDTree(np.array([self.M.xc, self.M.yc]).transpose())
        elif self.info['iloc'] == 'node':
            self.mesh_tree   = KDTree(np.array([self.M.x,  self.M.y]).transpose())
        self.M.calculate_tri_area()

    def redistribute_runoff(self, Small, Large, Runoff):
        """
        Figure out how much water each river should discharge
        """
        # Get the volume transport through big rivers
        Fraction_Large       = Large.areal/Large.landareal
        self.Large_Runoff    = Runoff.transport[:, Large.Vl[:,0]-1]*Fraction_Large[:,0]
        self.Large_LongName  = [nedbor.split(' ')[0] + ' - ' + name.split(' ')[0] for nedbor, name in zip(Large.nedborfelt, Large.name)]
        self.Large_ShortName = [nedbor.split(' ')[0] for nedbor in Large.nedborfelt]

        # Figure out how much area in each vassdrag is left for the small rivers, and return corresponding runoff
        Small_Runoff = []
        for vdrag in self.vassdrag:
            Fraction_Small  = 1.0 - np.sum(Fraction_Large[np.where(Large.Vl[:,0]==vdrag)[0]])
            Small_Runoff.append(Runoff.transport[:, vdrag-1]*Fraction_Small)

        Small_Runoff = np.array(Small_Runoff).T

        # Share the leftover the runoff among the small rivers
        self.Small_Runoff = np.empty((len(Small_Runoff[:,0]),0))
        for i, vdrag in enumerate(self.vassdrag):
            num_small         = len(np.where(Small.Vs == vdrag)[0])
            if num_small > 0:
                runoff_each_small = Small_Runoff[:,i]/num_small
                runoff_small_here = np.tile(runoff_each_small, (num_small,1)).T
                self.Small_Runoff = np.append(self.Small_Runoff, runoff_small_here, axis = 1)

        # Store the names
        self.Small_LongName  = [str(vassdrag[0]) + '.Z-small-'+str(i+1) for i, vassdrag in enumerate(Small.Vs)]
        self.Small_ShortName = [str(vassdrag[0]) + '.Z-s'+str(i+1) for i, vassdrag in enumerate(Small.Vs)]

    def combine_small_and_large(self, Large, Small):
        """
        Prepare the vectors that will go to the output
        """
        if self.info['whichrivers'] == 'all':
            self.xriv = np.append(Large.x, Small.x)
            self.yriv = np.append(Large.y, Small.y)
            self.transport = np.append(self.Large_Runoff, self.Small_Runoff, axis = 1)
            self.river_names = self.Large_LongName + self.Small_LongName
            self.short_names = self.Large_ShortName + self.Small_ShortName
            self.vassdrag = np.append(Large.Vl, Small.Vs)
            self.river_temp  = np.append(Large.river_temp, Small.river_temp, axis = 1)

        elif self.info['whichrivers'] == 'small':
            self.xriv = Small.x
            self.yriv = Small.y
            self.transport   = self.Small_Runoff
            self.river_names = self.Small_LongName
            self.short_names = self.Small_ShortName
            self.vassdrag = Small.Vs
            self.river_temp = Small.river_temp

        elif self.info['whichrivers'] == 'large':
            self.xriv = Large.x
            self.yriv = Large.y
            self.transport   = self.Small_Runoff
            self.river_names = self.Large_LongName
            self.short_names = self.Large_ShortName
            self.vassdrag    = Large.Vl
            self.river_temp  = Large.river_temp

        else:
            raise NameError('"' + self.info['whichrivers'] + '" is not a supported whichrives-option')


    def make_time(self, start, stop, Runoff, Temp, dt = 3/24):
        start_tuple = start.split('-')
        stop_tuple  = stop.split('-')
        self.start  = datetime(int(start_tuple[0]), int(start_tuple[1]), int(start_tuple[2]))
        self.stop   = datetime(int(stop_tuple[0]), int(stop_tuple[1]), int(stop_tuple[2]))

        # Convert to easy-to-deal-with time
        runoff_dates = np.array(netCDF4.date2num(Runoff.dates, units = 'days since 1858-11-17 00:00:00'))
        start_num    = netCDF4.date2num(self.start, units = 'days since 1858-11-17 00:00:00')
        stop_num     = netCDF4.date2num(self.stop, units = 'days since 1858-11-17 00:00:00')

        # Check if the time covers the model period
        # transport
        if stop_num > runoff_dates[-1]:
            raise ValueError(self.info['vassdrag'] + ' does not extend to the stop date')

        elif start_num < runoff_dates[0]:
            raise ValueError(self.info['vassdrag'] + ' starts after the start date')

        # temperature
        if start_num < Temp.river_time[0]:
            raise ValueError('All file(s) in '+self.info['rivertemp'] + ' starts after start date')

        elif stop_num > Temp.river_time[-1]:
            raise ValueError('All file(s) in '+self.info['rivertemp'] + ' ends before end date')

        # Prepare the output files
        self.model_time = np.arange(start_num, stop_num+dt, dt)

        # Interpolate to output structure
        self.RiverTransport = np.zeros((len(self.model_time), len(self.xriv)))
        self.RiverTemp      = np.zeros((len(self.model_time), len(self.xriv)))

        # These fields will be dumped to the model forcing file
        for i in range(len(self.xriv)):
            f_transport     = interpolate.interp1d(runoff_dates, self.transport[:,i])
            f_temperature   = interpolate.interp1d(Temp.river_time, self.river_temp[:,i])
            self.RiverTransport[:,i]  = f_transport(self.model_time)
            self.RiverTemp[:,i]       = f_temperature(self.model_time)

    def crop_river_to_obc(self, river_object):
        """
        Figure out which model point each river is closest too
        """
        # Find the nearest model cell to each river
        d, land_ind  = self.land_tree.query(np.array([river_object.x, river_object.y]).transpose())

        # Remove rivers too far away from land
        close_enough = np.where(d<=self.info['dRmax'])[1]

        # Crop the field
        river_object = crop_object(river_object, close_enough)

        # Remove rivers that are too close to the OBC
        d, obc_ind   = self.obc_tree.query(np.array([self.x_land[land_ind[0,close_enough]], \
                                                     self.y_land[land_ind[0,close_enough]]]).transpose())

        # Remove rivers too far away from land
        far_enough   = np.where(d>=self.info['dRmax'])[0]

        # Crop again
        river_object = crop_object(river_object, far_enough)

        return river_object

    def connect_to_mesh(self):
        """
        Figure out which node/cell the flux should go to.
        """
        first = True
        while True:
            d, land_loc      = self.land_tree.query(np.array([self.xriv, self.yriv]).transpose())
            if first:
                self.river_connection(land_loc)
                first = False

            d, mesh_location = self.mesh_tree.query(np.array([self.x_land[land_loc], self.y_land[land_loc]]).transpose())
            self.mesh_location = mesh_location

            print('- merge rivers that go to the same mesh point')
            self.merge_rivers()
            self.river_stability()

            # Split problematic rivers if we are troubled with such things
            bad_rivers = np.where(self.rcoef > self.info['minrcoef'])[0]
            if any(bad_rivers):
                print('- split rivers that need to be distributed over larger areas')
                self.split_problematic_river(bad_rivers)

            else:
                break

        d, land_loc      = self.land_tree.query(np.array([self.xriv, self.yriv]).transpose())
        d, mesh_location = self.mesh_tree.query(np.array([self.x_land[land_loc], self.y_land[land_loc]]).transpose())
        self.mesh_location = mesh_location

        plt.scatter(self.xriv, self.yriv, label = 'final river nodes')
        plt.legend()

    def merge_rivers(self):
        """
        Put rivers into the same structure
        """
        # Loop over each location and dump the river data fields
        self.unique_mesh = np.unique(self.mesh_location)
        transport   = np.zeros((self.RiverTransport.shape[0],len(self.unique_mesh)))
        temperature = np.zeros((self.RiverTemp.shape[0],len(self.unique_mesh)))

        # Merge rivers
        names       = []
        short_names = []
        for i, mesh_id in enumerate(self.unique_mesh):
            places            = np.where(self.mesh_location == mesh_id)[0]
            transport[:,i]   += np.sum(self.RiverTransport[:, places], axis = 1)

            # Temperatures can not be added. Divide
            temperature[:,i] += np.mean(self.RiverTemp[:, places], axis = 1)
            if len(places > 1):
                lname = ''
                sname = ''
                for place in places:
                    lname += self.river_names[place] + ', '
                    sname += self.short_names[place] + ', '
                names.append(lname[:-2]) # to avoid that ugly comma :)
                short_names.append(sname[:-2])

            else:
                names.append(self.river_names[place])
                short_names.append(self.short_names[place])

        # Update self
        self.RiverTransport = transport
        self.RiverTemp   = temperature
        self.river_names = names
        self.short_names = short_names
        if self.info['iloc'] == 'edge':
            self.xriv    = self.M.xc[self.unique_mesh]
            self.yriv    = self.M.yc[self.unique_mesh]

        elif self.info['iloc'] == 'node':
            self.xriv    = self.M.xc[self.unique_mesh]
            self.yriv    = self.M.yc[self.unique_mesh]

    def river_stability(self):
        """
        As mentioned in the FVCOM manual page 73:
        - To avoid negative salinities due to advection-related issues,
          the flux ratio can not exceed a certain threshold:

        Depth_cell > internal_delta_t * river_flux / Control_volume_area

        This routine is not exact, ie. we use mesh areas that are approximate, but not correct.
        That is, however, not a problem, since we will overestimate the severity of the river-
        problem, and thus spread the rivers over too-big areas as a countermeasure.
        """
        # The control volume area is not very easy to find, let's settle
        # for the triangle area (which is ~ 1/2 of the CV area on average)
        if self.info['iloc'] == 'edge':
            tri_area = self.M.tri_area[self.unique_mesh]

        elif self.info['iloc'] == 'node':
            print('- We will use the triangle area since it is easier to deal with!')
            cell = KDTree(np.array([self.M.xc, self.M.yc]).transpose())
            d, i = cell.query(np.array([self.xriv, self.yriv]).transpose())
            tri_area = self.M.tri_area[i]

        # Timestep
        dt_internal = min(self.M.ts)*self.info['Isplit']

        # Depth at river locations
        if self.info['iloc'] == 'edge':
            hc = np.mean(self.M.h[self.M.tri], axis = 1)
            h  = hc[self.unique_mesh]-self.info['tideamp']

        elif self.info['iloc'] == 'node':
            h  = self.M.h[self.unique_mesh]-self.info['tideamp']

        # Calculate the stability number for all the river-cells
        self.rcoef = dt_internal * self.RiverTransport.max(axis=0) / (h*tri_area)

    def split_problematic_river(self, bad_rivers):
        """
        We must make sure that a river does not completely fill a control
        volume in one timestep. (Isn't this a silly problem?)
        """
        # Find nearest land nodes, share the river with them untill rcoef should be < minrcoef
        # --> May only work for rivers that can stay relatively close...
        for river in bad_rivers:
            # Rough estimate of new rivers
            n_newland     = int(np.ceil(self.rcoef[river]/self.info['minrcoef'])+1)

            # This can put rivers on the wrong side of bays, update
            # ----
            # Loop to get two each query, store unique?
            d, this_land  = self.land_tree.query(np.array([self.xriv[river], self.yriv[river]]).transpose(), k = n_newland)

            # ----
            new_x         = np.copy(self.x_land[this_land])
            new_y         = np.copy(self.y_land[this_land])

            # Copy the stuff we are removing - should be made more general...
            transport           = np.copy(self.RiverTransport[:,river])/n_newland
            temp                = np.copy(self.RiverTemp[:,river])                # The energy should not be split
            self.RiverTemp      = np.delete(self.RiverTemp, river, 1)
            self.RiverTransport = np.delete(self.RiverTransport, river, 1)
            self.xriv           = np.delete(self.xriv,river)
            self.yriv           = np.delete(self.yriv,river)
            long_name           = self.river_names.pop(river)
            short_name          = self.short_names.pop(river)

            # Insert the split version in the new nodes
            for i in range(int(n_newland)):
                self.RiverTransport = np.append(self.RiverTransport, transport[:,None], axis = 1)
                self.RiverTemp = np.append(self.RiverTemp, temp[:,None], axis = 1)
                self.river_names.append(long_name+'-p'+str(i))
                self.short_names.append(short_name+'-p'+str(i))

            # Update river location
            self.xriv = np.append(self.xriv, new_x)
            self.yriv = np.append(self.yriv, new_y)

    def river_connection(self, land_loc):
        """
        Show how far rivers have been moved from NVE location
        - Add option to activate / deactivate rivers?
        - Add option to move rivers to other nodes?
        """
        plt.figure()
        xl = self.x_land[land_loc]
        yl = self.y_land[land_loc]
        xvec = np.array([xl, self.xriv]).transpose()
        yvec = np.array([yl, self.yriv]).transpose()
        xvec_nan = np.insert(xvec, 2, np.nan, axis = 1).ravel()
        yvec_nan = np.insert(yvec, 2, np.nan, axis = 1).ravel()
        plt.scatter(self.x_land, self.y_land, label = 'land')
        plt.scatter(xl, yl, c = 'k',          label = 'land with river')
        plt.scatter(self.xriv, self.yriv,     label = 'river location from NVE')
        plt.plot(xvec_nan, yvec_nan, c = 'r', label = 'river connected to FVCOM land')
        plt.axis('equal')

    def dump(self):
        """
        Write the riverdata.nc file for river forcing
        """
        # Initialize file
        d = netCDF4.Dataset('riverdata.nc', 'w')

        # Set dimensions
        d.createDimension('time', None)
        d.createDimension('rivers', len(self.xriv))
        d.createDimension('DateStrLen', 26)
        d.createDimension('namelen', 80)

        # Add netcdf information
        d.source      = 'Akvaplan-niva BuildRiver, version '+str(version)
        d.history     = 'Created '+ datetime.now().strftime('%Y-%m-%d at %H:%M h')+' by '+os.getlogin()
        d.description = 'River forcing (temperature and runoff) for FVCOM 4.x'

        # Create variables:
        # - time
        time = d.createVariable('time', 'single', ('time',))
        time.long_name   = 'time'
        time.units       = 'days since '+str(datetime(1858, 11, 17, 0, 0, 0))
        time.format      = 'modified julian day (MJD)'
        time.time_zone   = 'UTC'

        # - Itime
        Itime = d.createVariable('Itime', 'int32', ('time',))
        Itime.long_name   = 'integer days'
        Itime.units       = 'days since '+str(datetime(1858, 11, 17, 0, 0, 0))
        Itime.format      = 'modified julian day (MJD)'
        Itime.time_zone   = 'UTC'

        # - Itime2
        Itime2           = d.createVariable('Itime2', 'int32', ('time',))
        Itime2.long_name = 'integer milliseconds'
        Itime2.units     = 'msec since 00:00:00'
        Itime2.time_zone = 'UTC'

        # - river_flux
        flux             = d.createVariable('river_flux', 'single', ('time','rivers'))
        flux.long_name   = 'river runoff volume flux, m**-3 s**-1'
        flux.units       = 'm^3s^-1'

        # - river_temp
        temp             = d.createVariable('river_temp', 'single', ('time','rivers'))
        temp.long_name   = 'river runoff temperature'
        temp.units       = 'Celsius'

        # - river_salt
        salt             = d.createVariable('river_salt', 'single', ('time','rivers'))
        salt.long_name   = 'river runoff salinity'
        salt.units       = 'PSU'

        # - river_names
        names            = d.createVariable('river_names', 'S1', ('rivers', 'namelen'))

        # Dump data:
        salt[:]   = np.zeros(self.RiverTemp.shape)
        temp[:]   = self.RiverTemp
        flux[:]   = self.RiverTransport
        time[:]   = self.model_time
        Itime[:]  = np.floor(self.model_time)
        Itime2[:] = np.round((self.model_time - np.floor(self.model_time)) * 60 * 60 * 1000, decimals = 0)*24

        # Dump river names
        # --> Make sure that each rivername has 80 character
        _names = []
        names._Encoding = 'ascii'
        for i, name in enumerate(self.river_names):
            if len(name) > 80:
                this_name = name[:80]
            else:
                this_name  = name + (80-len(name))*' '
            this_name = self.fix_nordic(this_name)
            names[i,:] = np.array(this_name, dtype = 'S80')

        d.close()

    def fix_nordic(self, this_name):
        this_name.replace('å','a')
        this_name.replace('Å','A')
        this_name.replace('ø','o')
        this_name.replace('Ø','O')
        this_name.replace('æ','e')
        this_name.replace('Æ','E')
        return this_name

    def write_namelist(self, namelist = 'RiverNamelist.nml', riverfile = 'riverdata.nc'):
        """
        Write a namelist to accompany the netCDF file
        """
        VQDIST = -np.diff(self.M.siglev[0,:])
        f = open(namelist, 'w+')
        for i, river in enumerate(self.river_names):
            river = self.fix_nordic(river)
            f.write(' &NML_RIVER\n')
            f.write(" RIVER_NAME = '" + river +"'\n")
            f.write(" RIVER_FILE = '" + riverfile + "'\n")
            f.write(' RIVER_GRID_LOCATION = ' + str(self.mesh_location[i]+1) + '\n')
            vertical_dist = np.array2string(np.round(VQDIST,6), separator = ' ', edgeitems = 6,
                                            precision = 5, floatmode = 'fixed').replace('\n',' ')[1:-1]
            f.write(' RIVER_VERTICAL_DISTRIBUTION = ' + vertical_dist + '\n')
            f.write('/\n')
        f.close()


# Crop the fields in an object to only cover indices
# ----
def crop_object(obj, indices):
    keys = obj.__dict__.keys()
    for key in keys:
        var = getattr(obj,key)
        if key == 'rivers_in_vassdrag':
            continue
        if type(var) == str:
            continue
        setattr(obj, key, var[indices])
    return obj


# Show what we will write to the riverdata forcing
# ----
def show_forcing(obj):
    """
    Simple figures to see that the routine got the basics right
    """
    plt.figure()
    plt.scatter(obj.x_land, obj.y_land, c = 'g', label = 'land nodes')
    plt.scatter(obj.xriv, obj.yriv, np.mean(obj.RiverTransport, axis = 0), \
                c = np.mean(obj.RiverTransport, axis = 0))
    plt.title('Average transport')
    plt.axis('equal')
    plt.colorbar(label = 'qubic metres / second')
    plt.show(block = False)

    plt.figure()
    plt.scatter(obj.x_land, obj.y_land, c = 'g', label = 'land nodes')
    plt.scatter(obj.xriv, obj.yriv, obj.RiverTemp.max(axis = 0), \
                c = obj.RiverTemp.max(axis = 0), cmap = 'inferno')
    plt.title('Max temperature in model period')
    plt.axis('equal')
    plt.colorbar(label = 'degrees celcius')
    plt.show(block = False)

class InputError(Exception): pass
