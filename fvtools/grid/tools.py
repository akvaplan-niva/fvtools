import sys
import os

if float(sys.version[0])<3:
    import cPickle as pickle

import datetime
import numpy as np
import netCDF4
import matplotlib.mlab as mp
from fvtools.grid import tge

def get_cstr():
    '''Returns case-string'''
    path = os.getcwd()
    return path.split(os.sep)[-1]

class Filelist():
    '''Object linking points in time to files and indices in files (model results)'''

    def __init__(self, name='fileList.txt', start_time=None, stop_time=None, time_format='FVCOM'):
        '''Load information from file. Crop if start or stop is given.'''

        fid=open(name,'r')
        self.time=np.empty(0)
        self.path=[]
        self.index=[]

        for line in fid:
            self.time=np.append(self.time, float(line.split()[0]))
            self.path.append(line.split()[1])
            self.index.append(int(line.split()[2]))

        fid.close()
        
        if time_format == 'FVCOM':
            self.time_units = [1858,11,17] # days since 1858-11-17 00:00:00
            self.time_type = 'days'

        elif time_format == 'WRF':
            self.time_units = [1948,1,1]   # days since 1948-01-01 00:00:00
            self.time_type = 'days'

        elif time_format == 'ROMS':
            self.time_units = [1970,1,1]   # seconds since 1970-01-01 00:00:00
            self.time_type = 'seconds'

        elif time_format == 'WRF_dk':
            self.time_units = [1900,1,1]   # minutes since 1900-01-01 00:00:00
            self.time_type = 'minutes'


        #self.datetime = netCDF4.num2date(self.time, units = self.time_units)
        self.datetime = num2date(time = self.time, year = self.time_units[0], month = self.time_units[1], 
                                 day = self.time_units[2], time_type = self.time_type)
        self.crop_list(start_time, stop_time)


    def crop_list(self, start_time=None, stop_time=None):
        '''Remove values oustide specified time range.'''
        
        if start_time is not None:

            year = int(start_time.split('-')[0])
            month = int(start_time.split('-')[1])
            day = int(start_time.split('-')[2])
            hour = int(start_time.split('-')[3])
            
            t1 = datetime.datetime(year, month, day, hour)
            ind = np.where(self.datetime >= t1)[0]
            self.time = self.time[ind]
            self.datetime = self.datetime[ind]
            self.path = [self.path[i] for i in list(ind)]
            self.index = [self.index[i] for i in list(ind)]

        if stop_time is not None:

            year = int(stop_time.split('-')[0])
            month = int(stop_time.split('-')[1])
            day = int(stop_time.split('-')[2])
            hour = int(stop_time.split('-')[3])
            t1 = datetime.datetime(year, month, day, hour)
            
            ind = np.where(self.datetime <= t1)[0]
            self.time = self.time[ind]
            self.datetime = self.datetime[ind]
            self.path =  [self.path[i] for i in list(ind)]
            self.index = [self.index[i] for i in list(ind)]



    def find_nearest(self, yyyy, mm, dd, HH=0):
        '''Find index of nearest fileList entry to given point in time.'''
        
        t = datetime.datetime(yyyy, mm, dd, HH)
        fvcom_time = netCDF4.date2num(t, units
                =self.time_units)
        dt = np.abs(self.time - fvcom_time)
        ind = np.argmin(dt)

        return ind

    def write2file(self, name):
        '''Write to file.'''

        fid = open(name, 'w')
        for t, p, i in zip(self.time, self.path, self.index):
            line = str(t) + '\t' + p + '\t' + str(i) + '\n'
            fid.write(line)

        fid.close()


    def unique_files(self):
        '''Find unique files (paths) in fileList.'''
        unique_files = []
        for p in self.path:
            if p not in unique_files:
                unique_files.append(p)
        return unique_files


def num2date(time = None, Itime = None, Itime2 = None, year = 1858, month = 11, day = 17, time_type = 'days', units = None):
    '''
    Convert float or integer time from fvcom output to python datetime

    FVCOM stores dates as days since 1858,11,17. Time is the difference in decimal days,
    Itime is the integer number of days and Itime2 is the milliseconds so far this day

    Input:
    ----
    time:   numpy.ma.core.MaskedArray from fvcom output (decimal days since reference time = 1858,11,17)
    
    Alternatively, (only for FVCOM):
    Itime:  numpy.ma.core.MaskedArray from fvcom output (Integer days since reference time = 1858,11,17)
    Itime2: numpy.ma.core.MaskedArray from fvcom output (Integer milliseconds since 00:00 *this* day)

    output:
    ----
    List with datetime objects

    hes@akvaplan.niva.no
    '''
    # Assuming julian day:
    # ----
    reference_time = datetime.datetime(year,month,day,0,0,0)

    # Time type:
    # ----
    if time_type == 'days':
        dt = 1.0
    elif time_type == 'minutes':
        print('Warning: this feature has not been tested yet, but will return a result. Make sure to QC it.')
        dt = 1.0/60.0
    elif time_type == 'seconds':
        print('Warning: this feature has not been tested yet, but will return a result. Make sure to QC it.')
        dt = 1.0/(60.0*60.0)

    if units is not None:
        if '1858' in units:
            dt = 1.0
        else:
            raise ValueError('Sorry, this routine just reads "units" for FVCOM output at the moment')

    # Timearray
    # ----
    if time is not None:
        try:
            raw_fvcom = [reference_time + datetime.timedelta(days = this_time) for this_time in time.astype(np.float64)*dt]
            out = np.array(raw_fvcom)
            
        except:
            out = reference_time + datetime.timedelta(days = time*dt)

    # Exclusive to FVCOM
    # ----
    elif Itime is not None and Itime2 is not None:
        days   = Itime.data.astype(np.float64) + Itime2.data.astype(np.float64)/(24*60*60*1000)
        try:
            raw_fvcom = [reference_time + datetime.timedelta(days = this_time) for this_time in days]
            out = np.array(raw_fvcom)
            
        except:
            out = reference_time + datetime.timedelta(days = days)

    else:
        raise ValueError('You need to specify either time or Itime and Itime2')

    return out

def date2num(date, reference_time = datetime.datetime(1858,11,17)):
    '''
    returns time as decimal number of days since reference
    '''
    return np.array([(d-reference_time).total_seconds()/86400 for d in date])


def load(name):
    '''Load object stored as p-file.'''
    obj = pickle.load(open(name, 'rb'))
    return obj

def coast_segments(line_segment):
    '''
    Read list of line segments, return coast segment
    '''
    node = []; n_pol = []; i = 0
    while True:
        # If we are out of boundary points
        if line_segment == []:
            break

        # Start at the start...
        s   = line_segment[0]
        
        # Append boundary index and polygon number
        node.append(s[0]); node.append(s[1])
        n_pol.append(i);   n_pol.append(i)

        # Remove the first segment from the list
        line_segment.remove(line_segment[0])
        s_old = s[1]
        
        # Connect the segments to closed polygons
        while True:
            s = next((p for p in line_segment if s_old in p), None)
            if s is not None:
                _s = s.copy()
                _s.remove(s_old)
                s_old = _s[0]
                node.append(s_old)
                n_pol.append(i)
                line_segment.remove(s)
                    
            else:
                # new polygon
                i+=1
                break
    return node, n_pol

def make_interpolation_matrices(zlevels, depth):
    '''
    Make matrices (numpy arrays) that, when multiplied with fvcom T or S matrix,
    interpolates data to a given depth.

    Is called by FVCOM_grid
    '''
    distance_to_interpolation_depth = np.abs(zlevels - depth)
    indices_of_min_distance = np.argmin(distance_to_interpolation_depth, axis=1)
    min_distance = np.min(distance_to_interpolation_depth,axis=1)
    min_distance_z_values = zlevels[range(0, zlevels.shape[0]), indices_of_min_distance]

    mask_upper = min_distance_z_values > depth
    mask_lower = min_distance_z_values < depth
    mask_exact = min_distance_z_values == depth
    mask_unequal = min_distance_z_values != depth
    mb = zlevels[:,-1] > depth
    ma = zlevels[:,-1] <= depth
    ms = zlevels[:,0] < depth

    ind1 = np.zeros(indices_of_min_distance.shape)
    ind1[mask_upper] = indices_of_min_distance[mask_upper]
    ind1[mask_lower] = indices_of_min_distance[mask_lower]-1
    ind2 = ind1 + 1

    ind1[mask_exact]=0
    ind1[mb]=0
    ind2[mask_exact]=1
    ind2[mb]=1

    ind1 = ind1.astype(int)
    ind2 = ind2.astype(int)

    r = np.array(range(0, zlevels.shape[0]))
    dz = np.zeros(len(r))
    dz = zlevels[r, ind1] - zlevels[r, ind2]

    interp_matrix = np.zeros(zlevels.shape)

    interp_matrix[mask_upper, ind1[mask_upper]] = \
    1 - ((zlevels[mask_upper, indices_of_min_distance[mask_upper]] - depth) / dz[mask_upper])

    interp_matrix[mask_upper, ind2[mask_upper]] = \
    1 - interp_matrix[mask_upper, ind1[mask_upper]]

    interp_matrix[mask_lower, ind2[mask_lower]] = \
    1 - ((depth - zlevels[mask_lower, indices_of_min_distance[mask_lower]]) / dz[mask_lower])

    interp_matrix[mask_lower, ind1[mask_lower]] = \
    1 - interp_matrix[mask_lower, ind2[mask_lower]]

    interp_matrix[mask_exact, indices_of_min_distance[mask_exact]] = 1

    interp_matrix[mb, :] = np.nan
    interp_matrix[ms, 1:] = 0
    interp_matrix[ms, 0] = 1

def mask_land_tris(x, y, tri, ctri):
    '''
    Mask triangles facing land
    '''
    # Figure out which elements connect to land
    xc = np.mean(x[tri], axis = 1)
    yc = np.mean(y[tri], axis = 1)
    NV       = tge.check_nv(tri, x, y)
    NBE      = tge.get_NBE(len(xc), len(x), NV)
    ISBCE, _ = tge.get_BOUNDARY(len(xc), len(x), NBE, NV)

    # Illegal triangles are connected to land at all three vertices
    # --> ie. where sum of ISBCE for all triangles is > 3
    all_bce   = ISBCE[:][ctri].sum(axis=1)
    mask      = all_bce == 3

    # Masked_tri:
    return ctri[mask==False]
    return interp_matrix
