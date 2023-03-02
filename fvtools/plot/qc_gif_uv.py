import sys
import os
import numpy as np
import cmocean as cmo
import matplotlib.pyplot as plt
import progressbar as pb
import matplotlib.animation as manimation

from fvtools.plot.fvcom_streamlines import streamlines
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.tools import Filelist, num2date
from fvtools.plot.geoplot import geoplot
from netCDF4 import Dataset, date2num
from datetime import datetime

# ---------------------------------------------------------------------------------------
#             Interpolate velocity data to depths of interest, create
#             a movie showing velocity maxima/minima and streamlines
# ---------------------------------------------------------------------------------------
def main(filelist = None,
         folder = None,
         sigma  = None,
         z      = None,
         xlim   = None,
         ylim   = None,
         start  = None,
         stop   = None,
         vlev   = None,
         hres   = None,
         mname  = None,
         verbose = False,
         fps    = 12):
    '''
    Interpolate velocity data to depths of interest, create
    a movie showing velocity maxima/minima and streamlines

    Mandatory input:
    ----
    - filelist or folder
    - sigma or z: sigma layer OR z-level to plot
                --> if not specified, we will use ua, va instead

    Optional input:
    ----
    - xlim, ylim: limited area
    - hres:       distance between streamline seeding points
    - start:      start limit
    - stop:       stop limit
    - fps:        framerate
    - verbose:    the routine will tell you where it is
    '''
    # To remove one input:
    if sigma is None and z is None:
        avg = True
    else:
        avg = False
    
    # Initialize the filelist
    fl  = parse_input(folder, filelist, start, stop)

    # Establish a grid object
    M   = FVCOM_grid(fl.path[0], verbose = verbose)

    # Check if we can use a cropped version of the grid
    if xlim is not None and ylim is not None:
        M.subgrid(xlim, ylim)

    # Establish the streamline maker
    stream         = streamlines(M, color = 'w', verbose = verbose)
    stream.verbose = verbose
        
    # Inititalize movie maker
    UV          = UVmov(M, fl, stream, xlim, ylim, hres, avg, vlev)
    UV.sig      = sigma; UV.z = z

    # Get the movie writer, prepare the show
    MovieWriter, codec = get_animator()

    # Inititalize the progressbar
    UV.progress = progress(len(fl.path))
    anim        = manimation.FuncAnimation(UV.fig,
                                           UV.animate,
                                           frames           = len(UV.fl.path),
                                           save_count       = len(UV.fl.path),
                                           repeat           = False,
                                           blit             = False,
                                           cache_frame_data = False)
        
    # Set framerate, write the movie
    writer = MovieWriter(fps = fps)

    # Choose a savaname, save the animation, close progressbar and return.
    if mname is None:
        savename = f'velocities.{codec}'
    else:
        savename = f'{mname}.{codec}'
    anim.save(savename, writer = writer)
    UV.progress.bar.finish()

def get_animator():
    '''
    Check which animator is available, get going
    '''
    avail          = manimation.writers.list()
    if 'ffmpeg' in avail:
        FuncAnimation = manimation.writers['ffmpeg']
        codec      = 'mp4'
       
    elif 'pillow' in avail:
        FuncAnimation = manimation.writers['pillow']
        codec      = 'gif'

    elif 'imagemagick' in avail:
        FuncAnimation = manimation.writers['imagemagic']
        codec      = 'gif'

    elif 'html' in avail:
        FuncAnimation = manimation.writers['html']
        codec      = 'html'

    else:
        raise ValueError('None of the standard animators are available')
    
    print(f'- This movie will be written as a {codec} file')
    return FuncAnimation, codec
    
class progress:
    '''
    Add a progressbar to the loop
    '''
    def __init__(self, frames):
        '''
        pre-define number of frames etc.
        '''
        self.bar = pb.ProgressBar(widgets = ['- streamline movie: ', pb.Percentage(), pb.Bar(), pb.ETA()], maxval = frames)
        self.bar.start()

class UVmov():
    '''
    Containts variables and procedures to create a velocity movie with
    streamplots and stuff.
    '''
    def __init__(self, M, fl, stream, xlim, ylim, hres, avg, vlev, verbose = True):
        '''
        Initialize grid, georeferenced image etc. for the movie maker
        '''
        # Store input
        self.M    = M;    self.fl   = fl;  self.stream = stream
        self.xlim = xlim; self.ylim = ylim
        self.avg  = avg
        self.vlev = vlev

        # Prepare streamlines
        self.stream.initialize_positions(xlim, ylim, hres)

        # Prepare file search
        self.old_path = 'loremipsum'

        # Prepare georeference
        if verbose:
            print('- Initialize georeference')
        if self.xlim is not None and self.ylim is not None:
            self.gp  = geoplot(self.xlim, self.ylim)
        else:
            self.gp  = geoplot(self.M.xc, self.M.yc)
        
        # Initialize the figure
        self.fig = plt.figure(figsize = (19.2, 8.74))

    def animate(self, i):
        '''
        Write frames in sigma levels
        '''
        # Clean the frame
        plt.clf()
        
        # See if we need to open new netcdf
        self.progress.bar.update(i)
        self.get_path(i)

        # Extract data from netcdf (and compute speed)
        self.load_data(i)

        # Add georeference
        plt.imshow(self.gp.img, extent = self.gp.extent)
        
        c = plt.tricontourf(self.M.xc, self.M.yc, self.M.ctri, self.sp_fv, 
                            levels = self.vlev, cmap = 'jet', extend = 'max')

        plt.colorbar(c, label = 'm/s')
        self.stream.get_streamlines(self.ufv, self.vfv)
        
        # Title to get when/where
        self.format_figure(i)

        return c

    def interpolate_to_z(self):
        '''
        map the velocity data from sigma to z-level before plotting, replace
        existing velocitity-data structure.
        '''
        self.ufv = self.M.interpolate_to_z(self.ufv, self.z)
        self.vfv = self.M.interpolate_to_z(self.vfv, self.z)

    def format_figure(self, i):
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.yticks([])
        plt.xticks([])
        if self.avg:
            plt.title(f'avg: {self.fl.datetime[i].strftime("%m/%d, %H:%M:%S")}')

        elif self.sig is not None:
            plt.title(f'sigma = {self.sig}: {self.fl.datetime[i].strftime("%m/%d, %H:%M:%S")}')

        elif self.z is not None:
            plt.title(f'z = {self.z}: {self.fl.datetime[i].strftime("%m/%d, %H:%M:%S")}')

    def get_path(self,i):
        if self.fl.path[i] != self.old_path:
            self.d          = Dataset(self.fl.path[i])
            self.old_path   = self.fl.path[i]

    def load_data(self, i):
        if any(self.M.cropped_cells):
            if self.avg:
                self.ufv = self.d['ua'][self.fl.index[i], self.M.cropped_cells]
                self.vfv = self.d['va'][self.fl.index[i], self.M.cropped_cells]
            else:
                if self.sig is not None:
                    self.ufv = self.d['u'][self.fl.index[i], self.sig, self.M.cropped_cells]
                    self.vfv = self.d['v'][self.fl.index[i], self.sig, self.M.cropped_cells]

                elif self.z is not None:
                    self.ufv = self.d['u'][self.fl.index[i], :, self.M.cropped_cells]
                    self.vfv = self.d['v'][self.fl.index[i], :, self.M.cropped_cells]
                else:
                    raise ValueError('Neither avg, sig or z is defined - what happened???')

        else:
            if self.avg:
                self.ufv = self.d['ua'][self.fl.index[i], :]
                self.vfv = self.d['va'][self.fl.index[i], :]
            else:
                if self.sig is not None:
                    self.ufv = self.d['u'][self.fl.index[i], self.sig, :]
                    self.vfv = self.d['v'][self.fl.index[i], self.sig, :]

                elif self.z is not None:
                    self.ufv = self.d['u'][self.fl.index[i], :]
                    self.vfv = self.d['v'][self.fl.index[i], :]
                else:
                    raise ValueError('Neither avg, sig or z is defined - what happened???')
                    
        # Compute speed
        # ----
        self.sp_fv = np.sqrt(self.ufv**2+self.vfv**2)
        if self.vlev is None:
            self.vlev = np.linspace(0, np.nanmax(self.sp_fv), 20)

def parse_input(folder, filelist, start, stop):
    '''
    Return the fields the routine expects
    '''
    if filelist is None:
        if folder is not None:
            files = allFiles(folder)

        else:
            raise ValueError('You must provide either lead me to a folder or a filelist')

        # Prepare time (string to fvcom time)
        # ----
        start, stop, units = parse_time_input(files[0], start, stop)

        # Couple file to timestep
        # ----
        time, dates, List, index = fileList(files, start, stop)
        dates = num2date(time)
        start = num2date(time = time[0]).strftime('%d/%B-%Y, %H:%M:%S')
        stop  = num2date(time = time[-1]).strftime('%d/%B-%Y, %H:%M:%S')

        # Print time
        # ----
        print(f'\nStart: {start}')
        print(f'End:   {stop}\n')
        
    else:
        # Prepare the filelist
        # ----
        fl   = Filelist(filelist, start, stop)
        time = fl.time; dates = fl.datetime; List = fl.path; index = fl.index

        start = dates[0].strftime('%d/%B-%Y, %H:%M:%S')
        stop  = dates[-1].strftime('%d/%B-%Y, %H:%M:%S')

        # Print time
        # ----
        print(f'\nStart: {start}')
        print(f'End:   {stop}')

    # Mimic the filelist structure
    # ----
    fl_out          = mini_filelist()
    fl_out.time     = time
    fl_out.datetime = dates
    fl_out.path     = List
    fl_out.index    = index
    
    return fl_out

def fileList(files, start, stop):
    '''
    Take a timestep, couple it to a file
    ----
    - files: FileList
    - start: Day to start
    - stop:  Day to stop
    '''
    print('Compiling filelist:\n-----')
    time  = np.empty(0)
    List  = []
    index = []

    for this in files:
        d = Dataset(this)
        t = d['time'][:]
        indices = np.arange(len(t))
        if start is not None:
            if t.min()<start and t.max()<start:
                print(' - '+this+' is before the date range')
                continue

            else:
                inds = np.where(t>=start)[0]
                if elements(inds)>0:
                    t = t[inds]
                    indices = indices[inds]

        if stop is not None:
            if t.min()>stop:
                print(' - '+this+' is after the date range')
                break

            inds = np.where(t<=stop)[0]
            if elements(inds) > 0:
                t = t[inds]
                indices = indices[inds]
                
        print(' - '+this)
        time     = np.append(time,t)
        dates    = num2date(time = time)
        List     = List + [this]*len(t)
        index.extend(list(indices))

    return time, dates, List, index

def parse_time_input(file_in, start, stop):
    """
    Translate time input to FVCOM time
    """
    d     = Dataset(file_in)
    units = d['time'].units

    if start is not None:
        start_num = start.split('-')
        start = date2num(datetime(int(start_num[0]), 
                                  int(start_num[1]), 
                                  int(start_num[2]), 
                                  int(start_num[3])),
                                  units = units)
    if stop is not None:
        stop_num = stop.split('-')
        stop = date2num(datetime(int(stop_num[0]), 
                                 int(stop_num[1]), 
                                 int(stop_num[2]), 
                                 int(stop_num[3])),
                                 units = units)
    return start, stop, units

def allFiles(folder):
    '''
    Return list of netCDF output files in a folder
    '''
    rawlist = os.listdir(folder)
    ncfiles = [file for file in rawlist if file[-3:] == '.nc' and 'restart' not in file]
    ncfiles.sort()
    ncfiles = [folder + file for file in ncfiles]
    return ncfiles

def elements(array):
    return array.ndim and array.size

class mini_filelist():
    pass
