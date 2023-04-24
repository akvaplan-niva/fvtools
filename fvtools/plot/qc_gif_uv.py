import sys
import os
import numpy as np
import cmocean as cmo
import matplotlib.pyplot as plt
import progressbar as pb
import matplotlib.animation as manimation

from fvtools.plot.fvcom_streamlines import streamlines
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.tools import Filelist, num2date, date2num
from fvtools.plot.geoplot import geoplot
from netCDF4 import Dataset
from datetime import datetime, timezone

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
         fps    = 12,
         dpi    = 100):
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
    - vlev:       color levels to plot for the velocity-contour plot
    '''
    # Figure out wether to read u, v or ua, va
    avg = False
    if sigma is None and z is None:
        avg = True
        
    # Initialize the filelist
    fl  = parse_input(folder, filelist, start, stop)

    # Establish a grid object
    M   = FVCOM_grid(fl.path[0], verbose = verbose)

    # Check if we can use a cropped version of the grid
    if xlim is not None and ylim is not None:
        M = M.subgrid(xlim, ylim)

    # Establish the streamline maker
    stream         = streamlines(M, color = 'w', verbose = verbose)
    stream.verbose = verbose
        
    # Inititalize movie maker
    UV           = UVmov(M, fl, stream, xlim, ylim, hres, avg, vlev, dpi = dpi)
    UV.sig, UV.z = sigma, z

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
        savename = f'{M.casename}_velocities.{codec}'
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

class UVmov:
    '''
    Containts variables and procedures to create a velocity movie with
    streamplots and stuff.
    '''
    def __init__(self, M, fl, stream, xlim, ylim, hres, avg, vlev = None, verbose = True, dpi = 300):
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
        self.fig = self.make_figure(dpi = dpi)

        # Use a wet and dry scheme
        self._wetndry = True

    def make_figure(self, size = 15., dpi = 300):
        dx = self.M.x.max() - self.M.x.min()
        dy = self.M.y.max() - self.M.y.min()
        aspect_ratio = float(dy)/(float(dx)*1.2)
        return plt.figure(figsize=(size / aspect_ratio, size), dpi = dpi)

    def animate(self, i):
        '''
        Write frames in sigma levels
        '''
        # Clean the frame
        # - future: remove contour and colorbar, keep georeference? Might not be too significant performance wise though
        plt.clf()
        
        # See if we need to open new netcdf
        self.progress.bar.update(i)

        # Extract data from netcdf (and compute speed)
        self.load_data(i)

        # Add georeference
        plt.imshow(self.gp.img, extent = self.gp.extent)

        # If _all_ contour lines are nan, we simply pass (can happen in wetting drying bug scenarios)
        try:
            c = self.M.plot_contour(self.sp_fv, levels = self.vlev, cmap = 'jet', extend = 'max', show = False)
            self.stream.get_streamlines(self.ufv, self.vfv)
            self.fig.colorbar(c, ax = plt.gca(), label = r'm$s^{-1}$', shrink = 0.5)
        except:
            pass
        
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

    # This will need some revision, maybe even using FVCOM_grids load_NETCDF function
    def load_data(self, i):
        if self.avg:
            self.ufv = self.M.load_netCDF(self.fl.path[i], 'ua', self.fl.index[i])
            self.vfv = self.M.load_netCDF(self.fl.path[i], 'va', self.fl.index[i])

        else:
            self.ufv = self.M.load_netCDF(self.fl.path[i], 'u', self.fl.index[i], sig = self.sig)
            self.vfv = self.M.load_netCDF(self.fl.path[i], 'v', self.fl.index[i], sig = self.sig)
        
        # fvcom_streamlines should not be given nans, hence we set to zero
        if np.isnan(self.ufv).any():
            naninds = np.where(np.isnan(self.ufv))[0]
            self.ufv[naninds] = 0
            self.vfv[naninds] = 0
            self.sp_fv = np.sqrt(self.ufv**2+self.vfv**2)
            self.sp_fv[naninds] = np.nan
        else:
            self.sp_fv = np.sqrt(self.ufv**2+self.vfv**2)

        if self.vlev is None:
            self.vlev = np.linspace(0, np.nanmax(self.sp_fv), 20)

def parse_input(folder, filelist, start, stop):
    '''
    Return the fields the routine expects
    '''
    if filelist is None:
        assert folder is not None, 'You must provide either lead me to a folder or a filelist'
        files = allFiles(folder)

        # Input string to fvcom time, initialize Filelist-like object
        start, stop = parse_time_input(files[0], start, stop)
        fl = qc_fileList(files, start, stop)

    else:
        fl   = Filelist(filelist, start, stop)

    # Print time
    print(' ')
    print(f"Start: {fl.datetime[0].strftime('%d/%B-%Y, %H:%M:%S')}")
    print(f"End:   {fl.datetime[-1].strftime('%d/%B-%Y, %H:%M:%S')}")
    
    return fl

def qc_fileList(files, start, stop):
    '''
    Take a timestep, couple it to a file
    ----
    - files: FileList
    - start: Day to start
    - stop:  Day to stop
    '''
    print('Making filelist:\n-----')
    fl = mini_filelist()
    time, List, index  = np.empty(0), [], []

    def crop_selection(t, indices, inds):
        return t[inds], indices[inds]

    for this_file in files:
        with  Dataset(this_file) as d:
            t = date2num(num2date(Itime = d['Itime'][:], Itime2 = d['Itime2'][:]))
            indices = np.arange(len(t))
            if start is not None:
                if t.min()<start and t.max()<start:
                    print(f' - {this_file} is before the date range')
                    continue

                else:
                    inds = np.where(t>=start)[0]
                    if elements(inds)>0:
                        t, indices = crop_selection(t, indices, inds)

            if stop is not None:
                if t.min()>stop:
                    break

                inds = np.where(t<=stop)[0]
                if elements(inds)>0:
                    t, indices = crop_selection(t, indices, inds)
                    
            print(f' - {this_file}')
            time     = np.append(time, t)
            List     = List + [this_file]*len(t)
            index.extend(list(indices))

    fl.time, fl.datetime, fl.path, fl.index = time, num2date(time = time), List, index
    return fl

def parse_time_input(file_in, start, stop):
    """
    Translate time input to FVCOM time
    """
    if start is not None:
        start_num = start.split('-')
        start = date2num([datetime(int(start_num[0]), int(start_num[1]), int(start_num[2]), int(start_num[3]), tzinfo = timezone.utc)])
    if stop is not None:
        stop_num = stop.split('-')
        stop = date2num([datetime(int(stop_num[0]), int(stop_num[1]), int(stop_num[2]), int(stop_num[3]), tzinfo = timezone.utc)])
    return start, stop

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

class mini_filelist:
    pass
