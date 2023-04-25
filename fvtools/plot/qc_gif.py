import os
import numpy as np
import cmocean as cmo
import progressbar as pb
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import matplotlib.animation as manimation
import warnings
warnings.filterwarnings("ignore")

from netCDF4 import Dataset
from matplotlib.colors import Normalize
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.tools import Filelist, num2date, date2num
from fvtools.plot.geoplot import geoplot
from datetime import datetime, timezone
from scipy.spatial import cKDTree
from functools import cached_property
    
# ----------------------------------------------------------------------------------------
#                     Create an animation from a number of files
# ----------------------------------------------------------------------------------------
def main(folder = None,
         fname  = None,
         filelist = None,
         var    = ['salinity', 'temp', 'zeta'],
         sigma  = None,
         z      = None,
         xlim   = None,
         ylim   = None,
         start  = None,
         stop   = None,
         section = None,
         section_res  = None,
         fps    = 12,
         cticks = None,
         mname  = None,
         dpi    = 100,
         reference = 'epsg:32633'):
    '''
    Reads files, creates animation.
    - from single files
    - from multiple files in a folder
    - from files referenced in a filelist
    
    Mandatory to choose one of these:
    ----
    fname:    file you want to read
    folder:   folder holding FVCOM output file
    filelist: "filelist.txt" file from fvcom_make_filelist.py
    
    At least one:
    ----
    sigma:        sigma layer to animate
    z:            z-level to animate
    section:      make a movie of a transect
                    - a .txt file with lon lat columns (at least 2 points separated by a space)
                    - True (then you can click on a map to create a section)

    Optional:
    ----
    var:          field names (in output), as list.
    xlim, ylim:   bound to a smaller domain
    start:        'yyyy-mm-dd-hh' string - first timestep in movie
    stop:         'yyyy-mm-dd-hh' string - last timestep in movie
    fps:          movie framerate ('out' by default)
    cticks:       color shading levels
    section_res:  horizontal resolution of transect (if not specified, we will use 60 points)

    Report issues/bugs to hes@akvaplan.niva.no
    '''    
    # Stop if insufficient input
    if section is None and sigma is None and z is None:
        raise ValueError('The routine needs at least one of "section", "sigma" or "z"')

    # Get the relevant files
    time, dates, List, index, cb = parse_input(folder, fname, filelist, start, stop, sigma, var)

    # Plot surface fields
    if sigma is not None:
        surface_movie(time, dates, List, index, var, sigma, cb, xlim, ylim, fps, cticks, mname, dpi, reference)

    if z is not None:
        zlevel_movie(time, dates, List, index, var, z, cb, xlim, ylim, fps, cticks, mname, dpi, reference)

    # Plot section fields
    if section is not None:
        section_movie(time, dates, List, index, var, cb, section, section_res, fps, cticks, mname, dpi, reference)

    print('--> Done.')

# Define writer
# --------
def get_animator():
    '''
    Check which animator is available, get going
    '''
    avail = manimation.writers.list()
    if 'ffmpeg' in avail:
        FuncAnimation = manimation.writers['ffmpeg']
        codec = 'mp4'
    elif 'imagemagick' in avail:
        FuncAnimation = manimation.writers['imagemagick']
        codec = 'gif'
    elif 'pillow' in avail:
        FuncAnimation = manimation.writers['pillow']
        codec = 'gif'         
    elif 'html' in avail:
        FuncAnimation = manimation.writers['html']
        codec = 'html'
    else:
        raise ValueError('None of the standard animators are available, can not make the movie')
    return FuncAnimation, codec

def write_movie(mmaker, anim, mname, field, codec, writer):
    if mname is None:
        mname = mmaker.M.casename

    anim.save(f'{mname}_{field}.{codec}', writer = writer)
    plt.close('all')
    mmaker.bar.finish()
    
# ----------------------------------------------------------------------------------------------------------------------
#                            For plotting movies of a sigma layer surface
# ----------------------------------------------------------------------------------------------------------------------
def surface_movie(time, dates, List, index, var, sigma, cb, xlim, ylim, fps, cticks, mname, dpi, reference):
    '''
    Surface movie maker
    '''
    # Dump to the movie maker
    print('\nFeeding data to the movie maker')
    mmaker = FilledAnimation(time, dates, List, index, var, cb, xlim, ylim, reference, sigma = sigma)
    MovieWriter, codec = get_animator()

    for field in var:
        mmaker.var    = field
        widget        = [f'- Make {field} movie: ', pb.Percentage(), pb.Bar(), pb.ETA()]
        mmaker.bar    = pb.ProgressBar(widgets=widget, maxval = len(time))
        mmaker.get_cmap(field, cb, cticks)

        # Prepare figure
        fig = mmaker.make_figure(dpi = dpi)

        # prepare movie maker
        mmaker.bar.start()
        anim           = manimation.FuncAnimation(fig,
                                                  mmaker.contourf_animate,
                                                  frames           = len(time),
                                                  save_count       = len(time),
                                                  repeat           = False,
                                                  blit             = False,
                                                  cache_frame_data = False)
        writer = MovieWriter(fps = fps)

        # Set framerate, write the movie
        write_movie(mmaker, anim, mname, field, codec, writer)

def zlevel_movie(time, dates, List, index, var, z, cb, xlim, ylim, fps, cticks, mname, dpi, reference):
    '''
    Surface movie maker
    '''
    # Dump to the movie maker
    print('\nFeeding data to the movie maker')
    mmaker = FilledAnimation(time, dates, List, index, var, cb, xlim, ylim, reference, z=z)
    MovieWriter, codec = get_animator()
    
    for field in var:
        if field in ['zeta', 'vorticity', 'pv']:
            break
        mmaker.var    = field
        widget        = [f'- Make z-level {field} movie: ', pb.Percentage(), pb.Bar(), pb.ETA()]
        mmaker.bar    = pb.ProgressBar(widgets=widget, maxval = len(time))
        mmaker.get_cmap(field, cb, cticks)
        mmaker.bar.start()
        fig = mmaker.make_figure(dpi = dpi)
        anim           = manimation.FuncAnimation(fig,
                                                  mmaker.zlevel_animate,
                                                  frames           = len(time),
                                                  save_count       = len(time),
                                                  repeat           = False,
                                                  blit             = False,
                                                  cache_frame_data = False)
        writer = MovieWriter(fps = fps)
        write_movie(mmaker, anim, mname, field, codec, writer)

def section_movie(time, dates, List, index, var, cb, section, section_res, fps, cticks, mname, dpi, reference):
    '''
    Plot movies from cross-sections
    - Some work to be done reducing the data we're iterating over when making the crossection (only download data in a buffer around the transect)
    '''
    print('\nCreate the section movie')
    MovieWriter, codec = get_animator()
    
    # Find grid info
    print('- Get grid info, section points and triangulation')
    M = FVCOM_grid(List[0], verbose = False, reference = reference, static_depth = True)
    if section is True:
        section = None
    M.prepare_section(section_file = section, res = section_res, store_transect_img = True)

    # Crop grid to a sausage covering the transect (so we don't need to load excessive amounts of data to memory)
    indices = []
    for i in M.cell_tree.query_ball_point(np.vstack((M.x_sec, M.y_sec)).T, r = 5000):
        indices.extend(i)

    # Temporarilly store x_sec and y_sec
    C = M.subgrid(cells=np.unique(indices))
    C.x_sec, C.y_sec = M.x_sec, M.y_sec
    section = C.get_section_data(C.h)

    # Create the movie maker
    print('- Prepare the movie maker')
    mmaker   = VerticalMaker(time, dates, List, index, ylimit = [section['h'].min()-1, 2])
    mmaker.M = C

    print('\nMovie maker:')
    for field in var:
        if field == 'zeta':
            continue
        mmaker.var = field
        widget     = [f'- Make {field} movie: ', pb.Percentage(), pb.Bar(), pb.ETA()]
        mmaker.bar = pb.ProgressBar(widgets=widget, maxval=len(time))
        mmaker.get_cmap(field, cb, cticks)
        fig   = plt.figure(figsize = (19.2, 8.74), dpi = dpi)
        mmaker.bar.start()
        anim  = manimation.FuncAnimation(fig,
                                         mmaker.vertical_animate,
                                         frames           = len(time),
                                         save_count       = len(time),
                                         repeat           = False,
                                         blit             = False,
                                         cache_frame_data = False)
        writer = MovieWriter(fps = fps)
        write_movie(mmaker, anim, mname, field, codec, writer)
        
def allFiles(folder):
    '''
    Return list of netCDF output files in a folder
    '''
    rawlist = os.listdir(folder)
    ncfiles = [file for file in rawlist if file[-3:] == '.nc' and 'restart' not in file]
    ncfiles.sort()
    ncfiles = [folder + file for file in ncfiles]
    return ncfiles

# Still some cleaning upping to do here
def qc_fileList(files, var, start, stop, sigma = None):
    '''
    Take a timestep, couple it to a file
    - files: FileList
    - var:   Field to read
    - sigma: Sigma layer to extract
    - start: time to start. format: (yyyy-mm-dd-HH)
    - stop:  time to stop.  format: (yyyy-mm-dd-HH)
    '''
    if sigma is None:
        sigma = 0

    def crop_selection(t, indices, inds):
        return t[inds], indices[inds]

    print('Compiling filelist')
    time, List, index  = np.empty(0), [], []

    # For contour plots
    cb      = {}
    for field in var:
        cb[field]        = {}
        cb[field]['max'], cb[field]['min'] = -100, 100
        if field in ['pv', 'vorticity', 'sp']:
            continue

        with Dataset(files[0]) as d:
            cb[field]['units'] = d[field].units

    for this in files:
        with Dataset(this) as d:
            t = date2num(num2date(Itime = d['Itime'][:], Itime2 = d['Itime2'][:]))
            indices = np.arange(len(t))
            if start is not None:
                if t.min()<start and t.max()<start:
                    print(f' - {this} is before the date range')
                    continue
                else:
                    inds = np.where(t>=start)[0]
                    if elements(inds)>0:
                        t, indices = crop_selection(t, indices, inds)

            if stop is not None:
                if t.min()>stop:
                    print(f' - {this} is after the date range')
                    break
                inds = np.where(t<=stop)[0]
                if elements(inds) > 0:
                    t, indices = crop_selection(t, indices, inds)

            # If within bounds
            print(f' - {this}')
            time = np.append(time, t)
            List = List + [this]*len(t)
            index.extend(list(indices))

            for field in var:
                if field in ['vorticity', 'pv', 'sp']:
                    continue

                if len(d[field].shape)==3:
                    if d.variables.get(field)[indices, sigma, :][:].min() < cb[field]['min']:
                        cb[field]['min'] = d.variables.get(field)[indices, sigma, :][:].min() 
                
                    if d.variables.get(field)[indices, sigma, :][:].max() > cb[field]['max']:
                        cb[field]['max'] = d.variables.get(field)[indices, sigma, :][:].max()
                else:
                    if d.variables.get(field)[indices, :][:].min() < cb[field]['min']:
                        cb[field]['min'] = d.variables.get(field)[indices, :][:].min() 
                
                    if d.variables.get(field)[indices, :][:].max() > cb[field]['max']:
                        cb[field]['max'] = d.variables.get(field)[indices, :][:].max()

    return time, num2date(time = time), List, index, cb

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

class AnimationFields:
    '''
    Some fields can be read directly from the netCDF file, some must be inferred.
    - It seems to be faster to get the entire array (not just those we are plotting)
    '''
    @property
    def field(self):
        '''
        Loads fields, we assume all "new" fields (ie. fabm tracers) to be 3D
        - will mask grid points that are dry
        '''
        if self.var in ['pv', 'vorticity', 'sp']:
            field = getattr(self, self.var)
        else:
            field = self.M.load_netCDF(self.files[self.i], self.var, self.index[self.i], sig = self.sigma)
        return field
    
    @property
    def vorticity(self):
        '''we load velocities for the entire grid, since the vorticity calculation at boundaries is fizzy, hence we can't necessarilly use the cropped grid'''
        with Dataset(self.files[i], 'r') as d:
            _vort = self.T.vorticity(d['ua'][self.index[self.i], :], d['va'][self.index[self.i],:])
        if self.M.cropped_cells.any():
            _vort = _vort[self.M.cropped_cells]
        return _vort
    
    @property
    def pv(self):
        return (self.M.f + self.vorticity)/self.M.h

    @property
    def sp(self):
        return np.sqrt(self.M.load_netCDF(self.files[i], 'ua', self.index[self.i])**2 + self.M.load_netCDF(self.files[i], 'va', self.index[self.i])**2)

class AnimationColorbars:
    '''
    Generic metod to retrieve colormap
    '''
    def get_cmap(self, var, cb, cticks):
        '''
        based on loaded field, choose colormap and set clim
        '''
        if var in 'salinity':
            self.cmap = cmo.cm.haline
            self.colorticks = np.linspace(29, 35+(35-29)/50, 50)
            self.label = 'psu'
            
        elif var == 'temp':
            self.cmap = cmo.cm.thermal
            self.colorticks = np.linspace(cb[var]['min'], cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, 50)
            self.label = '$^\circ$C'

        elif var == 'zeta':
            self.colorticks = np.linspace(cb[var]['min'], cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, 50)
            self.cmap = cmo.tools.crop(cmo.cm.balance, min(self.colorticks), max(self.colorticks), 0)
            self.label = 'm'

        elif var == 'vorticity':
            self.cmap = cmo.cm.curl
            self.colorticks = np.linspace(-0.0001, 0.0001, 30)
            self.label = '1/s'

        elif var == 'pv':
            self.cmap = cmo.cm.curl
            self.colorticks = np.linspace(-0.00001, 0.00001, 30)
            self.label = '1/ms'

        elif self.var == 'sp':
            self.cmap = cmo.cm.speed
            self.colorticks = np.linspace(0, 0.4, 30)
            self.label = 'm/s'

        else:
            self.cmap = cmo.cm.turbid
            self.colorticks = np.linspace(cb[var]['min'], cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, 50)
            self.label = cb[var]['units']

        if cticks is not None:
            self.colorticks = cticks

class GeoReference:
    '''
    Temporarilly here, I want to use this one in roms_movie as well
    '''
    def _get_georeference(self, x, y, reference):
        '''
        Expects that your 
        '''
        for source in ['hot', 'mapnik', 'voyager']:
            try:
                georef = geoplot(x, y, source = source, projection = reference)
                break

            except:
                georef = None
                pass
        
        return georef

    # Can actually also just use FVCOM_grids georeference method, look into that once we can expect cartopy on all machines.

class FilledAnimation(AnimationFields, AnimationColorbars, GeoReference):
    '''
    All the data needed by the 
    '''
    def __init__(self, time, dates, List, index, var, cb, xlim, ylim, reference, sigma = None, z = None):
        '''
        Let the writer know which frames to make
        '''
        # Write input to class
        self.index, self.files, self.time, self.datetime = index, List, time, dates
        self.sigma, self.z    = sigma, z
        self.xlim,  self.ylim = xlim, ylim

        # Prepare grid
        self.M = FVCOM_grid(List[0], reference = reference)

        if self.xlim is not None and self.ylim is not None:
            self.M = self.M.subgrid(self.xlim, self.ylim)

        if 'pv' in var:
            self.M.get_coriolis()
            
        print(' - Downloading georeference')
        if self.xlim is not None and self.ylim is not None:
            self.gp = self._get_georeference(self.xlim, self.ylim, reference)
        else:
            self.gp = self._get_georeference(self.M.x, self.M.y, reference)

        if self.gp is None:
            print('  - Failed to download georeference, the background will not be shaded.')

        if 'vorticity' in var or 'pv' in var:
            import grid.tge as tge
            self.T = tge.main(self.M, verbose = True)
            self.T.get_art1(verbose = False)

        self._wetndry = True

    def make_figure(self, size = 15., dpi = 300):
        dx = self.M.x.max() - self.M.x.min()
        dy = self.M.y.max() - self.M.y.min()
        aspect_ratio = float(dy)/(float(dx)*1.2)
        return plt.figure(figsize=(size / aspect_ratio, size), dpi = dpi)

    def animate(self, i):
        '''
        Write frames
        '''
        self.bar.update(i)
        self.i = i
        plt.clf()
        if self.gp is not None:
            plt.imshow(self.gp.img, extent = self.gp.extent)
        cont = self.M.plot_cvs(self.field, cmap = cmo.cm.dense, verbose = False)
        self.update_figure(title = self.datetime[i].strftime('%d/%B-%Y, %H:%M:%S') + f' in sigma = {self.sigma}')
        return cont

    def contourf_animate(self, i):
        '''
        Write frames of data at z level using tricontourf
        '''
        self.bar.update(i)
        self.i = i
        plt.clf()
        if self.gp is not None:
            plt.imshow(self.gp.img, extent = self.gp.extent)
        if self.var in ['vorticity', 'pv', 'sp']:
            plt.tricontour(self.M.x, self.M.y, self.M.tri, self.M.h, 40, colors = 'gray', alpha = 0.5, linewidths = 0.5, zorder = 10)
        cont = self.update_figure(field = self.field, title = self.datetime[i].strftime('%d/%B-%Y, %H:%M:%S'))
        return cont

    def zlevel_animate(self, i):
        '''
        Write frames using tricontourf
        '''
        self.bar.update(i)
        self.i = i
        plt.clf()
        if self.gp is not None:
            plt.imshow(self.gp.img, extent = self.gp.extent)
        outdata = self.M.interpolate_to_z(self.field, z = self.z)
        cont = self.update_figure(field = outdata, title = self.datetime[i].strftime('%m/%d, %H:%M:%S') + f' at {self.z} m depth')
        return cont

    def update_figure(self, field=None, title=None):
        '''
        Add cosmetics to the figure
        '''
        if field is not None:
            cont = self.M.plot_contour(field, show = False, cmap = self.cmap, levels = self.colorticks, extend = 'both', zorder = 5)
        if self.xlim is not None and self.ylim is not None:
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
        plt.colorbar(cont, label = self.label, shrink = 0.5)
        plt.title(title)   
        if field is not None:
            return cont

class VerticalMaker(AnimationFields, AnimationColorbars):
    '''
    A class that creates cross-section movies
    '''
    def __init__(self, time, dates, List, index, ylimit = None):
        '''
        Let the writer know which frames to make
        '''
        self.index = index
        self.files = List
        self.time  = time
        self.datetime = dates
        self.sigma = None # a bit of a hack
        self.ylimit = ylimit

    def vertical_animate(self, i):
        '''
        Write a section movie using contourf
        '''
        self.i = i
        self.bar.update(i)
        plt.clf()
        self.M.zeta = self.M.load_netCDF(self.files[i], 'zeta', self.index[i])
        out = self.M.get_section_data(self.field)
        cont = plt.contourf(out['dst'], out['h'], out['transect'], cmap = self.cmap, levels = self.colorticks, extend = 'both')

        # Settings
        if self.ylimit is not None:
            plt.ylim(self.ylimit)
        plt.colorbar(label = self.label)
        plt.xlabel('km from start of transect')
        plt.ylabel('meter depth')
        plt.title(self.datetime[i].strftime('%m/%d, %H:%M:%S'))
        ax = plt.gca()
        ax.set_facecolor('tab:grey')

def parse_input(folder, fname, filelist, start, stop, sigma, var):
    '''
    Return the fields the routine expects
    '''
    if filelist is None:
        if folder is not None:
            files = allFiles(folder)
        elif fname is not None:
            files = [fname]
        else:
            raise InputError('You must provide the routine one of: folder, fname or filelist')

        # Prepare time (string to fvcom time)
        start, stop = parse_time_input(files[0], start, stop)

        # Couple file to timestep
        time, dates, List, index, cb = qc_fileList(files, var, start, stop, sigma = sigma)

    else:
        # Prepare the filelist
        # ----
        fl = Filelist(filelist, start, stop)
        time, dates, List, index = fl.time, fl.datetime, fl.path, fl.index        
        
        # Prepare the colorbar (quick version)
        cb = {}
        d_min = Dataset(List[0])
        d_max = Dataset(List[-1])
        for field in var:
            if field in ['vorticity','pv']:
                continue
            cb[field] = {}
            cb[field]['max'] = max(d_min[field][:].max(), d_max[field][:].max())
            cb[field]['min'] = min(d_min[field][:].min(), d_max[field][:].min())
            if field == 'salinity':
                cb['salinity']['min'] = 29
            with Dataset(fl.path[0]) as d:
                cb[field]['units'] = d[field].units

    print(f"Start: {dates[0].strftime('%d/%B-%Y, %H:%M:%S')}")
    print(f"End:   {dates[-1].strftime('%d/%B-%Y, %H:%M:%S')}")
    return time, dates, List, index, cb

def elements(array):
    return array.ndim and array.size

class PiecewiseNorm(Normalize):
    ''' piecewise normalization of colormap'''
    def __init__(self, levels, clip = False):
        self._levels = np.sort(levels)
        self._normed = np.linspace(0,1,len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip = None):
        # Linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))

class InputError(Exception): pass