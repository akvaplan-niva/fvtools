import sys
import os
import numpy as np
import cmocean as cmo
import matplotlib.pyplot as plt
import progressbar as pb
import matplotlib.tri as tri
import matplotlib.colors as mcolor
import matplotlib.animation as manimation

from netCDF4 import Dataset, date2num
from matplotlib.colors import Normalize
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.tools import Filelist, num2date
from fvtools.plot.geoplot import geoplot
from datetime import datetime
    
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
         dpi    = 100):
    '''
    Reads files in a folder, makes a movie out of them. 
    Also support reading single files (eg. forcing)
    
    Mandatory (choose one):
    ----
    folder:   folder where the files you want to read are located
    fname:    file you want to read
    filelist: "filelist.txt" file from fvcom_make_filelist.py
    
    Optional:
    ----
    var:     Field names (in output), as list.
    sigma:   Sigma layer to read (None for pure section movie)
             - 2D fields will be animated using this routine if you set sigma = 0.
    x,ylim:  Bound the movie to a smaller domain
    start:   'yyyy-mm-dd-hh' string
    stop:    'yyyy-mm-dd-hh' string
    fps:     movie framerate
    cticks:  Set colorticks
    section: Make a movie of a transect
                - a .txt file with lon, lat columns (at least 2 points)
                - True (then you can click on a map to create a section)
    section_res:  What shall the horizontal resolution of the transect be?

    hes@akvaplan.niva.no
    '''    
    # Stop if insufficient input
    # ----
    if section is None and sigma is None and z is None:
        raise ValueError('The routine needs at least one of "section", "sigma" or "z"')

    # Get the relevant files
    # ----
    time, dates, List, index, cb = parse_input(folder, fname, filelist, start, stop, sigma, var)

    # Plot surface fields
    # ----
    if sigma is not None:
        surface_movie(time, dates, List, index, var, sigma, cb, xlim, ylim, fps, cticks, mname, dpi)

    if z is not None:
        zlevel_movie(time, dates, List, index, var, z, cb, xlim, ylim, fps, cticks, mname, dpi)

    # Plot section fields
    # ----
    if section is not None:
        section_movie(time, dates, List, index, var, cb, section, section_res, fps, cticks, mname, dpi)

    print('--> Done.')

# Define writer
# --------
def get_animator():
    '''
    Check which animator is available, get going
    '''
    avail          = manimation.writers.list()
    if 'ffmpeg' in avail:
        FuncAnimation = manimation.writers['ffmpeg']
        codec      = 'mp4'
       
    elif 'imagemagick' in avail:
        FuncAnimation = manimation.writers['imagemagick']
        codec      = 'gif'

    elif 'pillow' in avail:
        FuncAnimation = manimation.writers['pillow']
        codec      = 'gif'        

    elif 'html' in avail:
        FuncAnimation = manimation.writers['html']
        codec      = 'html'

    else:
        raise ValueError('None of the standard animators are available')
    
    return FuncAnimation, codec

# ----------------------------------------------------------------------------------
#                 For plotting movies of a sigma layer surface
# ----------------------------------------------------------------------------------
def surface_movie(time, dates, List, index, var, sigma, cb, xlim, ylim, fps, cticks, mname, dpi):
    '''
    Surface movie maker
    '''
    # Dump to the movie maker
    print('\nFeeding data to the movie maker')
    mmaker = FilledAnimation(time, dates, List, index, var, cb, xlim, ylim, sigma = sigma)
    MovieWriter, codec = get_animator()

    for field in var:
        print('--> '+field+':')
        mmaker.var    = field
        widget        = [f'- Make {field} movie: ', pb.Percentage(), pb.Bar()]
        mmaker.bar    = pb.ProgressBar(widgets=widget, maxval = len(time))
        mmaker.get_cmap(field, cb, cticks)

        # Prepare figure
        fig = plt.figure(figsize = (19.2, 8.74), dpi = dpi)

        # prepare movie maker
        mmaker.bar.start()
        anim           = manimation.FuncAnimation(fig,
                                                  mmaker.contourf_animate,
                                                  frames           = len(time),
                                                  save_count       = len(time),
                                                  repeat           = False,
                                                  blit             = False,
                                                  cache_frame_data = False)

        # Set framerate, write the movie
        writer = MovieWriter(fps = fps)
        if mname is None:
            anim.save(f'out_{field}.{codec}', writer = writer)
        else:
            anim.save(f'{mname}_{field}.{codec}', writer = writer)
        plt.close('all')
        mmaker.d.close()
        mmaker.bar.finish()
    
# For plotting movies of a z-level surface
# ----
def zlevel_movie(time, dates, List, index, var, z, cb, xlim, ylim, fps, cticks, mname, dpi):
    '''
    Surface movie maker
    '''
    # Dump to the movie maker
    print('\nFeeding data to the movie maker')
    mmaker = FilledAnimation(time, dates, List, index, var, cb, xlim, ylim, z=z)
    MovieWriter, codec = get_animator()
    
    # Start the animation routines
    for field in var:
        if field in ['zeta', 'vorticity', 'pv']:
            break

        print('--> ' + field)
        mmaker.var    = field
        widget        = [f'- Make z-level {field} movie: ', pb.Percentage(), pb.Bar()]
        mmaker.bar    = pb.ProgressBar(widgets=widget, maxval = len(time))
        mmaker.get_cmap(field, cb, cticks)
        fig = plt.figure(figsize = (19.2, 8.74), dpi = dpi)
        mmaker.bar.start()
        anim           = manimation.FuncAnimation(fig,
                                                  mmaker.zlevel_animate,
                                                  frames           = len(time),
                                                  save_count       = len(time),
                                                  repeat           = False,
                                                  blit             = False,
                                                  cache_frame_data = False)

        writer = MovieWriter(fps = fps)
        if mname is None:
            anim.save(f'out_{field}_{z}m.{codec}', writer = writer)
        else:
            anim.save(f'{mname}_{field}_{z}m.{codec}', writer = writer)
        plt.close('all')
        mmaker.d.close()
        mmaker.bar.finish()

def section_movie(time, dates, List, index, var, cb, section, section_res, fps, cticks, mname, dpi):
    '''
    Plot movies from cross-sections
    '''
    print('\nCreate the section movie')
    if section_res is None:
        section_res = 30

    if section is True:
        section = None

    MovieWriter, codec = get_animator()
    
    # Find grid info
    print('- Get grid info, section points and triangulation')
    M       = FVCOM_grid(List[0], verbose = False)
    section = M.get_section_data(M.h, section_file = section, res = section_res, store_transect_img = True)
    M.fresh_section = True # To force the depth to be re-calculated for the data we will make movies of

    # Create the movie maker
    print('- Prepare the movie maker')
    mmaker     = vertical_mmaker(time, dates, List, index)
    mmaker.M   = M

    print('\nMovie maker:')
    for field in var:
        mmaker.M.fresh_section = True
        if field == 'zeta':
            continue
        mmaker.var = field

        widget     = [f'- Make {field} movie: ', pb.Percentage(), pb.Bar()]
        mmaker.bar = pb.ProgressBar(widgets=widget, maxval=len(time))
        mmaker.get_cmap(field, cb, cticks)
        fig        = plt.figure(figsize = (19.2, 8.74), dpi = dpi)
        mmaker.bar.start()
        anim       = manimation.FuncAnimation(fig,
                                              mmaker.vertical_animate,
                                              frames           = len(time),
                                              save_count       = len(time),
                                              repeat           = False,
                                              blit             = False,
                                              cache_frame_data = False)

        writer     = MovieWriter(fps = fps)
        if mname is None:
            anim.save(f'out_{field}_transect.{codec}', writer = writer)
        else:
            anim.save(f'{mname}_{field}_transect.{codec}', writer = writer)
        plt.close()
        mmaker.d.close()
        mmaker.bar.finish()
        
def allFiles(folder):
    '''
    Return list of netCDF output files in a folder
    '''
    rawlist = os.listdir(folder)
    ncfiles = [file for file in rawlist if file[-3:] == '.nc' and 'restart' not in file]
    ncfiles.sort()
    ncfiles = [folder + file for file in ncfiles]
    return ncfiles

def fileList(files, var, start, stop, sigma = None):
    '''
    Take a timestep, couple it to a file
    ----
    - files: FileList
    - var:   Field to read
    - sigma: Sigma layer to extract
    - start: Day to start
    - stop:  Day to stop
    '''
    if sigma is None:
        sigma = 0

    print('Compiling filelist')
    time  = np.empty(0)
    List  = []
    index = []

    # For contour plots
    # ----
    cb      = {}
    for field in var:
        cb[field]        = {}
        cb[field]['max'] = -100
        cb[field]['min'] =  100

    first = True
    for this in files:
        d = Dataset(this)
        if first:
            for field in var:
                try:
                    cb['units'] = d[field].units
                except:
                    cb['units'] = ' '
            first = False

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

        for field in var:
            if field in ['vorticity', 'pv']:
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
        d.close()
    return time, dates, List, index, cb

def parse_time_input(file_in, start, stop):
    """
    Translate time input to FVCOM time
    """
    d     = Dataset(file_in)
    units = d['time'].units
    d.close()

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

# Animation manager
# ----
class FilledAnimation():
    '''
    All the data needed by the 
    '''
    def __init__(self, time, dates, List, index, var, cb, xlim, ylim, sigma = None, z = None):
        '''
        Let the writer know which frames to make
        '''
        # Write input to class
        self.index = index
        self.files = List
        self.time  = time
        self.datetime = dates
        self.sigma = sigma
        self.z     = z
        self.xlim  = xlim; self.ylim = ylim

        # Prepare grid info
        self.old_path = 'loremipsum'
        self.M     = FVCOM_grid(List[0], verbose = False)

        if self.xlim is not None and self.ylim is not None:
            self.M.subgrid(self.xlim, self.ylim, full = True)

        if 'pv' in var:
            self.M.get_coriolis()
            
        print(' - Downloading georeference')
        if self.xlim is not None and self.ylim is not None:
            self.gp = geoplot(self.xlim, self.ylim)
        else:
            self.gp = geoplot(self.M.x, self.M.y)

        # Find tge (or load file) if vorticity or potential vorticity is requested
        if 'vorticity' in var or 'pv' in var:
            import grid.tge as tge
            self.T = tge.main(self.M, verbose = True)
            self.T.get_art1(verbose = False)

    def animate(self, i):
        '''
        Write frames
        '''
        self.bar.update(i)
        plt.clf()
        path = self.files[i]
        if path != self.old_path:
            self.load_d(path)

        plt.imshow(self.gp.img, extent = self.gp.extent)

        if len(self.d[self.var][:].shape) == 3:
            cont = self.M.plot_cvs(self.d[self.var][self.index[i],self.sigma,:], \
                                   cmap = cmo.cm.dense, verbose = False)
        else:
            cont = self.M.plot_cvs(self.d[self.var][self.index[i],:], \
                                   cmap = cmo.cm.dense, verbose = False)
        # Make the colorbar once
        plt.colorbar(label = self.label)

        # Title to get when/where
        plt.title(self.datetime[i].strftime('%m/%d, %H:%M:%S'))
        
        return cont

    def contourf_animate(self, i):
        '''
        Write frames of data at z level using tricontourf
        '''
        # Keep track of progress
        self.bar.update(i)
        plt.clf()
        path = self.files[i]

        # Load new netcdf dataset only when necessary
        if path != self.old_path:
            self.load_d(path)

        # Add georeference
        plt.imshow(self.gp.img, extent = self.gp.extent)

        # Load the data (A bit messy since the data we load depend on wether we are using cropped data or not)
        # ----
        if self.var in ['vorticity', 'pv']:
            field     = self.T.vorticity(self.d['ua'][self.index[i],:],
                                         self.d['va'][self.index[i],:])
            if self.var == 'pv':
                field = (self.M.f[:,0]+field)/self.M.h[:,0]
        else:
            if len(self.d[self.var].shape) == 3:
                if self.M.cropped_object:
                    field = self.d[self.var][self.index[i],self.sigma, :][self.M.cropped_nodes]
                else:
                    field = self.d[self.var][self.index[i],self.sigma, :]
            else:
                if self.M.cropped_object:
                    field = self.d[self.var][self.index[i],:][self.M.cropped_nodes]
                else:
                    field = self.d[self.var][self.index[i],:]
                
        
        # Plot the fields
        cont  = plt.tricontourf(self.M.x, self.M.y, self.M.tri, \
                                field, cmap = self.cmap, levels = self.colorticks, extend = 'both')

        # Make the colorbar once
        plt.colorbar(label = self.label)

        if self.var in ['vorticity', 'pv']:
            plt.tricontour(self.M.x, self.M.y, self.M.tri, self.M.h, 40,
                           colors = 'gray', alpha = 0.5, linewidths = 0.5)
        plt.axis('equal')

        if self.xlim is not None:
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)

        # Title to get when/where
        plt.title(self.datetime[i].strftime('%d/%B-%Y, %H:%M:%S'))

        return cont

    def zlevel_animate(self, i):
        '''
        Write frames using tricontourf
        '''
        # Keep track of progress
        self.bar.update(i)

        # Clear old figure
        plt.clf()

        # Check if we need to load next dataset
        path = self.files[i]
        if path != self.old_path:
            self.load_d(path)

        # Create georeference
        plt.imshow(self.gp.img, extent = self.gp.extent)

        # Interpolate to z-level
        if self.M.cropped_object:
            depth    = -(self.M.h[:,None]*self.M.siglay-self.d['zeta'][self.index[i], :][self.M.cropped_nodes][:,None]).T
            outdata  = self.M.interpolate_to_z(self.d[self.var][self.index[i],:][:, self.M.cropped_nodes], 
                                               self.z, depths = depth, verbose = False)

        else:
            depth    = -(self.M.h[:,None]*self.M.siglay-self.d['zeta'][self.index[i],:][:,None]).T
            outdata  = self.M.interpolate_to_z(self.d[self.var][self.index[i],:], self.z, depths = depth, verbose = False)

        if i == 0:
            self.colorticks   = np.linspace(np.nanmin(outdata), np.nanmax(outdata), 20)

        # Visualize
        cont     = plt.tricontourf(self.M.x, self.M.y, self.M.tri, \
                                   outdata, cmap = self.cmap, levels = self.colorticks, 
                                   extend = 'both', mask = np.isnan(outdata)[self.M.tri].any(axis=1))

        # Crop to mini domain if needbe
        if self.xlim is not None:
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)

        # Make the colorbar once
        plt.colorbar(label = self.label)

        # Title to get when/where
        plt.title(self.datetime[i].strftime('%m/%d, %H:%M:%S') +' at '+str(self.z)+' m depth')
        
        return cont

    def load_d(self, path):
        '''to close an existing netCDF4 Dataset and open a new one'''
        if hasattr(self, 'd'):
            try:
                self.d.close()
            except:
                pass
                
        self.d = Dataset(path)
        self.old_path = path

    def create_cmap(self, levels):
        minc = levels[0]
        maxc = levels[1]

        # Load the base-colormaps
        cmapminr   = cmo.cm.algae
        cmapmidr   = cmo.cm.dense
        cmapmaxr   = cmo.cm.matter

        # Set the colorticks
        minrange   = np.linspace(0,minc,10)
        midrange   = np.linspace(minc, maxc, 10)
        maxrange   = np.linspace(maxc, 0.3, 10)
        colorticks = np.concatenate((minrange, midrange, maxrange))
        colorticks = np.unique(colorticks)

        # Find indices
        lt0008 = (colorticks < 0.00083)
        lt02   = (colorticks >= 0.00083) & (colorticks <= 0.02)
        gt02   = (colorticks > 0.02)

        # Get the colors
        cmin = cmapminr(np.linspace(0, 1, len(colorticks[lt0008])))
        cmid = cmapmidr(np.linspace(0, 1, len(colorticks[lt02])))
        cmax = cmapmaxr(np.linspace(0, 1, len(colorticks[gt02])))
        cmap = np.concatenate((cmin, cmid, cmax))

        # Store as colormap
        self.cmap = mcolor.ListedColormap(cmap)
        self.colorticks = colorticks 

    def get_cmap(self, var, cb, cticks):
        '''
        based on loaded field, choose colormap and set clim
        '''
        if var == 'salinity':
            self.cmap = cmo.cm.haline
            if cticks is None:
                self.colorticks = np.linspace(29, 35+(35-29)/50, 50)
            else:
                self.colorticks = cticks
            self.label = 'psu'

        elif var == 'temp':
            self.cmap = cmo.cm.thermal
            if cticks is None:
                self.colorticks = np.linspace(cb[var]['min'], \
                                              cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, \
                                              50)
            else:
                self.colorticks = cticks
            self.label = '$^\circ$C'
        
        elif var == 'zeta':
            self.cmap = cmo.cm.amp
            if cticks is None:
                self.colorticks = np.linspace(cb[var]['min'], \
                                              cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, \
                                              50)
            else:
                self.colorticks = cticks
            self.label = 'm'

        elif var == 'vorticity':
            self.cmap = cmo.cm.curl
            if cticks is None:
                self.colorticks = np.linspace(-0.0001, 0.0001, 30)
            else:
                self.colorticks = cticks
            self.label = '1/s'

        elif var == 'pv':
            self.cmap = cmo.cm.curl
            if cticks is None:
                self.colorticks = np.linspace(-0.00001, 0.00001, 30)
            else:
                self.colorticks = cticks
            self.label = '1/ms'

        else:
            self.cmap = cmo.cm.turbid
            if cticks is None:
                self.colorticks = np.linspace(cb[var]['min'], \
                                              cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, \
                                              50)
            else:
                self.colorticks = cticks
            self.label = cb['units']

class vertical_mmaker():
    '''
    A class that creates cross-section movies
    '''
    def __init__(self, time, dates, List, index):
        '''
        Let the writer know which frames to make
        '''
        # Write input to class
        # ----
        self.index = index
        self.files = List
        self.time  = time
        self.datetime = dates
                  
        # Prepare grid info
        # ----
        self.old_path = 'loremipsum'

    def vertical_animate(self, i):
        '''
        Write a section movie using contourf
        '''
        self.bar.update(i)
        plt.clf()
        path = self.files[i]
        if path != self.old_path:
            self.load_d(path)

        out  = self.M.get_section_data(self.d[self.var][self.index[i],:])
        cont = plt.contourf(out['dst'], -out['h'], out['transect'], \
                            cmap = self.cmap, levels = self.colorticks, extend = 'both')
        plt.gca().invert_yaxis()
        plt.colorbar(label = self.label)
        plt.xlabel('km from start of transect')
        plt.ylabel('meter depth')
        plt.title(self.datetime[i].strftime('%m/%d, %H:%M:%S'))
        ax = plt.gca()
        ax.set_facecolor('tab:grey')

    def load_d(self, path):
        '''to close an existing netCDF4 handle and open a new one'''
        if hasattr(self, 'd'):
            try:
                self.d.close()
            except:
                pass
        self.d = Dataset(path)
        self.old_path = path

    def get_cmap(self, var, cb, cticks):
        '''
        based on loaded field, choose colormap and set clim
        '''
        if var == 'salinity':
            self.cmap = cmo.cm.haline
            if cticks is None:
                self.colorticks = np.linspace(29, 35+(35-29)/50, 50)
            else:
                self.colorticks = cticks
            self.label = 'psu'

        elif var == 'temp':
            self.cmap = cmo.cm.thermal
            if cticks is None:
                self.colorticks = np.linspace(cb[var]['min'], \
                                              cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, \
                                              50)
            else:
                self.colorticks = cticks
            self.label = '$^\circ$C'
        
        elif var == 'zeta':
            self.cmap = cmo.cm.amp
            if cticks is None:
                self.colorticks = np.linspace(cb[var]['min'], \
                                              cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, \
                                              50)
            else:
                self.colorticks = cticks
            self.label = 'm'

        elif var == 'vorticity':
            self.cmap = cmo.cm.curl
            if cticks is None:
                self.colorticks = np.linspace(-0.0001, 0.0001, 30)
            else:
                self.colorticks = cticks
            self.label = '1/s'

        elif var == 'pv':
            self.cmap = cmo.cm.curl
            if cticks is None:
                self.colorticks = np.linspace(-0.00001, 0.00001, 30)
            else:
                self.colorticks = cticks
            self.label = '1/ms'

        else:
            self.cmap = cmo.cm.turbid
            if cticks is None:
                self.colorticks = np.linspace(cb[var]['min'], \
                                              cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, \
                                              50)
            else:
                self.colorticks = cticks
            self.label = cb['units']

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
            raise ValueError('You must provide the routine one of: folder, fname or filelist')

        # Prepare time (string to fvcom time)
        # ----
        start, stop, units = parse_time_input(files[0], start, stop)

        # Couple file to timestep
        # ----
        time, dates, List, index, cb = fileList(files, var, start, stop, sigma = sigma)
        dates = num2date(time)
        start = num2date(time = time[0]).strftime('%d/%B-%Y, %H:%M:%S')
        stop  = num2date(time = time[-1]).strftime('%d/%B-%Y, %H:%M:%S')

        # Print time
        # ----
        print(f'Start: {start}')
        print(f'End:   {stop}')
        
    else:
        # Prepare the filelist
        # ----
        fl   = Filelist(filelist, start, stop)
        time = fl.time; dates = fl.datetime; List = fl.path; index = fl.index

        start = dates[0].strftime('%d/%B-%Y, %H:%M:%S')
        stop  = dates[-1].strftime('%d/%B-%Y, %H:%M:%S')

        # Print time
        # ----
        print(f'Start: {start}')
        print(f'End:   {stop}')
        
        
        # Prepare the colorbar (quick version)
        cb    = {}
        d_min = Dataset(List[0])
        d_max = Dataset(List[-1])
        for field in var:
            if field in ['vorticity','pv']:
                continue
            cb[field]         = {}
            cb[field]['max']  = max(d_min[field][:].max(), d_max[field][:].max())
            cb[field]['min']  = min(d_min[field][:].min(), d_max[field][:].min())
            if field == 'salinity':
                cb['salinity']['min'] = 29
    
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

if __name__ == '__main__':
    folder = sys.argv[1]
    main(folder)
