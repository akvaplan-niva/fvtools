"""
Create a movie in an area based on results from ROMS
"""
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import fvtools.nesting.roms_nesting_fg as rn
import matplotlib.tri as tri
import cmocean as cmo
import cartopy.crs as ccrs
import pyproj
import progressbar as pb
import matplotlib.animation as manimation
from netCDF4 import Dataset
from fvtools.grid.roms_grid import get_roms_grid, RomsDownloader
from fvtools.plot.geoplot import geoplot

def main(start, 
         stop,  
         mother, 
         FVCOM = None,
         lats = None, 
         lons = None, 
         var = ['salt','temp','zeta'],
         fps = 12,
         dpi = 100,
         mname  = 'roms',
         reference = 'epsg:32633'):
    """
    Create a movie of norshelf model results

    Parameters:
    ----
     fvcom:  M-object
    or
     lats:   [min, max]
     lons:   [min, max]

    start:  'yyyy-mm-dd-hh'
    stop:   'yyyy-mm-dd-hh'
    mother: 'H-NS', 'D-NS', 'HI-NK' or 'MET-NK'

    Optional: (default)
    ----
    variables: ['salt', 'temp', 'zeta'] by default
    frames:    None
    fps:       12 (eg. 2 s pr. day for hourly files)
    avg:       plot daily average files
    dpi:       figure resolution
    mname:     name of the movie
    """
    # Get the ROMS grid, set xlim- ylim
    # ----
    if FVCOM is None:
        projection = Proj(reference)
        xlim, ylim = projection(lon, lat)
        ROMS = get_roms_grid(mother, projection)
    else:
        projection = FVCOM.Proj
        xlim = [FVCOM.x.min(), FVCOM.x.max()]
        ylim = [FVCOM.y.min(), FVCOM.y.max()]
        ROMS = get_roms_grid(mother, FVCOM.Proj)

    # Find files to make movie from
    # ----
    print('Make filelist')
    time, path, index = rn.make_fileList(start, stop, ROMS)

    # Load the actual grid
    # ----
    print('\nPrepare the ROMS grid for data downloading and plotting')
    ROMS.load_grid(xlim, ylim, offset = 5000)

    # Prepare colorbar
    # ----
    print('- Find colorbar limits for ROMS data')
    cb = prepare_colorbar(path, var)

    # Create the moviemaker
    # ----
    MovieWriter, codec = get_animator()
    mmaker = FilledAnimation(path, index, ROMS, xlim, ylim, FVCOM)

    # Prepare the figure:
    # ----
    fig = plt.figure(figsize = (19.2, 8.74), dpi = dpi)

    # Start the animation routines
    # ---
    if len(var)>1:
        print('\nMake animations')
    else:
        print('\nMake animation')

    for field in var:
        mmaker.var = field
        mmaker.get_cmap(field, cb)
        widget        = [f'- Make {field} movie: ', pb.Percentage(), pb.Bar(), pb.ETA()]
        mmaker.bar    = pb.ProgressBar(widgets=widget, maxval = len(time))
        mmaker.bar.start()
        anim       = manimation.FuncAnimation(fig,
                                   mmaker.animate,
                                   frames           = len(time),
                                   save_count       = len(time),
                                   repeat           = False,
                                   blit             = False,
                                   cache_frame_data = False)
        writer = MovieWriter(fps = fps)
        anim.save(f'{mname}_{field}.{codec}', writer = writer)
        mmaker.bar.finish()

# Animation
# -----------------------------------------------------------------------------
# Choose codec
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

def prepare_colorbar(files, variables):
    '''
    return a colorbar
    '''
    d_start = Dataset(files[0])
    d_stop  = Dataset(files[-1])
    cb = {}
    for field in variables:
        cb[field] = {}
        try:
            cb[field]['units'] = d_start[field].units
        except:
            cb[field]['units'] = ' '
        cb[field]['min']   = np.min([d_stop[field][:,-1,:].min(), d_start[field][:,-1,:].min()])
        cb[field]['max']   = np.max([d_stop[field][:,-1,:].max(), d_start[field][:,-1,:].max()])

    # Close netcdf handles
    d_start.close()
    d_stop.close()
    return cb

class FilledAnimation(RomsDownloader):
    '''
    Animation manager for ROMS movies
    - will download data from thredds
    '''
    def __init__(self, List, index, ROMS, xlim, ylim, FVCOM):
        '''
        Let the writer know which frames to make
        '''
        # Write input to class
        self.files, self.index = List, index
        self.xlim, self.ylim = xlim, ylim
        self.ROMS, self.FVCOM = ROMS, FVCOM
        self.N4 = ROMS # hack for now, just used when finding a cropped version of the model domain
        self.gp = geoplot(xlim, ylim)

        # Prepare grid info
        self.transform = ccrs.RotatedPole(pole_longitude = 177.5, pole_latitude = 37.5)

    def animate(self, i):
        '''
        Write frames using contourf
        '''
        plt.clf()
        self.bar.update(i)
        timestep = self.read_timestep(self.index[i], self.files[i], variables=[self.var], sigma = -1)

        # Plot the raw ROMS field
        plt.imshow(self.gp.img, extent = self.gp.extent)
        cont = plt.contourf(self.ROMS.cropped_x_rho_grid, self.ROMS.cropped_y_rho_grid, getattr(timestep, self.var), 
                            cmap = self.cmap, levels = self.colorticks, extend = 'both')
        if self.FVCOM is not None:
            plt.plot(*self.FVCOM.model_boundary.xy, c='r')
        plt.axis('equal')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.colorbar(cont, label = self.label)

        # Title to get timestamp of this frame
        with Dataset(self.files[i]) as nc:
            plt.title(netCDF4.num2date(nc['ocean_time'][self.index[i]], units = nc['ocean_time'].units).strftime('%m/%d, %H:%M:%S'))
        return cont

    def get_cmap(self, var, cb):
        '''
        based on loaded field, choose colormap and set clim
        '''
        if var == 'salt':
            self.cmap       = cmo.cm.haline
            self.colorticks = np.linspace(29, 35+(35-29)/50, 50)
            self.label      = 'psu'

        elif var == 'temp':
            self.cmap       = cmo.cm.thermal
            self.colorticks = np.linspace(cb[var]['min'], \
                                          cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, \
                                          50)
            self.label = cb[var]['units']
        
        elif var == 'zeta':
            self.cmap = cmo.cm.amp
            self.colorticks = np.linspace(cb[var]['min'], \
                                          cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, \
                                          50)
            self.label = cb[var]['units']
