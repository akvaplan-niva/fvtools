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
from netCDF4 import Dataset
from matplotlib.animation import FuncAnimation, ImageMagickWriter

def main(lats, 
         lons, 
         start, 
         stop,  
         mother, 
         var = ['salt','temp','zeta'],
         frames = None,
         fps = 12,
         avg = False):
    """
    Create a movie of norshelf model results

    Parameters:
    ----
    lats:   [min, max]
    lons:   [min, max]
    start:  'yyyy-mm-dd-hh'
    stop:   'yyyy-mm-dd-hh'
    mother: 'NS' or 'NK'

    Optional: (default)
    ----
    variables: ['salt', 'temp', 'zeta'] by default
    frames:    None
    fps:       12 (eg. 2 s pr. day)
    """
    # Find files to make movie from
    # ----
    time, path, index, cb = make_fileList(start, stop, mother, var, avg)

    # Create the moviemaker
    # ----
    mmaker = FilledAnimation(path, index, lats, lons)

    # Prepare the figure:
    # ----
    fig = plt.figure(figsize = (19.2, 8.74))

    # Start the animation routines
    # ---
    if frames is None:
        frames = len(time)

    print('Running func animation')
    for field in var:
        print(field)
        mmaker.var = field
        mmaker.get_cmap(field, cb)
        anim       = FuncAnimation(fig,
                                   mmaker.animate,
                                   frames           = frames,
                                   save_count       = frames,
                                   repeat           = False,
                                   blit             = False,
                                   cache_frame_data = False)

        print('Writing '+field+' animation')
        writer = ImageMagickWriter(fps = fps)
        anim.save('out_'+field+'.gif', writer = writer)

class ROMS_grid():
    """
    Class containing the grid we will use in this 
    """
    def __init__(self, pathToROMS, latlim, lonlim):
        self.nc     = pathToROMS
        self.name   = pathToROMS.split('/')[-1].split('.')[0]

        # Open the ROMS file 
        # ----
        ncdata      = netCDF4.Dataset(pathToROMS, 'r')

        # temp, salt and zeta
        # ----
        self.lon = ncdata.variables.get('lon_rho')[:]
        self.lat = ncdata.variables.get('lat_rho')[:]

        # ROMS fractional landmask (from pp file). Let "1" indicate ocean.
        # ----
        self.mask = ((ncdata.variables.get('mask_rho')[:]-1)*(-1)).astype(bool)
        ncdata.close()

        # Convert to UTM33 for convenience
        # ----
        UTM33W       = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
        self.x, self.y = UTM33W(self.lon, self.lat, inverse=False)

        # Crop to get the grid to your domain<
        # ----
        self.crop(latlim, lonlim)

        # Generate grid for this domain
        # ----
        self.lon  = self.lon[self.lor[0]:self.lor[1], self.lar[0]:self.lar[1]]
        self.lat  = self.lat[self.lor[0]:self.lor[1], self.lar[0]:self.lar[1]]
        self.mask = self.mask[self.lor[0]:self.lor[1], self.lar[0]:self.lar[1]]
        self.x  = self.x[self.lor[0]:self.lor[1], self.lar[0]:self.lar[1]]
        self.y  = self.y[self.lor[0]:self.lor[1], self.lar[0]:self.lar[1]]
        
    def crop(self, lats, lons):
        """
        Find indices of grid points inside specified domain
        """
        # Crop the domain
        # ----
        ind1      = np.logical_and(self.lon  >= lons[0], self.lon <= lons[1])
        ind2      = np.logical_and(self.lat  >= lats[0], self.lat <= lats[1])
        
        # Return the longitude and latitude range
        # ----
        self.lor = [min(np.where(ind1)[0]), max(np.where(ind1)[0])+1]
        self.lar = [min(np.where(ind2)[1]), max(np.where(ind2)[1])+1]

# Animation manager
# -----------------------------------------------------------------------------
class FilledAnimation():
    '''
    All the data needed by the 
    '''
    def __init__(self, List, index, latlim, lonlim):
        '''
        Let the writer know which frames to make
        '''
        # Write input to class
        # ----
        self.index = index
        self.files = List

        # Prepare grid info
        # ----
        self.old_path  = 'start up'
        self.ROMS      = ROMS_grid(self.files[0], latlim, lonlim)
        self.transform = ccrs.RotatedPole(pole_longitude = 177.5, pole_latitude = 37.5)

        # Convert limits to UTM
        # ----
        UTM33W       = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
        self.xlim, self.ylim = UTM33W(lonlim, latlim, inverse=False)

    def animate(self, i):
        '''
        Write frames using contourf
        '''
        plt.clf()
        path = self.files[i]
        if path != self.old_path:
            self.d = netCDF4.Dataset(path)
            self.old_path = path

        # Load range
        # ----
        lonr = self.ROMS.lor
        latr = self.ROMS.lar

        if i == 0 and self.var != 'salt':
            d2 = netCDF4.Dataset(self.files[-1])
            if self.var != 'zeta':
                dta1 = self.d[self.var][:,-1,lonr[0]:lonr[1], latr[0]:latr[1]]
                dta2 = d2[self.var][:,-1,lonr[0]:lonr[1], latr[0]:latr[1]]
                
            else:
                dta1 = self.d[self.var][:,lonr[0]:lonr[1], latr[0]:latr[1]]
                dta2 = d2[self.var][:,-1,lonr[0]:lonr[1], latr[0]:latr[1]]
            miv = min(dta1.min(), dta2.min())
            mav = max(dta1.max(), dta2.max())
            self.colorticks = np.linspace(miv,mav+(mav-miv)/50, 50)

        # Draw movie
        # ----
        if self.var == 'zeta':
            data = self.d[self.var][self.index[i], lonr[0]:lonr[1], latr[0]:latr[1]]
        else:
            data = self.d[self.var][self.index[i], -1, lonr[0]:lonr[1], latr[0]:latr[1]]

        cont = plt.contourf(self.ROMS.x, self.ROMS.y, data, cmap = self.cmap, \
                            levels = self.colorticks, extend = 'both')
        plt.axis('equal')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)

        # Make the colorbar once
        # ----
        plt.colorbar(cont, label = self.label)

        # Title to get when/where
        # ----
        plt.title(netCDF4.num2date(self.d['ocean_time'][self.index[i]], units = self.d['ocean_time'].units).strftime('%m/%d, %H:%M:%S'))
        
        return cont

    def get_cmap(self, var, cb):
        '''
        based on loaded field, choose colormap and set clim
        '''
        if var == 'salt':
            self.cmap       = cmo.cm.haline
            self.colorticks = np.linspace(29, 35+(35-29)/50, 50)
            self.label      = cb[var]['units']#'psu'

        elif var == 'temp':
            self.cmap       = cmo.cm.thermal
            self.colorticks = np.linspace(cb[var]['min'], \
                                          cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, \
                                          50)
            self.label = cb[var]['units']#'$^\circ$C'
        
        elif var == 'zeta':
            self.cmap = cmo.cm.amp
            self.colorticks = np.linspace(cb[var]['min'], \
                                          cb[var]['max']+(cb[var]['max']-cb[var]['min'])/50, \
                                          50)
            self.label = cb[var]['units']#'m'

def make_fileList(start_time, stop_time, mother, var, avg):
    '''
    Go through the met office thredds server and link points in time to files.
    format: yyyy-mm-dd-hh
    '''
    cb      = {}
    for field in var:
        cb[field]        = {}
        cb[field]['max'] = -100
        cb[field]['min'] =  100

    dates   = rn.prepare_dates(start_time, stop_time)
    time    = np.empty(0)
    path    = []
    index   = []
    local_file = []; thredds_file = []
    first   = True
    for date in dates:
        # See where the files are available
        # ----
        try:
            # NorKyst
            if mother == 'NK':
                thredds_file = rn.test_thredds_path(date, mother)
                file         = thredds_file

            # NorShelf
            elif mother == 'NS':
                thredds_file = rn.test_norshelf_path(date, avg)
                file         = thredds_file

            else:
                raise OSError(mother + ' is not a valid option\n Add it! :)')

        # OSErrors will occur when the thredds server is offline/the requested file is not available
        except:
            if mother == 'NS':
                print(str(date)+ " - not found")
                continue

            # If NorKyst was not available locally, check thredds
            if mother == 'NK':
                try:
                    local_file = rn.test_local_path(date, avg)
                    file       = local_file

                except:
                    print('We did not find NorKyst-800 data for ' + str(date) + 'on thredds or in the specified folders.'+\
                          '--> Run "mend_gaps.py" to fill the gaps.')
                    continue

        # We will come in problems the day before thredds lacks norkyst data since the
        # files are not stored with the same datum. We avoid that problem by storing both.
        # ----
        if any(local_file) and any(thredds_file):
            print('- checking: ' + thredds_file + '\n- checking: ' + local_file)
            d                = Dataset(thredds_file, 'r')
            dlocal           = Dataset(local_file, 'r')
                
            # Load the timevectors
            # ----
            t_thredds_roms   = netCDF4.num2date(d.variables['ocean_time'][:],units = d.variables['ocean_time'].units)
            if type(t_thredds_roms) is np.ndarray:
                ttroms = t_thredds_roms.data
            else:
                ttroms = t_thredds_roms.data

            t_thredds_fvcom  = netCDF4.date2num(ttroms, units = 'days since 1858-11-17 00:00:00')
            t_local_roms     = netCDF4.num2date(dlocal.variables['ocean_time'][:],units = dlocal.variables['ocean_time'].units)

            if type(t_local_roms) is np.ndarray:
                tlroms = t_local_roms
            else:
                tlroms = t_local_roms.data
            t_local_fvcom    = netCDF4.date2num(tlroms, units = 'days since 1858-11-17 00:00:00')
            
            # Expand the total time, path and index vectors
            # ----
            time             = np.append(time, t_thredds_fvcom)
            path             = path + [thredds_file]*len(t_thredds_fvcom)
            index.extend(list(range(len(t_thredds_fvcom))))

            time             = np.append(time, t_local_fvcom)
            path             = path + [local_file]*len(t_local_fvcom)
            index.extend(list(range(len(t_local_fvcom))))

        # if we just have thredds or a local copy
        # -----
        else:
            print('- checking: '+file)
            d        = Dataset(file)
            
            # Convert from ROMS units to datetime
            # ----
            t_roms   = netCDF4.num2date(d.variables['ocean_time'][:], units = d.variables['ocean_time'].units)

            # Convert from datetime to FVCOM units
            # ----
            if type(t_roms) is np.ndarray:
                trom = t_roms
            else:
                trom = t_roms.data

            t_fvcom  = netCDF4.date2num(trom, units = 'days since 1858-11-17 00:00:00')
            
            # Append the timesteps, paths and indices to a fileList
            # ----
            time     = np.append(time, t_fvcom)
            path     = path + [file]*len(t_fvcom)
            index.extend(list(range(len(t_fvcom))))
        
        # Get stuff for the colorbar
        # ----

        # Colorbar units
        # ----
        if first:
            for field in var:
                try:
                    cb[field]['units'] = d[field].units
                except:
                    cb[field]['units'] = ' '
            first = False

        # Colorbar limits
        # ----
        #for field in var:
        #    print('climits '+field)
        #    if len(d[field].shape)==4:
        #        if d.variables.get(field)[0, -1,:].min() < cb[field]['min']:
        #            cb[field]['min'] = d.variables.get(field)[0, -1, :].min() 
        #    
        #        if d.variables.get(field)[0, -1, :].max() > cb[field]['max']:
        #            cb[field]['max'] = d.variables.get(field)[0, -1, :].max()#

       #     else:
       #         if d.variables.get(field)[0, :].min() < cb[field]['min']:
       #             cb[field]['min'] = d.variables.get(field)[0, :].min() 
            
       #         if d.variables.get(field)[0, :].max() > cb[field]['max']:
       #             cb[field]['max'] = d.variables.get(field)[0, :].max()
        d.close()
        local_file = []; thredds_file = []; file = []

    # --------------------------------------------------------------------------------------------
    #     Remove overlap
    # --------------------------------------------------------------------------------------------
    time_no_overlap     = [time[-1]]
    path_no_overlap     = [path[-1]]
    index_no_overlap    = [index[-1]]

    for n in range(len(time)-1, 0, -1):
        if time[n-1] < time_no_overlap[0]:
            time_no_overlap.insert(0, time[n-1])
            path_no_overlap.insert(0, path[n-1])
            index_no_overlap.insert(0, index[n-1])

    return np.array(time_no_overlap), path_no_overlap, index_no_overlap, cb
