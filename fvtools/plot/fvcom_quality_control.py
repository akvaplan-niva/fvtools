import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
from netCDF4 import Dataset
import netCDF4
from pyproj import Proj
from scipy.io import loadmat
import cmocean
import cv2
import sys

# Experimental:

from plot.new_geoplot import geoplot

def main(save_name,
         folder, 
         experiment  = None,
         file_range  = 'all',
         xlim        = None,
         ylim        = None,
         levels      = None, 
         save_folder = os.getcwd(), 
         section_file= None,
         types       = 'S,T,zeta',
         georef      = None):
    '''
    Make surface temp/salinity movies. Vertical section is made if section file is given.
    
    Input arguments:
    ----
    save_name:    String with name to use to store the movies 
    folder:       Path to data folder

    Optional input arguments:
    ----
    file_range:   List witn the number of first and last file to include. E.g. [5, 10]. 
                  Default: 'all'
    save_folder:  Path to directory where movies shoul be saved to file.
                  Default: Current working directory
    section_file: Path to file containing positions of points to be included in the 
                  vertical section. Default: None
    types:        Dictate which data shall be plotted. Default: types = 'S,T,zeta'.
                  If empty, you will just plot the section.
    georef:       Fits an openstreetmap georeferenced image to the video if set True.
                  Default: False
    '''

    # Load grid, setup metrics and find the area to show
    # -------------------------------------------
    Mobj     = FVCOM_grid(folder+get_gridfile(folder, experiment=experiment))
    infocus  = get_focus(Mobj, xlim, ylim)

    # Arange files that are to be used in a list
    # -------------------------------------------
    fileList = make_fileList(folder, file_range=file_range, experiment=experiment, focus=infocus)

    # Prepare the georeference
    # ----
    if georef is not None:
        georef, Mobj = prepare_georef(Mobj, focus = infocus)

    if types.find('T')>=0:
        print('\nMaking temperature movie...')
        field_movie(save_name, Mobj, fileList, 'temp', georef, layer=0, save_folder=save_folder, xlim=xlim, ylim=ylim)
        print('Finished')

    if types.find('S')>=0:
        print('\nMaking salinity movie...')
        field_movie(save_name, Mobj, fileList, 'salinity', georef, layer=0, save_folder=save_folder, xlim=xlim, ylim=ylim)
        print('Finished')

    if types.find('zeta')>=0:
        print('\nMaking zeta movie...')
        field_movie(save_name, Mobj, fileList, 'zeta', georef, layer=0, save_folder=save_folder, xlim=xlim, ylim=ylim)
        print('\nFinished \n')

    if section_file is not None:
        if geoplot is not None:
            # Since geoplot changes the reference...
            Mobj = FVCOM_grid(folder+get_gridfile(folder, experiment=experiment))

        # setup indices:
        # ---------------------------------------------------------
        ind_section = prepare_section(Mobj,section_file,save_folder)

        print('Making section movies:\n')
        print('Temperature...')
        x,y,z,data  = prepare_vdata(Mobj,ind_section,fileList,'temp')
        plot_section(save_name, Mobj, x, y, z, data, fileList, ind_section, 'temp', save_folder=save_folder)

        print('Finished\n\n' + 'Salinity')
        x,y,z,data  = prepare_vdata(Mobj,ind_section,fileList,'salinity')
        plot_section(save_name, Mobj, x, y, z, data, fileList, ind_section, 'salinity', save_folder=save_folder)
        print('Finished. \n')

# Grid and fileList stuff
# ---------------------------
def get_focus(Mobj, xlim, ylim):
    # Setup x and y limits, find focus (won't be good for southern hemisphere)
    # -------------------------------------------
    if xlim is not None:
        if max(xlim)<=90:
            print('Detected latlon format')
            xlim, ylim = Mobj.Proj(xlim,ylim)
        else:
            print('Detected utm-format')

    if xlim is not None:
        infocus = Mobj.isinside(xlim,ylim)[:,0]
    else:
        infocus  = np.ones(len(Mobj.x),dtype=bool)
    
    return infocus

def prepare_georef(Mobj, focus = None):
    '''
    Download georeferenced image tiles and redefine the x- and y- axis of Mobj to suit the georeferenced coordinates
    '''

    # Inform the user
    print('\nDownloading georeferenced image tiles')
    
    # Prepare the tiles
    gp     = geoplot(np.array([min(Mobj.x[focus])-2000, max(Mobj.x[focus])+2000]),\
                     np.array([min(Mobj.y[focus])-2000, max(Mobj.y[focus])+2000]))
    gp.prepare_field(Mobj.x,Mobj.y)

    # Update the coordinates for plotting
    Mobj.x = gp.x
    Mobj.y = gp.y
    georef = gp
    return georef, Mobj

def get_gridfile(folder, experiment=None):
    '''
    Get the grid file from the folder you will make a filelist from
    '''

    # All files in folder
    allFiles = os.listdir(folder)
    
    # Only keep nc-files
    ncfiles = [f for f in allFiles if pl.Path(f).suffix=='.nc']
    
    # Only keep experiment named files
    if experiment:
        ncfiles = [f for f in ncfiles if f.split('_')[0]==experiment]

    # Remove restart files
    ncfiles = [f for f in ncfiles if ('restart' not in f)]
    ncfiles.sort()

    return ncfiles[0]

def make_fileList(folder, file_range='all',experiment=None,focus=None):
    fl = FileList()

    # All files in folder
    allFiles = os.listdir(folder)
    
    # Only keep nc-files
    ncfiles = [f for f in allFiles if pl.Path(f).suffix=='.nc']
    
    # Only keep experiment named files
    if experiment:
        ncfiles = [f for f in ncfiles if f.split('_')[0]==experiment]

    # Remove restart files
    ncfiles = [f for f in ncfiles if ('restart' not in f)]
    ncfiles.sort()

    # Select range if given
    if file_range is not 'all':
        file_range = np.array(file_range)
        nc_file_numbers = [int(n[-7:-3]) for n in ncfiles]
        nc_file_numbers = np.array(nc_file_numbers)

        in_range = (nc_file_numbers>=file_range[0]) & (nc_file_numbers<=file_range[1])
        ncfiles = [n for n, ind in zip(ncfiles, in_range) if ind] 

    # Loop through files, read time, and max/min sal/temp/zeta.
    fl.S_min    =  100
    fl.S_max    = -100
    fl.T_min    =  100
    fl.T_max    = -100
    fl.zeta_min =  100
    fl.zeta_max = -100

    for ncfile in ncfiles:
        print(ncfile)
        nc        = Dataset(os.path.join(folder, ncfile), 'r')
        t         = nc.variables['time'][:]
        fl.fvtime = np.append(fl.fvtime, t)
        fl.path.extend([os.path.join(folder, ncfile)] * len(t))
        fl.index.extend(range(len(t)))
    
        if nc.variables.get('salinity')[:, 0, :][:,focus].min() < fl.S_min:
            fl.S_min = nc.variables.get('salinity')[:, 0, :][:,focus].min() 
        if nc.variables.get('salinity')[:, 0, :][:,focus].max() > fl.S_max:
            fl.S_max = nc.variables.get('salinity')[:, 0,:][:,focus].max()
        if nc.variables.get('temp')[:, 0, :][:,focus].min() < fl.T_min:
            fl.T_min = nc.variables.get('temp')[:, 0, :][:,focus].min() 
        if nc.variables.get('temp')[:, 0, :][:,focus].max() > fl.T_max:
            fl.T_max = nc.variables.get('temp')[:, 0, :][:,focus].max()

        if nc.variables.get('zeta')[:].min() < fl.zeta_min:
            fl.zeta_min = nc.variables.get('zeta')[:].min() 
        if nc.variables.get('zeta')[:].max() > fl.zeta_max:
            fl.zeta_max = nc.variables.get('zeta')[:].max()
 
        nc.close()
        fl.fvtime2datetime()
     
    return fl

class FileList():
      '''       '''
      def __init__(self):
          self.path = []
          self.fvtime = np.empty(0)
          self.index = []
    

      def fvtime2datetime(self):
          self.datetime = netCDF4.num2date(self.fvtime, units='days since 1858-11-17 00:00:00')


def summary_stats(fileList):
    '''Loop through results in fileList and get summary stats.'''
    pass

def prepare_section(Mobj,section_file,save_folder):
    '''
    Create the section based on points given in the namelist
    '''
    # ------------------- Find indices of section ------------------------- 
    # load section positions. If there are only two, 
    # define more along the straight line between the points. 

    lon, lat = np.loadtxt(section_file, delimiter=',', unpack=True)

    if len(lon) == 2:
        x, y = Mobj.Proj(lon, lat)
        if x[0] > x[1]: # Ensure that the westernmost point in the section is first.
            x1 = x[1]
            x2 = x[0]
            y1 = y[1]
            y2 = y[0]
        else:
            x1 = x[0]
            x2 = x[1]
            y1 = y[0]
            y2 = y[1]

        # Find the equation for the straight line between the points.
        a = (y2-y1) / (x2-x1) # slope
        b = y1 - a*x1 # intersection 

        x_section = np.arange(np.round(x1), np.round(x2), ds)
        y_section = a*x_section + b
        lon, lat = Mobj.Proj(x_section, y_section, inverse=True)

    else:
        x_section, y_section = Mobj.Proj(lon, lat)

    # Find nearest node to those extracted
    # ----------------------------------------------------------------
    ind_section, ri = np.unique(Mobj.find_nearest(x_section, y_section), return_index=True)
    ind_section     = ind_section[np.argsort(ri)]
    
    # Section can consist of segments, find all nodes in the segment
    # ----------------------------------------------------------------
    new_ind_section = []
    for i in range(len(ind_section)-1):
        secind_back  = ind_section[i]   # section index at segment start
        secind_front = ind_section[i+1] # section index at segment end
        myind        = secind_back      # current segment index

        while myind != secind_front:
            new_ind_section.append(myind)
            nearind  = Mobj.nbsn[myind,np.where(Mobj.nbsn[myind,:]>=0)][0] # all nodes surrounding node
            nearind  = nearind.data
            dst      = np.sqrt((Mobj.x[nearind]-Mobj.x[secind_front])**2+(Mobj.y[nearind]-Mobj.y[secind_front])**2)
            myind    = nearind[np.where(dst==dst.min())[0][0]] # The node nearest next segment end
            
    # Append the last index
    # ---------------------------------------------------------------
    new_ind_section.append(myind)
    ind_section = np.array(new_ind_section)

    # Plot section
    # ---------------------------------------------------------------
    print('Plotting the section')
    Mobj.plot_grid()
    plt.plot(Mobj.x[ind_section], Mobj.y[ind_section], 'r.')
    p1 = plt.plot(Mobj.x[ind_section[0]], Mobj.y[ind_section[0]], 'k*', label='Start')
    p2 = plt.plot(Mobj.x[ind_section[-1]], Mobj.y[ind_section[-1]], 'b*', label='Stop')
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'Section_map.png'), dpi=400)
    plt.close()
    return ind_section

def prepare_vdata(Mobj,ind_section,fileList,variable):
    # ----- Extract data from section ------------------------
    unique_files = list(set(fileList.path))
    unique_files.sort()
    # rearrange the files:
    
    first_file = True
    #data = np.empty((0, 0, len(ind_section)))
    print('Storing the requested data in matrices')
    for ncfile in unique_files:
        print('Working on ' + ncfile)
        nc = Dataset(ncfile, 'r')
        if first_file:
            data       = nc.variables.get(variable)[:, :, :][:, :, ind_section]
            x          = Mobj.x[ind_section,0]
            y          = Mobj.y[ind_section,0]
            z          = Mobj.siglayz[ind_section, :].T
            first_file = False
        else:
            data = np.vstack((data, nc.variables.get(variable)[:, :, :][:, :, ind_section]))

        nc.close()
    return x,y,z,data

def plot_section(movie_name,
                 Mobj,
                 x,y,z,
                 data,
                 fileList,
                 ind_section,
                 variable,
                 ds          = 200, 
                 save_folder = os.getcwd(), 
                 delete_png  = True):
    '''Plot section, images and video'''
    # ---------------------------- Plot images -------------------------------------
    dist     = np.empty(len(x))
    dist[0]  = 0
    dist[1:] = np.cumsum(np.sqrt(np.square(x[1:]-x[0:-1]) + np.square(y[1:]-y[0:-1])))  
    dist     = np.tile(dist, (z.shape[0], 1))

    time_stamps = fileList.datetime    
    if variable == 'temp':
        cm  = cmocean.cm.thermal
        lev = np.linspace(data.min(), data.max(), 100)

    elif variable == 'salinity':
        cm  = cmocean.cm.haline
        lev = np.linspace(29, data.max(), 100)

    else:
        cm  = cmocean.cm.matter
        lev = np.linspace(data.min(), data.max(), 100)
 
    # Do the actual plotting
    # ------------------------------------------------------------
    images = []   
    for n in range(0, len(time_stamps)):
        tn = time_stamps[n]
        data_n = data[n, :, :]

        fig, ax1 = plt.subplots(figsize=(15, 5))
        ax1.set_facecolor('grey')
        ax1.set_ylim(bottom=z.min()-10, top=0) # Change to let zeta alter sea surface in the future?
        yticks = ax1.get_yticks()
        yticks = np.abs(yticks.astype('int'))

        ax1.set_yticklabels(yticks)
        ax1.set_ylabel('Depth (m)')
        ax1.set_xlabel('Distance (m)')

        c = ax1.contourf(dist, z, data_n, cmap=cm, levels=lev, extend='both') 
        #c.set_clim(data.min(), data.max())
        cbar = plt.colorbar(c, pad=0.01)
        cbar.set_label(variable)
        
        plt.title(tn.date())
        save_name = os.path.join(save_folder, 
                                 variable + '_section_' + 
                                 str(tn.year) + 
                                 str(tn.month) + 
                                 str(tn.day) + 
                                 str(tn.hour) +'_' + 
                                 str(n) + '.png')


        images.append(save_name)
        plt.savefig(save_name, dpi=200)
        plt.close('all')   
   
    # Make movie
    make_video(save_folder, movie_name + '_' + variable + '_section.wmv', images, speed = 6)

    # Delete png-files
    if delete_png:
        delete_files(images)

def field_movie(movie_name,
                Mobj, 
                fileList, 
                variable,
                georef,
                layer=0, 
                save_folder=os.getcwd(), 
                delete_png=True,
                xlim = None,
                ylim = None):
    '''Make horizontal field movie.'''
    if xlim is not None:
        infocus = Mobj.isinside(xlim,ylim)

    if variable == 'temp':
        cm = cmocean.cm.thermal
        lev = np.linspace(fileList.T_min, fileList.T_max, 100)

    elif variable == 'salinity':
        cm = cmocean.cm.haline
        if xlim is not None:
            lev = np.linspace(fileList.S_min, fileList.S_max, 100)
        else:
            lev = np.linspace(29, 35+(35-29)/100, 100)

    else:
        cm  = cmocean.cm.matter
        lev = np.linspace(fileList.zeta_min, fileList.zeta_max, 100)
         
    images       = []
    already_read = fileList.path[0]
    first_time   = True
    for date, file_name, index  in zip(fileList.datetime, fileList.path, fileList.index): 
        if (file_name != already_read) or (first_time):
            if 'nc' in locals():
                nc.close()

            first_time = False
            nc = Dataset(file_name, 'r')
            if variable == 'zeta':
                data = nc.variables[variable][:]
            else:
                data = nc.variables[variable][:, layer, :]
            print(file_name)

        already_read = file_name

        fig, ax1 = plt.subplots()
        ax1.set_aspect('equal')

        # Add georeference
        if georef is not None:
            plt.imshow(georef.img, extent = georef.extent, interpolation = 'spline36')

        c = ax1.tricontourf(Mobj.x[:,0], 
                            Mobj.y[:,0], 
                            Mobj.tri, 
                            data[index, :], 
                            cmap=cm, 
                            levels=lev, 
                            extend='both')
        if xlim is not None:
            ax1.set_ylim(ylim)
            ax1.set_xlim(xlim)

        co = plt.colorbar(c, ax=ax1)
        co.set_label(variable)
        plt.title(date.date())
        plt.xticks([])
        plt.yticks([])

        save_name = os.path.join(save_folder, 
                                 variable + '_' + 
                                 str(date.year) + 
                                 str(date.month) + 
                                 str(date.day) + 
                                 str(date.hour) +'_' + 
                                 str(index) + '.png')

        images.append(save_name)
        plt.savefig(save_name, dpi=200)
        plt.close()   

    # Make movie
    make_video(save_folder, movie_name + '_' + variable + '.wmv', images)

    # Delete png files
    if delete_png:
        delete_files(images)



def make_video(save_folder, movie_name,  images, speed = 12):
    '''Make movie of png-files in input list.'''
    
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(os.path.join(save_folder, movie_name), 0, speed, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    

def delete_files(files):
    '''Delete files in input list'''
    for f in files:
        os.remove(f)


class FVCOM_grid():
    'Represents FVCOM grid'

    def __init__(self, pathToFile, proj="+proj=utm +zone=33W, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs"):
        '''Create FVCOM-grid object'''
        self.filepath=pathToFile
        self.Proj = Proj(proj)

        if pathToFile[-3:] == 'mat':
            self.add_grid_parameters(['x', 'y', 'lon', 'lat', 'h', 'xc', 'yc','tri','siglayz','ntsn','nbsn'])
            self.lonc, self.latc = self.Proj(self.xc, self.yc, inverse=True)
            self.tri = self.tri - 1 
            self.nbsn = self.nbsn - 1
            self.siglayzc()
        
        if pathToFile[-3:] == '2dm':
            self.tri, self.nodes, self.x, self.y, self.z, self.types \
                                = fvgrid.read_sms_mesh(self.filepath)
             
            self.xc = (self.x[self.tri[:, 0]] + 
                       self.x[self.tri[:, 1]] + 
                       self.x[self.tri[:, 2]]) / 3
        
            self.yc = (self.y[self.tri[:, 0]] + 
                       self.y[self.tri[:, 1]] + 
                       self.y[self.tri[:, 2]]) / 3
  
            self.lon, self.lat = self.Proj(self.x, self.y, inverse=True)
            self.lonc, self.latc = self.Proj(self.xc, self.yc, inverse=True)

        if pathToFile[-3:] == '.nc':
            self.add_nc_grid(['x', 'y', 'lon', 'lat'])
            self.add_nc_grid(['xc', 'yc', 'lonc', 'latc'])
            self.add_nc_grid(['tri', 'nbsn', 'nbe', 'ntsn', 'ntve', 'nbve'])
            self.add_nc_grid(['art1', 'art2'])
            self.add_nc_grid(['siglev_c', 'siglay_c', 'siglev', 'siglay', ' h'])

          # load netcdf
            tmpdata = Dataset(pathToFile)
            
          # Triangulation
            self.tri  = tmpdata['nv'][:].transpose()-1

          # Grid horizontal locations
          # -Initialize vector to make it look like matlab output
            self.lonc      = np.zeros((len(tmpdata['lonc'][:]),1))
            self.latc      = np.zeros((len(tmpdata['lonc'][:]),1))
            self.lon       = np.zeros((len(tmpdata['lon'][:]),1))
            self.lat       = np.zeros((len(tmpdata['lon'][:]),1))
            self.x         = np.zeros((len(tmpdata['lon'][:]),1))
            self.y         = np.zeros((len(tmpdata['lon'][:]),1))
            self.xc        = np.zeros((len(tmpdata['lonc'][:]),1))
            self.yc        = np.zeros((len(tmpdata['lonc'][:]),1))

          # load data
            self.lonc[:,0] = tmpdata['lonc'][:]
            self.latc[:,0] = tmpdata['latc'][:]
            self.lon[:,0]  = tmpdata['lon'][:]
            self.lat[:,0]  = tmpdata['lat'][:]
            self.x[:,0]    = tmpdata['x'][:]
            self.y[:,0]    = tmpdata['y'][:]
            self.xc[:,0]   = tmpdata['xc'][:]
            self.yc[:,0]   = tmpdata['yc'][:]

          # Grid id
            self.nbsn = tmpdata['nbsn'][:].transpose()-1
            self.nbe  = tmpdata['nbe'][:].transpose()-1
            self.ntsn = tmpdata['ntsn'][:].transpose()-1
            self.ntve = tmpdata['ntve'][:].transpose()-1
            self.nbve = tmpdata['nbve'][:].transpose()-1

          # Grid area
            self.art1 = tmpdata['art1'][:]
            self.art2 = tmpdata['art2'][:]

          # Vertical coordinate information
            self.siglev_c = tmpdata['siglev_center'][:]
            self.siglay_c = tmpdata['siglay_center'][:]
            self.siglev   = tmpdata['siglev'][:]
            self.siglay   = tmpdata['siglay'][:]
            self.h        = tmpdata['h'][:]
            tmpdata.close()

            self.siglayz  = (self.h*self.siglay).T
            
    def siglayzc(self):
        '''Calculate depths (z) at uv-points'''
        self.siglayz_uv = (self.siglayz[self.tri[:, 0], :] + 
                           self.siglayz[self.tri[:, 1], :] + 
                           self.siglayz[self.tri[:, 2], :]) / 3
    def add_nc_grid(self,names):
        for name in names:
            setattr(self,name,0)

    def add_grid_parameters(self, names):
        '''Read grid attributes from mfile and add them to FVCOM_grid object'''
        grid_mfile = loadmat(self.filepath)
        
        if type(names) is str:
            names=[names]
       
        for name in names:
            setattr(self, name, grid_mfile['Mobj'][0,0][name])
    

    def find_nearest(self, x, y):
        '''Find indices of nearest grid point to given points.'''
        indices = []
        for xi, yi in zip(x, y):
            ds = np.sqrt(np.square(self.x-xi) + np.square(self.y - yi))
            indices.append(ds.argmin())

        return indices


    def find_nearest_uv(self, x, y):
        '''Find indices of nearest grid point to given points.'''
        indices = []
        for xi, yi in zip(x, y):
            ds = np.sqrt(np.square(self.xc-xi) + np.square(self.yc - yi))
            indices.append(ds.argmin())

        return indices


    def isinside(self, x, y, x_buffer=0, y_buffer=0):
        '''Check which nodes are inside the rectangle bounded by the xy limits of the grid'''
        x_min = min(x) - x_buffer
        x_max = max(x) + x_buffer
        y_min = min(y) - y_buffer
        y_max = max(y) + y_buffer

        inside_x = np.logical_and(self.x>=x_min, self.x<=x_max)
        inside_y = np.logical_and(self.y>=y_min, self.y<=y_max)
        inside   = np.logical_and(inside_x, inside_y)
        
        return inside



    def plot_grid(self, show=False):
        '''Plot mesh grid'''
        plt.triplot(np.squeeze(self.x), 
                    np.squeeze(self.y), 
                    self.tri, 
                    'g-', markersize=0.2, linewidth=0.2)
        
        plt.axis('equal')
        if show:
            plt.show()    
 

    def plot_field(self, field):
        '''Plot scalar field on grid'''
        plt.tripcolor(np.squeeze(self.x), np.squeeze(self.y), self.tri, np.squeeze(field))
        plt.axis('equal')
        plt.show()    


