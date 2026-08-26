"""
BuildCase - Calculates river runoff from individual rivers based on their areal fraction of the watershed and the total runoff from each watershed
"""
import os
import netCDF4
import numpy as np
import pandas as pd
import shapely as shp
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import fvtools.grid.grid_metrics as gm

from datetime import datetime, timedelta, timezone
from fvtools.grid.fvcom_grd import FVCOM_grid
from scipy.spatial import cKDTree as KDTree
from shapely.plotting import plot_polygon
from functools import cached_property
from scipy import interpolate
from pyproj import Proj

import warnings
warnings.filterwarnings("ignore")

global version
version = 2.0

def main(start, stop, vassdrag, mesh_dict, info = None):
    '''
    BuildRiver use data from the NVE HBV model and feed all the mapped rivers leading to the ocean to FVCOM

    Parameters:
    ----
    start:     yyyy-mm-dd
    stop:      yyyy-mm-dd
    vassdrag:  [233, 234] etc, any array containing all ids (integers) will do.

    Optional:
    ----
    mesh_dict: (M.npy by default)
    info:      Dict where all paths and river specific settings are stored. Change the basic settings by giving it as
               an input: info. Get the basic settings by calling BuildRivers.get_input(),
               which can then be edited and passed back to main if other paths etc. are
               needed in the given experiment.

    hes@akvaplan.niva.no
    '''
    # Convert time
    start = datetime(*[int(t) for t in start.split('-')], tzinfo = timezone.utc)
    stop  = datetime(*[int(t) for t in stop.split('-')], tzinfo = timezone.utc)

    if info is None:
        info = get_input()

    # 1. Load the FVCOM grid, the national runoff data, the river outlet positions and the river temperature data. 
    #    Initialize the object that will move rivers to the mesh and ensure numerical stability
    # -------------------------------------------------------------------------------
    # - Read the mesh
    M = FVCOM_grid(mesh_dict)
    info['grid_projection'] = M.reference
    M.re_project(info['river_projection'])
    M.get_cfl(verbose=False)

    print('----------------------------------------------------------------------------')
    print(f'                       BuildRivers: {M.casename}')
    print('----------------------------------------------------------------------------')

    # - Load the necessary data and data handlers
    print('Compute grid connectivity and identify landpoints')
    Forcing   = FVCOM_rivers(info, M, vassdrag, start, stop)
    Runoff    = HBVRunoff(info['runoff'], start = start - timedelta(days = 1), stop = stop + timedelta(days=1))
    Positions = RiverPositions(info['riverpositions'], vassdrag, info['river_projection'])
    Temp      = RiverTemperatures('riverdata/', start = start - timedelta(days=1), stop = stop + timedelta(days=1))

    # 2. Reduce the rivers to only load those that are relevant to this simulation
    #    and prepare to force the model
    # ------------------------------------------------------------------------------
    # - Remove the rivers that are too far away from land and too close to the obc
    print('\nSubset the river database to cover the model domain')
    Positions = Forcing.crop_rivers_far_from_land_and_close_to_OBC(Positions)

    # - Re-distribute the runoff according to catchment area
    print('- Determine runoff from rivers by the rivers catchment area')
    Forcing.redistribute_runoff(Positions, Runoff)

    # - Add temperatures to the rivers
    print('\nAdd temperature profiles to the rivers')
    print(f'- read temperature databases from {info["rivertemp"]}, calculate the river climatology for each watershed')
    river_temp = Temp.make_individual_river_climatology(Positions)

    print('- Add observed temperatures to the rivers where available')
    river_temp =  Temp.insert_observed_temperature(river_temp, vassdrag)

    print('- Set the runoff and temperauture for each river in the model, interpolate to model time')
    Forcing.interpolate_forcing_to_model_time(Runoff, Temp, river_temp)

    print('\nConnect rivers to the mesh')
    Forcing.connect_to_mesh(M)

    # 4. Do a simple quality control, and write the necessary forcing files
    # - Show the rivers we'll use, and the values they will force FVCOM with
    print('\nFinished, plotting the forcing')
    M.re_project(info['grid_projection'])
    show_forcing(Forcing, M, Positions)
    M.re_project(info['river_projection'])

    # - Write to netCDF, write RiverNamelist.nml
    Forcing.dump()
    Forcing.write_namelist()

def identify_movable_rivers(info, vassdrag, mesh_dict):
    '''
    Figure out which rivers we want to move
    '''
    # - Read the mesh
    M = FVCOM_grid(mesh_dict)
    info['grid_projection'] = M.reference
    M.re_project(info['river_projection'])
    M.get_cfl(verbose=False)

    # - Load the necessary data and data handlers
    Forcing   = FVCOM_rivers(info, M, vassdrag, datetime.now()-timedelta(days=1), datetime.now())
    Positions = RiverPositions(info['riverpositions'], vassdrag, info['river_projection'])

    print('- Remove irrelevant rivers')
    Positions = Forcing.crop_rivers_far_from_land_and_close_to_OBC(Positions, info)

    return Positions, M

def get_input(river_data_path = 'riverdata'):
    """
    Pre-defined paths are stored here. They are distributed to other parts of the code via main.

    iloc:        Determine if the input is given as a flux at edge or at the node
    land buffer: For identifying rivers draining to the model domain, will include rivers within *land buffer* from the model coastline
    obc buffer:  For masking rivers within *obc buffer* from the model open boundary
    min_depth:   Minimum depth at rivers (since if h=0 (wetting/drying on), rivers would be infinitely wide to meet the stability criteria)
    Isplit:      Baroclinic split
    tideamp:     Tidal amplityde
    plot:        Show the results on a map
    rivertemp:   River temperature folder
    vassdrag:    File containing runoff data from vassdrag
    LargeRivers: File containing info about the large rivers in Norway
    SmallRivers: File containing info about the small rivers in Norway
    minrcoef:    Tunable parameter to determine the maximum volume of a CV we will let a river fill over a timestep
    river_projection: Coordinate system the river positions are stored in (UTM33)
    """
    info = {
        'iloc': 'edge',
        'land buffer': 1500,
        'buffer domain ratio': 0.1,
        'obc buffer': 5000,
        'min_depth': 3,
        'Isplit': 8,
        'tideamp': 1,
        'plot': True,
        'compile river': True,
        'rivertemp': f'{river_data_path}/',
        'runoff': f'{river_data_path}/Niva_1990-2024_2018v20.05/',
        'riverpositions': f'{river_data_path}/river_positions.csv',
        'minrcoef': 0.3,
        'river_projection': 'epsg:32633'
    }
    return info

class HBVRunoff:
    def __init__(self, hbvfolder, start = None, stop = None):
        '''
        Read all hbv output files in the hbvfolder directory
        - Assumes that all .var files in the hbv folder are runoff data.
        - crop to time between start and stop
        '''
        # Load data for the full country
        data = []
        for i, file in enumerate(os.listdir(hbvfolder)):
            # Figure out which number this vassdragsområde is
            vassdragsomraade = int(file.split('.var')[0][-3:])
        
            # Load the data
            _data = pd.read_csv(f'{hbvfolder}{file}', header = None, sep=r'\s+')
        
            # Load the dates, will have the same coverage in all files
            if i == 0:
                dates = [datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[9:11]), tzinfo = timezone.utc) for d in _data[0]]
        
            data.append(_data.rename(columns = {1: vassdragsomraade}).drop(columns=0))
        
        # Re-scale vassdragsområde 183 (python index 182), as suggested by James Sample at NIVA
        data[182] = 6/16 * data[181]
        self.transport = pd.concat(data, axis = 1).T
        self.dates = np.array(dates)

        assert any(self.dates <= start), f'the database does not cover dates prior to {self.dates[0]}'
        assert any(self.dates >= stop),  f'the database does not cover dates after {self.dates[-1]}'
            
        
        # Crop to requested time-span
        crop = np.logical_and(self.dates > start, self.dates < stop)

        self.transport = self.transport.loc[:, crop]
        self.dates = self.dates[crop]

class RiverPositions:
    def __init__(self, path, vassdrag, epsg_code):
        '''
        Connect river outlets identified by fvtools.rivers.NVEforcing.elvis_riveroutlets to the FVCOM mesh
        - path:      e.g. riverdata/river_positions.csv
        - vassdrag:  list of vassdrags covered by the model
        - epsg_code: EPSG code of the projection the river positions is stored in. Should always be in UTM33 (epsg:32633) in Norway.
        '''
        # Set georeference, and prepare a Proj object for the river data
        self.reference = epsg_code
        self.Proj = Proj(epsg_code)

        # load all rivers
        rivers = pd.read_csv(path).set_index('rivers')

        # Remove those who are not in the requested vassdragsområde
        self.rivers = rivers[[v in vassdrag for v in rivers.vassdragsomraade]]

        # Set initial x- and y-positions
        self.x, self.y = self.rivers.x_outlet.values, self.rivers.y_outlet.values

    @cached_property
    def crs(self):
        return ccrs.epsg(int(self.reference.split(':')[-1]))

    @property
    def x(self):
        return self.rivers.x_outlet.values 

    @x.setter
    def x(self, val):
         self.rivers.x_outlet = val

    @property
    def y(self):
        return self.rivers.y_outlet.values

    @y.setter
    def y(self, val):
        self.rivers.y_outlet = val

    # Derive lon, lat from x,y
    @property
    def lonlat(self):
        _lon, _lat = self.Proj(self.x, self.y, inverse = True)
        return [_lon, _lat]
    
    @property
    def lon(self):
        return self.lonlat[0]
    
    @property
    def lat(self):
        return self.lonlat[1]

    @property
    def names(self):
        '''
        Large rivers have names (not necessarilly unique), small rivers just have their IDs
        '''
        names = []
        for r in self.rivers.iterrows():
            if type(r[1].elvenavn) == str:
                names.append(f'{r[0]} - {r[1].elvenavn}')
            else:
                names.append(r[0])
        return names

    def crop(self, indices):
        '''
        Remove rivers that are not requested from the rivers dataframe
        '''
        self.rivers = self.rivers.iloc[indices]

class DraggablePoints:
    '''
    Drag and drop solution suggested by Google AI (from the google search bar)
    '''
    def __init__(self, ax, x_data, y_data):
        self.ax = ax
        self.was_modified = False

        # Plot the points and keep a reference to the Line2D artis
        self.points, = ax.plot(x_data, y_data, 'ko', markersize=10)
        
        # Internal state tracking
        self.selected_index = None
        self.x_data = list(x_data)
        self.y_data = list(y_data)
        
        # Connect to matplotlib mouse events
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        # Ensure click occurs inside the axes plot area
        if event.inaxes != self.ax: 
            return
        
        # Transform data coordinates to display (pixel) coordinates
        xy_pixels = self.ax.transData.transform(np.column_stack((self.x_data, self.y_data)))
        click_pixel = np.array([event.x, event.y])
        
        # Calculate distance from click to all points
        distances = np.linalg.norm(xy_pixels - click_pixel, axis=1)
        
        # If click is within 15 pixels of a point, select the closest one
        if np.min(distances) < 15:
            self.selected_index = np.argmin(distances)

    def on_motion(self, event):
        # Cancel if no point is selected or mouse leaves axes bounds
        if self.selected_index is None or event.inaxes != self.ax: 
            return

        # Change state if any point was modified
        self.was_modified = True

        # Update point data with new cursor coordinates
        self.x_data[self.selected_index] = event.xdata
        self.y_data[self.selected_index] = event.ydata
        
        # Refresh the artist data and redraw canvas
        # Could be interesting to add a line to the original position?
        self.points.set_data(self.x_data, self.y_data)
        self.canvas.draw_idle()

    def on_release(self, event):
        # Deselect point on mouse release
        self.selected_index = None

class CropRivers:
    '''
    Use FVCOM_rivers to crop the national river database to the FVCOM experiment
    - Includes functions to move rivers in the domain to match with the mesh
    '''
    @property
    def model_domain(self):
        '''
        The outermost solid boundary of the model domain
        '''
        return shp.geometry.Polygon(self.M.model_boundary)

    @property
    def buffered_domain(self):
        '''
        A concave hull of the solid boundary, used to filter out rivers in nearby fjords draining to areas outside of the model domain
        '''
        return shp.concave_hull(shp.geometry.Polygon(self.model_domain), ratio = self.info['buffer domain ratio']).buffer(self.info['land buffer'])

    def crop_rivers_far_from_land_and_close_to_OBC(self, Rivers):
        """
        Remove rivers that run off to points outside of the model domain, and that run off very close to the open boundary
        - note: I believe you have to run matplotlib with the tk backend for the draggable points to work out
        """
        # Filter rivers so that we only keep those within the model domain, and within a certain buffer of it
        i = 0
        print('- Adjust river positions and remove rivers that do not drain to the model domain')
        while True:
            i += 1
            print(f' - iteration {i}')
            # Identify which rivers are within the current coverage
            in_model = self.buffered_domain.contains([shp.geometry.Point(x, y) for (x,y) in zip(Rivers.x, Rivers.y)])

            plt.figure(figsize = [10,10])
            # Reproject since Norgeskart WMS requires that the region we plot are within valid bounds of the UTM coordinate,
            # and UTM33/32 -- which the river data is stored in -- is not in general valid.
            self.M.re_project(self.info['grid_projection'])
            self.M.georeference()
            ax = plt.gca()

            # Plot the model land
            #plot_polygon(shp.geometry.Polygon(self.model_domain), ax = ax, facecolor='black', edgecolor='red', alpha=0.5)
            ax.plot(self.x_land, self.y_land, 'b.', label = 'land nodes', zorder = 1)

            # Plot the buffered domain
            plot_polygon(self.buffered_domain, ax = ax, facecolor='lightblue', edgecolor='blue', alpha=0.5)
        
            self.M.re_project(Rivers.reference)

            ax.set_title('Rivers near the domain scaled with their runoff. Black points are draggable, close figure to continue.')

            # Plot rivers that will not be used in the model
            ax.scatter(
                Rivers.x[~in_model], Rivers.y[~in_model], 
                s = Rivers.rivers.areal[~in_model], 
                color = 'r', 
                transform = Rivers.crs, 
                label = 'Will not be used'
                )

            # Plot rivers that will be used in the model
            ax.scatter(
                Rivers.x[in_model], Rivers.y[in_model],
                s = Rivers.rivers.areal[in_model], 
                color = 'g', 
                transform = Rivers.crs, 
                label = 'Will be used'
                )

            # Go back to the map projection
            self.M.re_project(self.info['grid_projection'])
            x, y = self.M.Proj(Rivers.lon, Rivers.lat)

            # Modify river positions in Rivers (will not change the database, just the rivers used in the model setup)
            river_positions = DraggablePoints(ax, x, y)
            ax.legend()
            plt.show(block = True)

            # Once the figure is closed, we look at all points and update the river position database
            lon, lat = self.M.Proj(river_positions.x_data, river_positions.y_data, inverse = True)
            self.M.re_project(Rivers.reference)
            
            # Check if rivers were adjusted. We continue
            if river_positions.was_modified:
                print('  - One or more river was modified, making a new river mask and illustrating.')
                x, y = Rivers.Proj(lon, lat)
                Rivers.x, Rivers.y = np.array(x), np.array(y)

            else:
                break

        # Remove the unused rivers from the Rivers object
        Rivers.crop(np.where(in_model)[0])

        # Store the adjusted river database for later use
        if i > 1:
            path = self.M.filepath.split(self.M.filepath.split('/')[-1])[0]
            new_name = f'{path}river_positions_{self.M.casename}.csv'
            print(f'- The river database was adjusted for this experiment, storing it as {new_name}')
            Rivers.rivers.to_csv(new_name)

        # And then we can identify and remove rivers that are too close to the OBC
        print(f'- Remove rivers that discharge within {self.info["obc buffer"]} m from the open boundary')
        d, _ = self.obc_tree.query(np.array([Rivers.x, Rivers.y]).T)
        too_close_to_obc_inds = np.where(d > self.info['obc buffer'])[0]
        Rivers.crop(too_close_to_obc_inds)
        
        return Rivers

class FVCOM_rivers(CropRivers):
    """
    Class storing data that eventually ends up as FVCOM forcing
    """
    def __init__(self, info, M, vassdrag, start, stop):
        self.info, self.vassdrag, self.M = info, vassdrag, M
        self.nodes, self.cells = gm.get_nbe(self.M)
        self.start, self.stop = start, stop

    @property
    def xy_land(self):
        if self.info['iloc'] == 'edge':
            model_boundary = np.array(np.where(self.M.nbe.min(axis=1)==-1))
            land = model_boundary[self.M.ISBCE[model_boundary] == 1]
            x_land = self.M.xc[land]
            y_land = self.M.yc[land]

        elif self.info['iloc'] == 'node':
            x_land  = self.M.x[self.nodes['boundary'][np.where(self.nodes['id']==1)[0]]]
            y_land  = self.M.y[self.nodes['boundary'][np.where(self.nodes['id']==1)[0]]]
            
        return np.array([x_land, y_land]).T

    @property
    def x_land(self):
        return self.xy_land[:,0]

    @property
    def y_land(self):
        return self.xy_land[:,1]

    @property
    def xy_obc(self):
        if self.info['iloc'] == 'edge':
            model_boundary = np.array(np.where(self.M.nbe.min(axis=1)==-1))
            obc = model_boundary[self.M.ISBCE[model_boundary] == 2]
            x_obc = self.M.xc[obc]
            y_obc = self.M.yc[obc]

        elif self.info['iloc'] == 'node':
            x_obc   = self.M.x[self.nodes['boundary'][np.where(self.nodes['id']==2)[0]]]
            y_obc   = self.M.y[self.nodes['boundary'][np.where(self.nodes['id']==2)[0]]]
        return np.array([x_obc, y_obc]).T

    @property
    def x_obc(self):
        return self.xy_obc[:,0]

    @property
    def y_obc(self):
        return self.xy_obc[:,1]

    @property
    def land_tree(self):
        return KDTree(np.array([self.x_land, self.y_land]).transpose())

    @property
    def obc_tree(self):
        return KDTree(np.array([self.x_obc,  self.y_obc]).transpose())

    @property
    def mesh_tree(self):
        if self.info['iloc'] == 'edge':
            _mesh_tree = KDTree(np.array([self.M.xc, self.M.yc]).transpose())
        else:
            _mesh_tree = KDTree(np.array([self.M.x,  self.M.y]).transpose())
        return _mesh_tree

    def redistribute_runoff(self, Rivers, Runoff):
        """
        Assign fraction of total runoff to watershed to the individual river
        - the area fraction of the vassdragsområde (catchment area) associated with each river is calculated 
          in fvtools.rivers.NVEforcing.elvis_riveroutlets. So the runoff from each individual river is independent 
          of the model domain, but you will loose runoff to the model domain if you exclude rivers that should 
          fill your domain. This is different from the historic setup, where the total runoff from a vassdragsområde was conserved.
        """
        # Redistribute the total runoff in each watershed to all known rivers in the watershed
        self.Runoff      = (Runoff.transport.loc[Rivers.rivers.vassdragsomraade] * Rivers.rivers.area_fraction.to_numpy()[:, None]).set_index(Rivers.rivers.index) # total runoff to the watershed is conserved in this operation
        self.river_names = Rivers.names
        self.short_names = Rivers.rivers.index.values

        # Store positions in the forcing object as well
        self.xriv, self.yriv = Rivers.x, Rivers.y

    def interpolate_forcing_to_model_time(self, Runoff, Temp, river_temp, dt = 3/24):
        '''
        Create a time-vector for the forcing file
        '''
        # Convert to easy-to-deal-with time
        runoff_dates = np.array(netCDF4.date2num(Runoff.dates, units = 'days since 1858-11-17 00:00:00'))
        temp_time    = np.array(netCDF4.date2num(Temp.time, units = 'days since 1858-11-17 00:00:00'))
        start_num    = netCDF4.date2num(self.start, units = 'days since 1858-11-17 00:00:00')
        stop_num     = netCDF4.date2num(self.stop, units = 'days since 1858-11-17 00:00:00')

        # Check if the time covers the model period
        # transport
        if stop_num > runoff_dates[-1]:
            raise ValueError(f'{self.info["runoff"]} does not cover the stop date, {stop_num} was requested, but the latest runoff is {runoff_dates[-1]}')

        elif start_num < runoff_dates[0]:
            raise ValueError(f'{self.info["runoff"]} does not cover the start date, {start_num} was requested, but the earliest runoff is {runoff_dates[0]}')

        # temperature
        if start_num < temp_time[0]:
            raise ValueError(f'{self.info["rivertemp"]} starts after start date, {start_num} was requested but the earliest temperatute available was {temp_time[0]}')

        elif stop_num > temp_time[-1]:
            raise ValueError(f'{self.info["rivertemp"]} ends before end date, {stop_num} was requested but the earliest temperatute available was {temp_time[-1]}')

        # Prepare the output files
        self.model_time = np.arange(start_num, stop_num + dt, dt)

        # Interpolate to output structure
        self.RiverTransport = np.zeros((len(self.model_time), len(self.xriv)))
        self.RiverTemp      = np.zeros((len(self.model_time), len(self.xriv)))

        # These fields will be dumped to the model forcing file
        for i in range(len(self.xriv)):
            f_transport   = interpolate.interp1d(runoff_dates, self.Runoff.to_numpy()[i, :])
            f_temperature = interpolate.interp1d(temp_time, river_temp.to_numpy()[i, :])
            self.RiverTransport[:, i] = f_transport(self.model_time)
            self.RiverTemp[:, i]      = f_temperature(self.model_time)

    def connect_to_mesh(self, M):
        """
        Figure out which node/cell the flux should go to.
        """
        first = True
        while True:
            _, land_loc = self.land_tree.query(np.array([self.xriv, self.yriv]).transpose())
            if first:
                self.river_connection(land_loc, M, self.info['river_projection'], self.info['grid_projection'])
                first = False

            _, mesh_location = self.mesh_tree.query(np.array([self.x_land[land_loc], self.y_land[land_loc]]).transpose())
            self.mesh_location = mesh_location

            print('- merge rivers that go to the same mesh point')
            self.RiverTransport, self.RiverTemp, self.river_names, self.short_names = self.merge_rivers()
            rcoef = self.river_stability()

            # Split problematic rivers if we are troubled with such things
            bad_rivers = np.where(rcoef > self.info['minrcoef'])[0]
            if any(bad_rivers):
                print('- split rivers that need to be distributed over larger areas')
                self.split_problematic_river(bad_rivers, rcoef)
            else:
                break

        # Figure out where the nearest land point to this river is, and where in the mesh this land point is
        _, land_loc      = self.land_tree.query(np.array([self.xriv, self.yriv]).transpose())
        _, mesh_location = self.mesh_tree.query(np.array([self.x_land[land_loc], self.y_land[land_loc]]).transpose())
        self.mesh_location = mesh_location

        # Show the final position of the river
        plt.scatter(
            self.xriv, self.yriv, s = 50, 
            c = 'm', 
            label = 'final river nodes', 
            transform = ccrs.epsg(int(self.info['river_projection'].split(':')[-1]))
            )
        plt.legend()

    def merge_rivers(self):
        """
        Put rivers into the same structure
        """
        # Loop over each location and dump the river data fields
        self.unique_mesh = np.unique(self.mesh_location)
        transport   = np.zeros((self.RiverTransport.shape[0], len(self.unique_mesh)))
        temperature = np.zeros((self.RiverTemp.shape[0], len(self.unique_mesh)))

        names = []
        short_names = []
        for i, mesh_id in enumerate(self.unique_mesh):
            places            = np.where(self.mesh_location == mesh_id)[0]       # Find all rivers going to the same FVCOM node
            transport[:,i]   += np.sum(self.RiverTransport[:, places], axis = 1) # Add the transport
            temperature[:,i] += np.mean(self.RiverTemp[:, places], axis = 1)     # Just average the temperature, there's probably a better way
            names.append(', '.join([self.river_names[place] for place in places]))
            short_names.append(', '.join([self.short_names[place] for place in places]))

        # We xriv, yriv will from now on be the location of the river in an FVCOM-sense
        if self.info['iloc'] == 'edge':
            self.xriv    = self.M.xc[self.unique_mesh]
            self.yriv    = self.M.yc[self.unique_mesh]

        elif self.info['iloc'] == 'node':
            self.xriv    = self.M.x[self.unique_mesh]
            self.yriv    = self.M.y[self.unique_mesh]

        return transport, temperature, names, short_names

    def river_stability(self):
        """
        As mentioned in the FVCOM manual page 73:
        - To avoid negative salinities due to advection-related issues,
          the flux ratio can not exceed a certain threshold:
        Depth_cell > internal_delta_t * river_flux / Control_volume_area
        """
        # The control volume area can take some time to compute, let's settle for the triangle area (which is ~ 1/2 of the CV area on average)
        mesh_index_of_rivers = self.M.find_nearest(self.xriv, self.yriv, 'cell')
        tri_area = self.M.tri_area[mesh_index_of_rivers]
        dt_internal = min(self.M.ts)*self.info['Isplit']

        # Depth at river locations
        if self.info['iloc'] == 'edge':
            h  = self.M.hc[self.unique_mesh]-self.info['tideamp']

        elif self.info['iloc'] == 'node':
            h  = self.M.h[self.unique_mesh]-self.info['tideamp']

        # Set min depth in case of wetting drying
        h[h < self.info['min_depth']] = self.info['min_depth']

        # Calculate the stability number for all the river-cells
        return dt_internal * self.RiverTransport.max(axis=0) / (h*tri_area)

    def split_problematic_river(self, bad_rivers, rcoef):
        """
        We must make sure that a river does not completely fill a control
        volume in one timestep. (Isn't this a silly problem?)
        """
        # Find nearest land nodes, share the river with them untill rcoef should be < minrcoef
        # --> May only work for rivers that can stay relatively close...
        for river in bad_rivers:
            # Find number of new points we need
            n_newland     = int(np.ceil(rcoef[river]/self.info['minrcoef'])+1)

            # Find nearby FVCOM land that we can use
            _, this_land  = self.land_tree.query(np.array([self.xriv[river], self.yriv[river]]).transpose(), k = n_newland)

            # Update with new positions
            new_x         = np.copy(self.x_land[this_land])
            new_y         = np.copy(self.y_land[this_land])

            # Copy the stuff we are removing
            transport           = np.copy(self.RiverTransport[:, river])/n_newland # Split the transport equally among all river points
            temp                = np.copy(self.RiverTemp[:, river]) # The energy should not be split

            # Delete current entry of this river in all relevant reference lists
            self.RiverTemp      = np.delete(self.RiverTemp, river, 1)
            self.RiverTransport = np.delete(self.RiverTransport, river, 1)
            self.xriv           = np.delete(self.xriv, river)
            self.yriv           = np.delete(self.yriv, river)
            long_name           = self.river_names.pop(river) # pop removes this item from the list and "dumps" it to long_name
            short_name          = self.short_names.pop(river)

            # Insert the split version in the new nodes
            for i in range(n_newland):
                self.RiverTransport = np.append(self.RiverTransport, transport[:, None], axis = 1)
                self.RiverTemp = np.append(self.RiverTemp, temp[:, None], axis = 1)
                self.river_names.append(f'{long_name}-p{i}')
                self.short_names.append(f'{short_name}-p{i}')

            # Update river location
            self.xriv = np.append(self.xriv, new_x)
            self.yriv = np.append(self.yriv, new_y)

    def river_connection(self, land_loc, M, river_projection, grid_projection):
        """
        Show how far rivers have been moved from NVE location
        """
        plt.figure()
        M.re_project(grid_projection)
        M.georeference()
        M.re_project(river_projection)
        transform = ccrs.epsg(int(river_projection.split(':')[-1]))
        
        # Make lines connecting rivers to their FVCOM location
        xvec = np.array([self.x_land[land_loc], self.xriv]).transpose()
        yvec = np.array([self.y_land[land_loc], self.yriv]).transpose()
        xvec_nan = np.insert(xvec, 2, np.nan, axis = 1).ravel()
        yvec_nan = np.insert(yvec, 2, np.nan, axis = 1).ravel()

        plt.plot(self.x_land, self.y_land, 'b.', label = 'land', transform = transform)
        plt.plot(self.x_land[land_loc], self.y_land[land_loc], 'k.', label = 'land with river', transform = transform)
        plt.plot(self.xriv, self.yriv, 'g.', label = 'river location from NVE', transform = transform)
        plt.plot(xvec_nan, yvec_nan, 'r', label = 'vector from NVE location to FVCOM location', transform = transform)
        plt.axis('equal')

    def dump(self):
        """
        Write the riverdata.nc file for river forcing
        """
        # Initialize file
        with netCDF4.Dataset('riverdata.nc', 'w') as d:
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
            time.units       = 'days since ' + str(datetime(1858, 11, 17, 0, 0, 0))
            time.format      = 'modified julian day (MJD)'
            time.time_zone   = 'UTC'
    
            # - Itime
            Itime = d.createVariable('Itime', 'int32', ('time',))
            Itime.long_name   = 'integer days'
            Itime.units       = 'days since ' + str(datetime(1858, 11, 17, 0, 0, 0))
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
            names._Encoding = 'ascii'
            for i, name in enumerate(self.river_names):
                if len(name) > 80:
                    this_name = name[:80]
                else:
                    this_name  = name + (80-len(name))*' '
                this_name = self.fix_nordic(this_name)
                names[i,:] = np.array(this_name, dtype = 'S80')

    def fix_nordic(self, this_name):
        '''
        FVCOM does not accept norwegian letters
        '''
        this_name = this_name.replace('å','a')
        this_name = this_name.replace('Å','A')
        this_name = this_name.replace('ø','o')
        this_name = this_name.replace('Ø','O')
        this_name = this_name.replace('æ','e')
        this_name = this_name.replace('Æ','E')
        return this_name

    def write_namelist(self, namelist = 'RiverNamelist.nml', riverfile = 'riverdata.nc'):
        """
        Write a namelist to accompany the netCDF file
        """
        VQDIST = -np.diff(self.M.siglev[0,:])
        with open(namelist, 'w+') as f:
            for i, river in enumerate(self.river_names):
                river = self.fix_nordic(river)
                f.write(' &NML_RIVER\n')
                f.write(f" RIVER_NAME = '{river}'\n")
                f.write(f" RIVER_FILE = '{riverfile}'\n")
                f.write(f' RIVER_GRID_LOCATION = {self.mesh_location[i]+1}\n')
                vertical_dist = np.array2string(np.round(VQDIST,6), separator = ' ', edgeitems = 6,
                                                precision = 5, floatmode = 'fixed').replace('\n',' ')[1:-1]
                f.write(f' RIVER_VERTICAL_DISTRIBUTION = {vertical_dist}\n')
                f.write('/\n')

class RiverTemperatures:
    '''
    Prepares climatological temperature at each watershed area for the FVCOM experiment
    - Also provides preliminary 
    '''
    def __init__(self, riverpath, start, stop):
        '''
        Read observed river temperatures and river climatologies
        '''
        self.start, self.stop = start, stop
        self.climatology = pd.read_csv(f'{riverpath}/temperature_climatology.csv', index_col = 0)
        self.climatology.columns = self.climatology.columns.astype(int)
        self.observations = {}
        observation_folder = f'{riverpath}/temperature_observations/'
        
        for f in os.listdir(observation_folder):
            if '.csv' in f:
                _tmp = pd.read_csv(f'{observation_folder}{f}')
                _tmp['time'] = pd.to_datetime(_tmp.time)
                self.observations[f.split('.csv')[0]] = _tmp

    @property
    def time(self):
        '''
        Days we will force the model
        '''
        return [self.start + timedelta(days = n) for n in range((self.stop - self.start).days + 1)]

    @property
    def timestamp(self):
        return pd.to_datetime([t.timestamp() for t in self.time])
    
    @property
    def day_of_year(self):
        '''
        Which day of the year each time is
        '''
        return np.array([t.timetuple().tm_yday for t in self.time])

    @property
    def vassdrag_temperature_at_time(self):
        '''
        Temperatures on the day we force the model
        '''
        temp = self.climatology.T
        return temp.loc[self.day_of_year].T

    def make_individual_river_climatology(self, Rivers):
        '''
        Connect individual rivers to the climatoligical temperature on the days we need data to force FVCOM
        '''
        river_temp = self.vassdrag_temperature_at_time.loc[Rivers.rivers.vassdragsomraade]
        river_temp['elv-ID'] = Rivers.rivers.index
        return river_temp.set_index('elv-ID')

    def insert_observed_temperature(self, river_temp, vassdrag):
        '''
        Use the observed temperature instead of the climatology where we have it
        '''
        # Identify which temperature observation are within the model domain
        within_domain = [key for key in self.observations.keys() if int(key.split('-')[0]) in vassdrag]
        
        # Interpolate observations to the forcing time
        if any(within_domain):
            for key in within_domain:
                observation = self.observations[key]
                
                # Only use quality controlled data
                observation = observation[observation.quality > 1]
        
                # Check if the observation covers the modelled period, crop the array is so
                if observation.time.iloc[0] < self.time[0] and observation.time.iloc[-1] > self.time[-1]:
                    observation = observation[np.logical_and(observation.time >= self.start - timedelta(days = 7), observation.time <= self.stop + timedelta(days = 7))]
                else:
                    print(f'  - Station {key} did not cover the simulation period')
                    continue
        
                # Require that the coverage of temperature data (daily) is at least 90% of the model period (to avoid making up too much data)
                if len(observation)/len(self.time) < 0.9:
                    print(f'  - Station {key} had too many gaps in the simulation period')
                    continue
                    
                # Interpolate the observed temperature to the forcing data
                print(f'  - Replacing climatology with observations at {key}')
                f = interpolate.interp1d([t.timestamp() for t in observation.time], observation.filtered_temperature)
                river_temp.loc[key] = f(self.timestamp)
        return river_temp

# Show what we will write to the riverdata forcing
# ----
def show_forcing(obj, M, Rivers):
    """
    Simple figures to see that the routine got the basics right
    """
    plt.figure()
    M.georeference()
    plt.plot(obj.x_land, obj.y_land, 'g.', label = 'land nodes', zorder = 1)
    plt.scatter(obj.xriv, obj.yriv, np.mean(obj.RiverTransport, axis = 0), c = np.mean(obj.RiverTransport, axis = 0), zorder = 5, transform = Rivers.crs)
    plt.title('Average transport')
    plt.axis('equal')
    plt.colorbar(label = r'm$^3$ s$^{-1}$')
    plt.show(block = False)

    plt.figure()
    M.georeference()
    plt.scatter(obj.xriv, obj.yriv, obj.RiverTemp.max(axis = 0), c = obj.RiverTemp.max(axis = 0), cmap = 'inferno', zorder = 5, transform = Rivers.crs)
    plt.title('Max temperature in model period')
    plt.axis('equal')
    plt.colorbar(label = 'degrees celcius')
    plt.show(block = False)

class InputError(Exception): pass

# Development:
# Create a river subsetting method
# -> Load mother model rivers as "raw data" that can be processed further using the existing fvtools scripts
from netCDF4 import Dataset
import f90nml

class MotherRivers:
    '''
    Class that reads rivers that have been used to force a larger model domain
    '''
    def __init__(self, nmlfile, river_nc, mothergrid, river_inflow_location = 'edge', reference = 'epsg:32633'):
        '''
        Reads
        - nmlfile:    River namelist for the mother rivers
        - river_nc:   River forcing file for the mother domain rivers
        - mothergrid: Grid file for the FVCOM mother domin
        '''
        # We do not support nodes input at the moment
        if river_inflow_location != 'edge':
            raise ValueError(f"Sorry, we only support river_inflow_location = 'edge' at the moment, {river_inflow_location} is not available at the moment.")

        # Process input
        nml     = f90nml.read(nmlfile)
        rivernc = Dataset(river_nc)
        M       = FVCOM_grid(mothergrid, reference = reference)

        # Load river locations and names
        self.river_locations = np.array([nml['river_grid_location'] - 1 for nml in nml['nml_river']])
        self.x = M.xc[self.river_locations]
        self.y = M.yc[self.river_locations]
        self.river_names = np.array([nml['river_name'] for nml in nml['nml_river']])

        # Load river transport and temperature for each of the rivers
        # --> Can we naively assume that the rivers follow the same sequence?
        self.transport  = []
        self.river_temp = []