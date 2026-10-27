"""FVCOM river forcing assembly, mesh assignment, and output."""
import os
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy import interpolate
from scipy.spatial import cKDTree as KDTree

import fvtools.grid.grid_metrics as gm
from fvtools.prepro._buildrivers.constants import version
from fvtools.prepro._buildrivers.interaction import CropRivers


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
        plt.show(block = False)

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
        print(f'Writing input/riverdata.nc')
        with netCDF4.Dataset('input/riverdata.nc', 'w') as d:
            # Set dimensions
            d.createDimension('time', None)
            d.createDimension('rivers', len(self.xriv))
            d.createDimension('DateStrLen', 26)
            d.createDimension('namelen', 80)
    
            # Add netcdf information
            d.source      = 'Akvaplan-niva BuildRiver, version '+str(version)
            try:
                user = os.getlogin()
            except:
                try:
                    user = os.getcwd().split('home/')[1].split('/')[0]
                except:
                    user = 'unknown user'

            
            d.history     = f'Created {datetime.now().strftime('%Y-%m-%d at %H:%M h')} by  {user}'
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

    def write_namelist(self, namelist = 'input/RiverNamelist.nml', riverfile = 'riverdata.nc'):
        """
        Write a namelist to accompany the netCDF file
        """
        difference = np.round(-np.diff(self.M.siglev_c, axis = 1), 6)
        with open(namelist, 'w+') as f:
            print(f'  - Writing namelist for {len(self.river_names)} rivers')
            for i, river in enumerate(self.river_names):
                river = self.fix_nordic(river)
                f.write(' &NML_RIVER\n')
                f.write(f" RIVER_NAME = '{river}',\n")
                f.write(f" RIVER_FILE = '{riverfile}',\n")
                f.write(f' RIVER_GRID_LOCATION = {self.mesh_location[i]+1},\n')
                vertical_dist = np.array2string(
                        difference[self.mesh_location[i],:], 
                        separator = ' ', 
                        edgeitems = 6,
                        precision = 5, 
                        floatmode = 'fixed'
                    ).replace('\n','')[1:-1]
                f.write(f' RIVER_VERTICAL_DISTRIBUTION = {vertical_dist}\n')
                f.write('/\n')
