from .riverpositions import SmallRivers, LargeRivers
from .rivertemp import RiverTemperatures

import sys
import os
import chardet
import netCDF4
import numpy as np
import fvtools.grid.grid_metrics as gm
import matplotlib.pyplot as plt
import pandas as pd

from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.tools import num2date, date2num
from scipy.spatial import cKDTree as KDTree
from scipy.io import loadmat
from scipy import interpolate
from scipy.signal import filtfilt
from datetime import datetime, timedelta, timezone

from fvtools.plot.geoplot import geoplot


def old_main(start, stop, vassdrag, mesh_dict = 'M.npy', info = None, temp = None):
    """
    BuildRiver use data from the NVE and feeds all the mapped rivers leading to the ocean

    Parameters:
    ----
    start:     yyyy-mm-dd
    stop:      yyyy-mm-dd
    vassdrag:  [233, 234] etc, any array containing all ids (integers) will do.
    temp:      Compile new temperatures.npy file, or use an existing one.
               ----
                Ideal use of this is to compile a new file (temp = 'compile') for mother-FVCOM models, and temp = 'PO10_temperatures.npy' for
                the smaller models you later nest into the mother run.

               temp = 'compile'
                - If you are running a large scale (nested to eg. NorShelf), you should compile a new *_temperatures.npy file by setting temp = None.
                  After this, it will return a "casename"_temperatures.npy file to the RiverTemperatures folder
                  --> On Stokes: /data/FVCOM/Setup_Files/Rivers/Raw_Temperatures/
                  --> On Betzy:  /cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/Raw_Temperatures/

               temp = '/data/FVCOM/Setup_Files/Rivers/Raw_Temperatures/PO10_temperatures.npy'
                - Giving the temperature string will let BuildRivers know that this is the pre-compiled
                  river temperatures file you want to use.
                  --> This option was included specifically for fvcom2fvcom nested runs.

    Optional:
    ----
    mesh_dict: (M.npy by default)
    info:      Dict where all paths and river specific settings are stored. Change the basic settings by giving it as
               an input: info. Get the basic settings by calling BuildRivers.get_input(),
               which can then be edited and passed back to main if other paths etc. are
               needed in the given experiment.

    hes@akvaplan.niva.no
    """
    if temp is None:
        raise InputError('You must decide whether to compile a new *_temperatures.npy or use an existing one. See docstring for instructions.')

    M = FVCOM_grid(mesh_dict)

    print('----------------------------------------------------------------------------')
    print(f'                       BuildRivers: {M.casename}')
    print('----------------------------------------------------------------------------')

    if info is None:
        info = get_input()

    if temp != 'compile':
        info['rivertemp']     = temp
        info['compile river'] = False

    M.re_project(info['river_projection'])
    M.get_cfl(verbose=False)

    # Initialize the object that will move rivers to the mesh and ensure numerical stability
    print('Identify FVCOM land:')
    Forcing = FVCOM_rivers(info, M, vassdrag)

    print('\nRunoff data')
    Runoff  = RiverRunoff(info)

    if info['rivertemp'] != 'compile':
        print(f'\nRiver temperature from {info["rivertemp"]}')
    Temp    = RiverTemperatures(info, vassdrag, M.casename, start)

    # Load river positions
    print('\nGet river positions')
    Large   = LargeRivers(info)
    Small   = SmallRivers(info)

    # Remove vassdrags that are not part of our domain
    print('- Connect rivers to nedborfelt')
    Large.connect_nedborsfelt(vassdrag)
    Small.connect_nedborsfelt(vassdrag)

    # Crop the large and small datasets to our vassdrag
    print('- Crop rivers to the vassdrag')
    Large.crop_to_vassdrag()
    Small.crop_to_vassdrag()

    # Remove the rivers that are too far away from land, and too close to the obc
    print('- Crop river to a distance from the obc')
    Small = Forcing.crop_river_to_obc(Small)
    Large = Forcing.crop_river_to_obc(Large)

    # Add temperatures to the small and large rivers
    print('- Add temperatures to the rivers')
    Small.add_temperature(Temp)
    Large.add_temperature(Temp)

    # --> Any other tracer that is released via the river runoff can be added here.

    # Re-distribute the runoff according to catchment area
    print('- Adjust runoff over rivers according to catchment area')
    Forcing.redistribute_runoff(Small, Large, Runoff)

    # Combine small and large rivers to the same list
    print('- Combine large and small rivers')
    Forcing.combine_small_and_large(Large, Small)

    # Trim the forcing to fit the desired start and stop time
    print('- Trim the forcing to fit with start and stop time')
    Forcing.make_time(start, stop, Runoff, Temp)

    # Find edge to connect outflow to. Distribute across rivers.
    print('- Connect the rivers to the FVCOM mesh')
    gp = geoplot(M.x, M.y, projection = M.reference)
    Forcing.connect_to_mesh(gp)

    # Show the variables
    print('\nFinished, plotting the forcing')
    show_forcing(Forcing, gp, M)

    # Write to netCDF, write RiverNamelist.nml
    Forcing.dump()
    Forcing.write_namelist()

# ---

class RiverRunoff:
    """
    Load river runoff files, prepare to be used by the routine
    """
    def __init__(self, info):
        """
        riverdata.dat files contain all the necessary information about river runoff from "Vassdrag".
        See atlas.nve.no for more information about the whats and wheres of Vassdrag.
        """
        self.info = info
        self.pathToRiver = self.info['vassdrag']
        print(f'- {self.pathToRiver}')
        Q = self.load_data()
        self.convert_dates(Q)
        self.transport = Q[:, 3:] # load the transport pr. "vassdragsområde"

    def load_data(self):
        '''
        get the time format as numbers
        '''
        Q = np.loadtxt(self.info['vassdrag'])
        return Q

    def convert_dates(self,Q):
        self.dates = []
        for i in range(Q.shape[0]):
            self.dates.append(datetime(int(Q[i,0]), int(Q[i,1]), int(Q[i,2]), tzinfo = timezone.utc))

class FVCOM_rivers:
    """
    Class storing data that eventually ends up as FVCOM forcing
    """
    def __init__(self, info, M, vassdrag):
        self.info, self.vassdrag, self.M = info, vassdrag, M
        self.nodes, self.cells = gm.get_nbe(self.M)

    @property
    def xy_land(self):
        if self.info['iloc'] == 'edge':
            x_land  = self.M.xc[self.cells['boundary'][np.where(self.cells['id']==1)[0]]]
            y_land  = self.M.yc[self.cells['boundary'][np.where(self.cells['id']==1)[0]]]

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
            x_obc   = self.M.xc[self.cells['boundary'][np.where(self.cells['id']==2)[0]]]
            y_obc   = self.M.yc[self.cells['boundary'][np.where(self.cells['id']==2)[0]]]

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

    def redistribute_runoff(self, Small, Large, Runoff):
        """
        Figure out how much water each river should discharge
        """
        # Get the volume transport through big rivers
        Fraction_Large       = Large.areal/Large.landareal
        self.Large_Runoff    = Runoff.transport[:, Large.Vl[:,0]-1]*Fraction_Large[:,0]
        self.Large_LongName  = [f'{nedbor.split(" ")[0]} - {name.split(" ")[0]}' for nedbor, name in zip(Large.nedborfelt, Large.name)]
        self.Large_ShortName = [nedbor.split(' ')[0] for nedbor in Large.nedborfelt]

        # Figure out how much area in each vassdrag is left for the small rivers, and return corresponding runoff
        Small_Runoff = []
        for vdrag in self.vassdrag:
            Fraction_Small  = 1.0 - np.sum(Fraction_Large[np.where(Large.Vl[:,0]==vdrag)[0]])
            Small_Runoff.append(Runoff.transport[:, vdrag-1]*Fraction_Small)

        Small_Runoff = np.array(Small_Runoff).T

        # Share the leftover the runoff among the small rivers
        self.Small_Runoff = np.empty((len(Small_Runoff[:,0]),0))
        for i, vdrag in enumerate(self.vassdrag):
            num_small         = len(np.where(Small.Vs == vdrag)[0])
            if num_small > 0:
                runoff_each_small = Small_Runoff[:,i]/num_small
                runoff_small_here = np.tile(runoff_each_small, (num_small,1)).T
                self.Small_Runoff = np.append(self.Small_Runoff, runoff_small_here, axis = 1)

        # Store the names
        self.Small_LongName  = [f'{vassdrag[0]}.Z-small-{i+1}' for i, vassdrag in enumerate(Small.Vs)]
        self.Small_ShortName = [f'{vassdrag[0]}.Z-s-{i+1}' for i, vassdrag in enumerate(Small.Vs)]

    def combine_small_and_large(self, Large, Small):
        """
        Prepare the vectors that will go to the output
        """
        if self.info['whichrivers'] == 'all':
            self.xriv = np.append(Large.x, Small.x)
            self.yriv = np.append(Large.y, Small.y)
            self.transport = np.append(self.Large_Runoff, self.Small_Runoff, axis = 1)
            self.river_names = self.Large_LongName  + self.Small_LongName
            self.short_names = self.Large_ShortName + self.Small_ShortName
            self.vassdrag = np.append(Large.Vl, Small.Vs)
            self.river_temp  = np.append(Large.river_temp, Small.river_temp, axis = 1)

        elif self.info['whichrivers'] == 'small':
            self.xriv = Small.x
            self.yriv = Small.y
            self.transport   = self.Small_Runoff
            self.river_names = self.Small_LongName
            self.short_names = self.Small_ShortName
            self.vassdrag = Small.Vs
            self.river_temp = Small.river_temp

        elif self.info['whichrivers'] == 'large':
            self.xriv = Large.x
            self.yriv = Large.y
            self.transport   = self.Small_Runoff
            self.river_names = self.Large_LongName
            self.short_names = self.Large_ShortName
            self.vassdrag    = Large.Vl
            self.river_temp  = Large.river_temp

        else:
            raise NameError(f'{self.info["whichrivers"]}" is not a supported whichrives-option, try "large", "small" or "all"')

    def make_time(self, start, stop, Runoff, Temp, dt = 3/24):
        '''
        Create a time-vector for the forcing file
        '''
        start_tuple = start.split('-')
        stop_tuple  = stop.split('-')
        self.start  = datetime(int(start_tuple[0]), int(start_tuple[1]), int(start_tuple[2]), tzinfo = timezone.utc)
        self.stop   = datetime(int(stop_tuple[0]), int(stop_tuple[1]), int(stop_tuple[2]), tzinfo = timezone.utc)

        # Convert to easy-to-deal-with time
        runoff_dates = np.array(netCDF4.date2num(Runoff.dates, units = 'days since 1858-11-17 00:00:00'))
        start_num    = netCDF4.date2num(self.start, units = 'days since 1858-11-17 00:00:00')
        stop_num     = netCDF4.date2num(self.stop, units = 'days since 1858-11-17 00:00:00')

        # Check if the time covers the model period
        # transport
        if stop_num > runoff_dates[-1]:
            raise ValueError(f'{self.info["vassdrag"]} does not extend to the stop date')

        elif start_num < runoff_dates[0]:
            raise ValueError(f'{self.info["vassdrag"]} starts after the start date')

        # temperature
        if start_num < Temp.river_time[0]:
            raise ValueError(f'All file(s) in {self.info["rivertemp"]} starts after start date')

        elif stop_num > Temp.river_time[-1]:
            raise ValueError(f'All file(s) in {self.info["rivertemp"]} ends before end date')

        # Prepare the output files
        self.model_time = np.arange(start_num, stop_num + dt, dt)

        # Interpolate to output structure
        self.RiverTransport = np.zeros((len(self.model_time), len(self.xriv)))
        self.RiverTemp      = np.zeros((len(self.model_time), len(self.xriv)))

        # These fields will be dumped to the model forcing file
        for i in range(len(self.xriv)):
            f_transport     = interpolate.interp1d(runoff_dates, self.transport[:,i])
            f_temperature   = interpolate.interp1d(Temp.river_time, self.river_temp[:,i])
            self.RiverTransport[:,i]  = f_transport(self.model_time)
            self.RiverTemp[:,i]       = f_temperature(self.model_time)

    def crop_river_to_obc(self, river_object):
        """
        Figure out which model point each river is closest too
        """
        # Remove rivers too far away from land
        d, land_ind  = self.land_tree.query(np.array([river_object.x, river_object.y]).transpose())
        close_enough = np.where(d<=self.info['dRmax'])[1] # Remove rivers too far away from land
        river_object = crop_object(river_object, close_enough) # Crop the river object

        # Remove rivers that are too close to the OBC
        d, _   = self.obc_tree.query(np.array([self.x_land[land_ind[0,close_enough]], \
                                                     self.y_land[land_ind[0,close_enough]]]).transpose())
        far_enough   = np.where(d>=self.info['dRmax'])[0] # Remove rivers too far away from obc
        river_object = crop_object(river_object, far_enough)

        return river_object

    def connect_to_mesh(self, gp):
        """
        Figure out which node/cell the flux should go to.
        """
        first = True
        while True:
            _, land_loc = self.land_tree.query(np.array([self.xriv, self.yriv]).transpose())
            if first:
                self.river_connection(land_loc, gp)
                first = False

            d, mesh_location = self.mesh_tree.query(np.array([self.x_land[land_loc], self.y_land[land_loc]]).transpose())
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
        d, land_loc      = self.land_tree.query(np.array([self.xriv, self.yriv]).transpose())
        d, mesh_location = self.mesh_tree.query(np.array([self.x_land[land_loc], self.y_land[land_loc]]).transpose())
        self.mesh_location = mesh_location
        plt.scatter(self.xriv, self.yriv, s = 50, c = 'm', label = 'final river nodes')
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
            d, this_land  = self.land_tree.query(np.array([self.xriv[river], self.yriv[river]]).transpose(), k = n_newland)

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
                self.RiverTransport = np.append(self.RiverTransport, transport[:,None], axis = 1)
                self.RiverTemp = np.append(self.RiverTemp, temp[:,None], axis = 1)
                self.river_names.append(f'{long_name}-p{i}')
                self.short_names.append(f'{short_name}-p{i}')

            # Update river location
            self.xriv = np.append(self.xriv, new_x)
            self.yriv = np.append(self.yriv, new_y)

    def river_connection(self, land_loc, gp):
        """
        Show how far rivers have been moved from NVE location
        """
        plt.figure()
        plt.imshow(gp.img, extent=gp.extent)
        # Make lines connecting rivers to their FVCOM location
        xvec = np.array([self.x_land[land_loc], self.xriv]).transpose()
        yvec = np.array([self.y_land[land_loc], self.yriv]).transpose()
        xvec_nan = np.insert(xvec, 2, np.nan, axis = 1).ravel()
        yvec_nan = np.insert(yvec, 2, np.nan, axis = 1).ravel()

        plt.plot(self.x_land, self.y_land, 'b.', label = 'land')
        plt.plot(self.x_land[land_loc], self.y_land[land_loc], 'k.', label = 'land with river')
        plt.plot(self.xriv, self.yriv, 'g.',     label = 'river location from NVE')
        plt.plot(xvec_nan, yvec_nan, 'r',        label = 'vector from NVE location to FVCOM location')
        plt.axis('equal')

    def dump(self):
        """
        Write the riverdata.nc file for river forcing
        """
        # Initialize file
        d = netCDF4.Dataset('riverdata.nc', 'w')

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
        _names = []
        names._Encoding = 'ascii'
        for i, name in enumerate(self.river_names):
            if len(name) > 80:
                this_name = name[:80]
            else:
                this_name  = name + (80-len(name))*' '
            this_name = self.fix_nordic(this_name)
            names[i,:] = np.array(this_name, dtype = 'S80')

        d.close()

    def fix_nordic(self, this_name):
        '''
        FVCOM does not accept norwegian letters
        '''
        this_name.replace('å','a')
        this_name.replace('Å','A')
        this_name.replace('ø','o')
        this_name.replace('Ø','O')
        this_name.replace('æ','e')
        this_name.replace('Æ','E')
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


# Crop the fields in an object to only cover indices
# ----
def crop_object(obj, indices):
    keys = obj.__dict__.keys()
    for key in keys:
        var = getattr(obj,key)
        if key == 'rivers_in_vassdrag':
            continue
        if type(var) == str:
            continue
        setattr(obj, key, var[indices])
    return obj

# Show what we will write to the riverdata forcing
# ----
def show_forcing(obj, gp, M):
    """
    Simple figures to see that the routine got the basics right
    """
    plt.figure()
    try:
        M.georeference()
    except:
        plt.imshow(gp.img, extent = gp.extent)
    plt.plot(obj.x_land, obj.y_land, 'g.', label = 'land nodes', zorder = 1)
    plt.scatter(obj.xriv, obj.yriv, np.mean(obj.RiverTransport, axis = 0), c = np.mean(obj.RiverTransport, axis = 0), zorder = 5)
    plt.title('Average transport')
    plt.axis('equal')
    plt.colorbar(label = r'm$^3$ s$^{-1}$')
    plt.show(block = False)

    plt.figure()
    try:
        M.gereference()
    except:
        plt.imshow(gp.img, extent = gp.extent)
    plt.plot(obj.x_land, obj.y_land, 'g.', label = 'land nodes', zorder = 1)
    plt.scatter(obj.xriv, obj.yriv, obj.RiverTemp.max(axis = 0), c = obj.RiverTemp.max(axis = 0), cmap = 'inferno', zorder = 5)
    plt.title('Max temperature in model period')
    plt.axis('equal')
    plt.colorbar(label = 'degrees celcius')
    plt.show(block = False)

class InputError(Exception): pass