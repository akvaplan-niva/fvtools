import numpy as np
from functools import cached_property
from scipy.spatial import cKDTree as KDTree
from fvtools.interpolators.roms_interpolators import N4ROMSNESTING
from fvtools.interpolators.glorys_interpolators import N4GLORYS


class MatchTopo:
    def __init__(self, COARSE, M, NEST, latlon):
        '''
        Rutine matching COARSE depth with FVCOM depth, same-same in nestingzone,
        but a smooth transition from ROMS to BuildCase topo on the outside of it
        '''
        self.M = M 
        self.COARSE = COARSE
        self.NEST = NEST
        self.latlon = latlon

    @property
    def R_edge_of_nestingzone(self):
        '''
        R_edge_of_nestingzone is the radius from obc nodes where we will modify the bathymetry to be identical to ROMS
        '''
        return 2*self.NEST.R

    @property
    def R_edge_of_smoothing_zone(self):
        '''
        Distance from the OBC where we will use the bathymetry computed by BuildCase
        '''
        return 6*self.NEST.R
    
    @cached_property
    def Rdst(self):
        '''
        The distance each node in the mesh is from the OBC
        '''
        obc_tree = KDTree(np.array([self.M.x[self.M.obc_nodes], self.M.y[self.M.obc_nodes]]).transpose())
        Rdst, _ = obc_tree.query(np.array([self.M.x, self.M.y]).transpose())
        return Rdst

    @property
    def nestingzone_nodes(self):
        '''
        Nodes where we will set h equal to h_roms
        '''
        return np.where(self.Rdst <= self.R_edge_of_nestingzone)[0]

    @property
    def transition_nodes(self):
        '''
        Nodes where we linearly (based on distance from the nestingzone) transition from ROMS depth to FVCOM depth
        '''
        return np.where(np.logical_and(self.Rdst > self.R_edge_of_nestingzone, self.Rdst <= self.R_edge_of_smoothing_zone))[0]

    @property
    def nodes_to_change(self):
        '''
        All nodes in the mesh where we will change the bathymetry
        '''
        return np.where(self.Rdst <= self.R_edge_of_smoothing_zone)[0]

    @cached_property
    def weight(self):
        '''
        weights for creathing a smooth bathymetry transition from ROMS to FVCOM near the nestingzone
        '''
        weight = np.zeros(self.M.x.shape)
        R_outer_edge = np.max(self.Rdst[self.transition_nodes])

        # Compute weights in the transition zone
        width_transition = self.Rdst[self.nestingzone_nodes].max() - R_outer_edge
        a = 1.0/width_transition
        b = R_outer_edge/width_transition
        weight[self.transition_nodes]  = a*self.Rdst[self.transition_nodes] - b

        # Just to make sure that the nestingzone is 1
        weight[self.nestingzone_nodes] = 1
        return weight

    def add_GLORYS_bathymetry_to_FVCOM_and_NEST(self):
        '''
        Adapted version of the roms routine
        '''
        # No need to re-do this step if we already have done it...
        try:
            if self.M.info['true if updated by MatchTopo'] == True:
                print('  - This experiment has already been matched with GLORYS bathymetry, skipping step')
                self.match_depth_in_nest_and_model()
                return self.M, self.NEST
        except:
            pass

        # Prepare interpolator
        N4B = N4GLORYS(
            self.COARSE, 
            x = self.M.x[self.nodes_to_change], 
            y = self.M.y[self.nodes_to_change], 
            uv = False, 
            land_check = False,
            proj = self.M.Proj
            ) # the depth at ROMS land is equal to min_depth        
        N4B.nearest4()

        # Make copy of FVCOM bathymetry, set depth in the "to change range" equal to GLORYS bathy
        h_coarse = np.copy(self.M.h)
        
        # Remove nans from the bathymetry, and smooth the transition from old to new
        h_coarse[self.nodes_to_change] = np.sum(self.COARSE.h_rho.ravel()[N4B.rho_index]*N4B.rho_coef, axis=1)
        h_coarse[np.isnan(h_coarse)] = np.copy(self.M.h[np.isnan(h_coarse)])


        # Update the nodes by to their distance from the obc
        self.M.h  = h_coarse*self.weight + self.M.h*(1 - self.weight)
        self.match_depth_in_nest_and_model()

        # Store the smoothed bathymetry to the mesh (ends up in casename_dep.dat)
        self.store_bathymetry()
        return self.M, self.NEST

    def add_ROMS_bathymetry_to_FVCOM_and_NEST(self):
        '''
        Add ROMS bathymetry to the nestingzone, create a smooth transition from ROMS bathymetry to FVCOM bathymetry. 
        Writes new bathymetry to _dpt.dat file and updates M.npy
        '''
        # No need to re-do this step if we already have done it...
        try:
            if self.M.info['true if updated by roms_nesting_fg'] == True:
                print('  - This experiment has already been matched with ROMS bathymetry, skipping step')
                self.match_depth_in_nest_and_model()
                return self.M, self.NEST
        except:
            pass

        # Prepare interpolator
        N4B = N4ROMSNESTING(
            self.COARSE, 
            x = self.M.x[self.nodes_to_change], 
            y = self.M.y[self.nodes_to_change], 
            uv = False, 
            land_check = False
            ) # the depth at ROMS land is equal to min_depth
        N4B.nearest4()

        # Make copy of FVCOM bathymetry, set depth in the "to change range" equal to ROMS bathy
        h_coarse = np.copy(self.M.h)
        h_coarse[np.isnan(h_coarse)] = self.M.h[np.isnan(h_coarse)]
        h_coarse[self.nodes_to_change] = np.sum(self.COARSE.h_rho[self.COARSE.fv_rho_mask][N4B.rho_index] * N4B.rho_coef, axis=1)

        # Update the nodes by to their distance from the obc
        self.M.h  = h_coarse*self.weight + self.M.h*(1 - self.weight)
        self.match_depth_in_nest_and_model()

        # Store the smoothed bathymetry to the mesh (ends up in casename_dep.dat)
        self.store_bathymetry()
        return self.M, self.NEST

    def match_depth_in_nest_and_model(self):
        '''
        Ensures that the nest and the mother model has the same bathymetry
        '''
        ind = self.M.find_nearest(self.NEST.x, self.NEST.y)
        self.NEST.h = self.M.h[ind]

    def store_bathymetry(self):
        '''
        First to FVCOM readable input file, then update the M.npy file with the correct bathymetry
        '''
        self.M.write_bath(filename = f"./input/{self.M.info['casename']}_dep.dat", latlon = self.latlon)
        print('- Updating M.npy with the new bathymetry')
        self.M.info['true if updated by roms_nesting_fg'] = True # for later
        self.M.to_npy()