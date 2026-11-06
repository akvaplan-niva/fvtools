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
        # Small meshes need not contain any nodes in the transition annulus.
        if len(self.transition_nodes) == 0:
            weight[self.nestingzone_nodes] = 1.0
            return weight
        if len(self.nestingzone_nodes) == 0:
            raise ValueError('No FVCOM nodes lie inside the ROMS nesting zone')
        R_outer_edge = np.max(self.Rdst[self.transition_nodes])

        # Compute weights in the transition zone
        width_transition = self.Rdst[self.nestingzone_nodes].max() - R_outer_edge
        a = 1.0/width_transition
        b = R_outer_edge/width_transition
        weight[self.transition_nodes]  = a*self.Rdst[self.transition_nodes] - b

        # Just to make sure that the nestingzone is 1
        weight[self.nestingzone_nodes] = 1
        return weight

    def _blend_coarse_depth(self, coarse_values, *, fallback_invalid=False):
        """Validate interpolated depth before changing the FVCOM grid.

        GLORYS historically retains original depth at missing source values.
        ROMS must not silently create an unmatched nesting bathymetry.
        """
        indices = self.nodes_to_change
        old_h = np.asarray(self.M.h)
        coarse_values = np.asarray(coarse_values)
        if coarse_values.shape != (len(indices),):
            raise ValueError('Interpolated bathymetry has wrong number of FVCOM nodes')
        if not np.all(np.isfinite(old_h)):
            raise ValueError('FVCOM bathymetry contains nonfinite values')
        local_weight = self.weight[indices]
        invalid = ~np.isfinite(coarse_values)
        if not fallback_invalid and np.any(invalid & (local_weight > 1e-12)):
            raise ValueError('Nonfinite ROMS bathymetry within the active nesting/transition zone')
        # Also replace invalid values at zero weight: NaN * 0 is still NaN.
        coarse_values = np.where(invalid, old_h[indices], coarse_values)
        h_coarse = old_h.copy()
        h_coarse[indices] = coarse_values
        updated = h_coarse * self.weight + old_h * (1 - self.weight)
        if not np.all(np.isfinite(updated)):
            raise ValueError('Nonfinite FVCOM bathymetry after MatchTopo blending')
        self.M.h = updated

    def add_GLORYS_bathymetry_to_FVCOM_and_NEST(self):
        """Use ROMS MatchTopo blending with GLORYS wet-cell bathymetry.

        The source bathymetry is the deepest wet GLORYS z-level, NOT the
        underlying high-resolution real seabed. An active nest must never
        silently retain old FVCOM depth when the GLORYS source is invalid.
        """
        if self.M.info.get('true if updated by glorys', False):
            print('  - GLORYS bathymetry already matched; reusing stored grid')
            self.match_depth_in_nest_and_model()
            self.NEST.inherit_sig(self.M)
            return self.M, self.NEST
        if self.M.info.get('true if updated by roms', False):
            raise ValueError('M.npy is already marked as ROMS-matched. '
                             'Start with the original BuildCase grid before matching GLORYS.')

        from fvtools.interpolators.structured_bilinear import structured_bilinear_weights
        wet = ~np.asarray(self.COARSE.land_mask, dtype=bool)
        if wet.shape != self.COARSE.lon.shape:
            raise ValueError('GLORYS wet mask and lon/lat grid have different shapes')
        xx, yy = self.M.Proj(self.COARSE.lon, self.COARSE.lat)
        xy_source = np.column_stack((xx[wet], yy[wet]))
        pts = np.column_stack((self.M.x[self.nodes_to_change],
                               self.M.y[self.nodes_to_change]))
        ids, w = structured_bilinear_weights(xy_source, wet, pts)
        h = np.asarray(self.COARSE.h_rho)[wet]
        coarse = np.sum(h[ids] * w, axis=1)
        self._blend_coarse_depth(coarse)
        self._update_depth_and_sigma(source='glorys')
        return self.M, self.NEST

    def add_ROMS_bathymetry_to_FVCOM_and_NEST(self):
        '''
        Add ROMS bathymetry to the nestingzone, create a smooth transition from ROMS bathymetry to FVCOM bathymetry. 
        Writes new bathymetry to _dpt.dat file and updates M.npy
        '''
        # No need to re-do this step if we already have done it...
        try:
            if self.M.info['true if updated by roms'] == True:
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

        # Validate ROMS bathymetry before mutating M.h or writing input files.
        coarse = np.sum(self.COARSE.h_rho[self.COARSE.fv_rho_mask][N4B.rho_index] * N4B.rho_coef, axis=1)
        self._blend_coarse_depth(coarse)

        # Update and save changes to the bathymetry and vertical grid
        self._update_depth_and_sigma()

        return self.M, self.NEST

    def _update_depth_and_sigma(self, source='roms'):
        # Souza sigma can depend on bathymetry; calculate it after matching.
        self.M.read_sigma()
        self.match_depth_in_nest_and_model()
        self.NEST.inherit_sig(self.M)
        self.store_bathymetry(source=source)

    def match_depth_in_nest_and_model(self):
        '''
        Ensures that the nest and the mother model has the same bathymetry
        '''
        ind = self.M.find_nearest(self.NEST.x, self.NEST.y)
        self.NEST.h = self.M.h[ind]

    def store_bathymetry(self, source='roms'):
        """Persist matched FVCOM depth and record WHICH mother model matched it."""
        if source not in {'roms', 'glorys'}:
            raise ValueError(f'Unknown bathymetry mother model: {source}')
        self.M.write_bath(
            filename=f"./input/{self.M.info['casename']}_dep.dat",
            latlon=self.latlon)
        print('- Updating M.npy with the new bathymetry')
        self.M.info[f'true if updated by {source}'] = True
        self.M.to_npy()
