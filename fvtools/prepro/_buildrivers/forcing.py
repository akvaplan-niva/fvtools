"""FVCOM river forcing assembly, mesh assignment, and output."""
import os
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy import interpolate
from scipy.spatial import cKDTree as KDTree

import fvtools.grid.fvcom.grid_metrics as gm
from fvtools.prepro._buildrivers.constants import version
from fvtools.prepro._buildrivers.interaction import CropRivers
from fvtools.prepro._buildrivers.output import RiverOutputMixin


class FVCOM_rivers(RiverOutputMixin, CropRivers):
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
        """Merge, distribute and validate FVCOM river outlets on the mesh.

        Each iteration uses the current outlet coordinates. The iteration limit
        and repeated-state check prevent a silent endless split/merge cycle
        when the coastline cannot accommodate the required runoff.
        """
        threshold = float(self.info['minrcoef'])
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('minrcoef must be a positive finite number')

        if not len(self.xriv):
            self.mesh_location = np.empty(0, dtype=np.int64)
            self.unique_mesh = self.mesh_location.copy()
            return

        shown_connection = False
        seen_states = set()
        max_iterations = 25

        for iteration in range(max_iterations):
            land_tree = self.land_tree
            _, land_loc = land_tree.query(np.column_stack((self.xriv, self.yriv)))
            land_loc = np.atleast_1d(land_loc).astype(np.intp)
            if not shown_connection:
                self.river_connection(land_loc, M,
                                      self.info['river_projection'],
                                      self.info['grid_projection'])
                shown_connection = True

            targets = np.column_stack((self.x_land[land_loc],
                                       self.y_land[land_loc]))
            _, mesh_location = self.mesh_tree.query(targets)
            self.mesh_location = np.atleast_1d(mesh_location).astype(np.intp)

            print('- merge rivers that go to the same mesh point')
            (self.RiverTransport, self.RiverTemp,
             self.river_names, self.short_names) = self.merge_rivers()
            # merge_rivers changes the number of columns; keep location IDs
            # aligned with the merged output even on the final iteration.
            self.mesh_location = self.unique_mesh.copy()
            rcoef = self.river_stability()
            bad_rivers = np.flatnonzero(rcoef > threshold)
            if bad_rivers.size == 0:
                break

            # A repeated aggregate state implies that splitting is simply
            # followed by merging back into the same unstable outlets.
            signature = (tuple(int(v) for v in self.unique_mesh),
                         self.RiverTransport.tobytes())
            if signature in seen_states:
                raise RuntimeError(
                    'River splitting did not converge: repeated mesh assignment. '
                    'Increase the available coastline or review minrcoef.'
                )
            seen_states.add(signature)

            print('- split rivers that need to be distributed over larger areas')
            self.split_problematic_river(bad_rivers, rcoef)
        else:
            raise RuntimeError(
                f'River splitting did not converge after {max_iterations} iterations. '
                'Inspect available coastline nodes and minrcoef.'
            )

        # After merging, xriv/yriv and mesh_location already refer to the
        # chosen mesh outlets. Avoid a redundant nearest-neighbour query.
        plt.scatter(self.xriv, self.yriv, s=50, c='m',
                    label='final river nodes',
                    transform=ccrs.epsg(int(
                        self.info['river_projection'].split(':')[-1])))
        plt.legend()
        plt.show(block=False)

    def merge_rivers(self):
        """Merge flux-conservatively, using flow-weighted temperatures.

        A zero-flow outlet has no advected heat. For a reproducible finite
        temperature value, retain the historical arithmetic-mean fallback.
        """
        self.unique_mesh = np.unique(self.mesh_location)
        flux_in = np.asarray(self.RiverTransport, dtype=float)
        temp_in = np.asarray(self.RiverTemp, dtype=float)
        if flux_in.ndim != 2 or temp_in.shape != flux_in.shape:
            raise ValueError('RiverTransport and RiverTemp must have matching 2-D shapes')
        if flux_in.shape[1] != len(self.mesh_location):
            raise ValueError('River columns must match mesh_location')
        if not np.all(np.isfinite(flux_in)) or np.any(flux_in < 0):
            raise ValueError('RiverTransport must be finite and nonnegative')
        if not np.all(np.isfinite(temp_in)):
            raise ValueError('RiverTemp must contain finite temperatures')

        transport = np.zeros((flux_in.shape[0], len(self.unique_mesh)))
        temperature = np.zeros_like(transport)
        names, short_names = [], []
        for i, mesh_id in enumerate(self.unique_mesh):
            places = np.flatnonzero(self.mesh_location == mesh_id)
            q = flux_in[:, places]
            t = temp_in[:, places]
            transport[:, i] = q.sum(axis=1)
            valid = transport[:, i] > 0
            if np.any(valid):
                numerator = np.sum(q * t, axis=1)
                temperature[valid, i] = numerator[valid] / transport[valid, i]
            if np.any(~valid):
                temperature[~valid, i] = t[~valid].mean(axis=1)
            names.append(', '.join(str(self.river_names[k]) for k in places))
            short_names.append(', '.join(str(self.short_names[k]) for k in places))

        if self.info['iloc'] == 'edge':
            self.xriv = self.M.xc[self.unique_mesh]
            self.yriv = self.M.yc[self.unique_mesh]
        elif self.info['iloc'] == 'node':
            self.xriv = self.M.x[self.unique_mesh]
            self.yriv = self.M.y[self.unique_mesh]
        else:
            raise ValueError("iloc must be 'node' or 'edge'")

        return transport, temperature, names, short_names

    def river_stability(self):
        """Return the original FVCOM-based maximum flux ratio by outlet.

        Preserve the existing area approximation and internal time step, but
        validate mesh metrics rather than silently allowing NaN/Inf ratios.
        """
        if self.RiverTransport.shape[1] == 0:
            return np.empty(0, dtype=float)

        nearest_cell = np.atleast_1d(
            self.M.find_nearest(self.xriv, self.yriv, 'cell')
        ).astype(np.intp)
        tri_area = np.asarray(self.M.tri_area[nearest_cell], dtype=float)
        dt_internal = float(np.min(self.M.ts)) * float(self.info['Isplit'])

        if self.info['iloc'] == 'edge':
            depth = np.asarray(self.M.hc[self.unique_mesh], dtype=float)
        elif self.info['iloc'] == 'node':
            depth = np.asarray(self.M.h[self.unique_mesh], dtype=float)
        else:
            raise ValueError("iloc must be 'node' or 'edge'")
        depth = np.maximum(depth - float(self.info['tideamp']),
                           float(self.info['min_depth']))

        if (not np.isfinite(dt_internal) or dt_internal <= 0
                or np.any(~np.isfinite(tri_area)) or np.any(tri_area <= 0)
                or np.any(~np.isfinite(depth)) or np.any(depth <= 0)):
            raise ValueError('Invalid cell area, river depth or internal time step')
        if not np.all(np.isfinite(self.RiverTransport)):
            raise ValueError('RiverTransport contains non-finite fluxes')

        return dt_internal * self.RiverTransport.max(axis=0) / (depth * tri_area)

    def split_problematic_river(self, bad_rivers, rcoef):
        """Split all flagged columns atomically, conserving each time-step flux.

        Child rivers keep their parent's temperature and are assigned to
        distinct nearest mesh locations. A distribution that cannot be placed
        on enough distinct cells fails explicitly instead of cycling forever.
        """
        threshold = float(self.info['minrcoef'])
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('minrcoef must be a positive finite number')
        bad = np.unique(np.asarray(bad_rivers, dtype=np.int64))
        nrivers = self.RiverTransport.shape[1]
        ratios = np.asarray(rcoef, dtype=float)
        if ratios.shape != (nrivers,):
            raise ValueError('rcoef must contain one ratio for each river')
        if np.any(bad < 0) or np.any(bad >= nrivers):
            raise IndexError('bad_rivers contains an invalid river index')
        if np.any(~np.isfinite(ratios[bad])):
            raise ValueError('Cannot split a river with a non-finite rcoef')
        if bad.size == 0:
            return

        land_xy = np.column_stack((self.x_land, self.y_land))
        nland = len(land_xy)
        land_tree, mesh_tree = self.land_tree, self.mesh_tree
        bad_set = set(int(i) for i in bad)
        q_columns, t_columns, x_positions, y_positions = [], [], [], []
        names, short_names, target_ids = [], [], []

        if len(self.mesh_location) != nrivers:
            raise ValueError('mesh_location must align with the river columns')
        for i in range(nrivers):
            q = np.asarray(self.RiverTransport[:, i], dtype=float)
            t = np.asarray(self.RiverTemp[:, i], dtype=float)
            if i not in bad_set:
                q_columns.append(q.copy())
                t_columns.append(t.copy())
                x_positions.append(float(self.xriv[i]))
                y_positions.append(float(self.yriv[i]))
                names.append(self.river_names[i])
                short_names.append(self.short_names[i])
                target_ids.append(int(self.mesh_location[i]))
                continue

            nchildren = int(np.ceil(ratios[i] / threshold)) + 1
            if nchildren > nland:
                raise ValueError(
                    f'Cannot split river {self.river_names[i]!r} into {nchildren} '
                    f'outlets: only {nland} land locations exist.'
                )

            # Request only nearby candidates; increase k if several land
            # points project to the same FVCOM mesh node/cell.
            k = nchildren
            chosen = []
            while True:
                _, nearest = land_tree.query(
                    [self.xriv[i], self.yriv[i]], k=k)
                nearest = np.atleast_1d(nearest)
                candidates = land_xy[nearest]
                _, mesh_ids = mesh_tree.query(candidates)
                seen = set()
                chosen = []
                for land_index, mesh_id in zip(nearest,
                                               np.atleast_1d(mesh_ids)):
                    mid = int(mesh_id)
                    if mid not in seen:
                        seen.add(mid)
                        chosen.append((int(land_index), mid))
                        if len(chosen) == nchildren:
                            break
                if len(chosen) == nchildren or k == nland:
                    break
                k = min(nland, max(k + 1, 2 * k))

            if len(chosen) != nchildren:
                raise ValueError(
                    f'Cannot distribute river {self.river_names[i]!r}: '
                    f'{nchildren} distinct mesh outlets required, '
                    f'only {len(chosen)} available.'
                )
            # Allocate all columns in temporary lists: if any split fails,
            # none of the original river arrays are modified.
            for j, (land_index, mesh_id) in enumerate(chosen):
                q_columns.append(q / nchildren)
                t_columns.append(t.copy())
                target_ids.append(mesh_id)
                x_positions.append(float(land_xy[land_index, 0]))
                y_positions.append(float(land_xy[land_index, 1]))
                names.append(f'{self.river_names[i]}-p{j}')
                short_names.append(f'{self.short_names[i]}-p{j}')

        self.RiverTransport = np.column_stack(q_columns)
        self.RiverTemp = np.column_stack(t_columns)
        self.xriv = np.asarray(x_positions)
        self.yriv = np.asarray(y_positions)
        self.river_names = names
        self.short_names = short_names
        self.mesh_location = np.asarray(target_ids, dtype=np.intp)

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



