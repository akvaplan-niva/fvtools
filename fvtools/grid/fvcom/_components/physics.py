"""FVCOM projected/geographic angles and physical diagnostics."""

import numpy as np
import matplotlib.pyplot as plt


class AnglesAndPhysics:
    '''
    Angles you need when rotating velocities from a UTM coordinate system to WGS84 (true north/east)
    '''
    @property
    def node_utm_angle(self):
        '''radians offset from true north in utm coordinates'''
        return self._get_utm_angle(self.x, self.y, self.lon, self.lat)

    @property
    def cell_utm_angle(self):
        '''radians offset from true north in utm coordinates'''
        return self._get_utm_angle(self.xc, self.yc, self.lonc, self.latc)

    def _get_utm_angle(self, x, y, lon, lat, dx = 100):
        '''
        utm coordinate offset of true north measured in radians

        Consider two coordinate systems (x,y), (x', y'). We want to convert vector quantities from one to the other,
        and to do so we find the angle between x and x'. In our case, we say that (x, y) are coordinates in WGS84 and (x', y') are any UTM coordinates.

        For a small displacement dx' in the UTM system, we move a distance dx and dy in the lat/lon system, R is earths radii:
        dx = R * cos(lat) * dLon/dx' * dx'
        dy = R * dLat/dx' * dx'
        angle = atan2(dx, dy)

        Use alpha rotate the coordinate system so that vec_true = (v_x, v_y)
        v_x = v_x'*cos(alpha) - v_y'*sin(alpha)
        v_y = v_x'*sin(alpha) + v_y'*cos(alpha)
        '''
        londx, latdx = self.Proj(x+dx, y, inverse = True)
        dLat_dx = (latdx-lat)/dx # distance made in lat/lon system when incrementing dx
        dLon_dx = ((londx-lon)/dx)*np.cos(lat*(2*np.pi/360)) 
        return np.arctan2(dLat_dx, dLon_dx)

    def rotate_vector_to_utm(self, u, v, angle):
        '''Rotate a vector from lat/lon to utm'''
        u_new =  u*np.cos(angle) + v * np.sin(angle)
        v_new = -u*np.sin(angle) + v * np.cos(angle)
        return u_new, v_new

    def rotate_vector_from_utm(self, u, v, angle):
        '''Rotate a vector to lat/lon from utm'''
        u_new =  u*np.cos(angle) - v * np.sin(angle)
        v_new =  u*np.sin(angle) + v * np.cos(angle)
        return u_new, v_new

    def get_coriolis(self, cell = False):
        '''Returns coriolis parameter at grid points'''
        if cell:
            self.f = np.sin(self.latc*2*np.pi/360)*4*np.pi/(24*60*60)
        else:
            self.f = np.sin(self.lat*2*np.pi/360)*4*np.pi/(24*60*60)

    def get_cfl(self, z = 1.5, u = 3.0, g = 9.81, verbose = True):
        '''
        Estimate the CFD timestep
        '''
        # Length of each triangle side
        lAB, lBC, lCA = self._get_sidewall_lengths()
        dpt      = np.max(self.h[self.tri], axis = 1) + z
        cg_speed = np.sqrt(dpt*g) + u
        ts_AB, ts_BC, ts_CA = np.array(lAB/cg_speed), np.array(lBC/cg_speed), np.array(lCA/cg_speed)
        ts_walls = np.array([ts_AB, ts_BC, ts_CA])
        ts_min   = np.min(ts_walls, axis = 0)/np.sqrt(2) # to adjust for transverse wave propagation

        if verbose:
            print(f'  - Required timestep approx: {min(ts_min):.2f} s')
            plt.figure()
            contour = self.plot_contour(ts_min, cmap = 'jet', levels = np.linspace(min(ts_min), 3*min(ts_min), 20), extend = 'max', show = False)
            plt.colorbar(contour, label = 's')
            plt.axis('equal')
        self.ts = ts_min
