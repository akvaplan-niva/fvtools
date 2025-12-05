import numpy as np
import matplotlib.pyplot as plt
from fvtools.interpolators.nearest4 import N4

class N4GLORYS(N4):
    '''
    Object with indices and coefficients for ROMS to FVCOM interpolation using bilinear coefficients
    '''
    mother = 'GLORYS'
    def __init__(self, GLORYS, x = None, y = None, tri = None, uv = False, land_check = True, proj = None):
        '''
        Find the nearest 4 Glorys points to any FVCOM point (node and cell), calculate bilinear interpolation coefficients.
        '''
        self.x, self.y = x, y
        self.tri = tri
        self.uv = uv
        self.land_check = land_check
        self.GLORYS = GLORYS
        self.Proj = proj

    # Mesh details
    # ----
    @property
    def xc(self):
        '''
        Cell easterly position in FVCOM mesh
        '''
        return np.mean(self.x[self.tri], axis = 1)
    
    @property
    def yc(self):
        '''
        Cell northerly position in FVCOM mesh
        '''
        return np.mean(self.y[self.tri], axis = 1)

    @property
    def fv_nodes(self):
        '''
        Array of FVCOM cell positions
        '''
        return np.array([self.x, self.y]).transpose()

    @property
    def fv_cells(self):
        '''
        Array of FVCOM cell positions
        '''
        return np.array([self.xc, self.yc]).transpose()

    @property
    def data_points_lonlat(self):
        '''
        GLORYS node positions in WGS84 coordiantes
        '''
        return self.GLORYS.lonlat_data

    @property
    def center_points_lonlat(self):
        '''
        GLORYS cell positions in WGS84 coordiantes (in centre of grid cells)
        '''
        return self.GLORYS.lonlat_center

    @property
    def data_points(self):
        '''
        GLORYS node positions in same projection as FVCOM
        '''
        return np.array(self.Proj(*self.GLORYS.lonlat_data.T)).T

    @property
    def center_points(self):
        '''
        GLORYS cell positions in same projection as FVCOM
        '''
        return np.array(self.Proj(*self.GLORYS.lonlat_center.T)).T

    # Call N4s methods to find the nearest4
    # ----
    def nearest4(self, M = None):
        '''
        Create nearest four indices and weights for all of the fields
        '''
        # Compute interpolation coefficients for nearest4 interpolation
        try:
            self.rho_index, self.rho_coef = self.get_interpolation_matrices(
                xy_source = self.data_points,
                xy_source_center = self.center_points,
                xy_fvcom = self.fv_nodes,
                widget_title = 'rho'
                )
            
            if self.uv:
                self.uv_index, self.uv_coef = self.get_interpolation_matrices(
                    xy_source = self.data_points,
                    xy_source_center = self.center_points,
                    xy_fvcom = self.fv_cells,
                    widget_title = 'uv'
                    )

        except ValueError:
            self.domain_exception_plot(self.center_points)
            raise DomainError('Your FVCOM domain is outside of the ROMS domain, it needs to be changed.')

        # Check if your mesh covers ROMS land, if so kill the routine and force the user to change the grid
        self.check_if_GLORYS_land_in_FVCOM_mesh(M)

    # Quality control. We don't want to use land points, since that would mean that we extrapolate the velocity field from ROMS.
    # ----
    def check_if_GLORYS_land_in_FVCOM_mesh(self, M = None):
        '''
        Check if FVCOM covers ROMS land, if so return
        '''
        if not self.land_check: # return if told to not care about land
            return

        error_occured = False
        for field in ['scalar', 'uv']:
            indices = self.GLORYS.land_mask.ravel()[getattr(self, f'{field}_index')]
            if indices.any():
                # First plot, this one is just to plot the fvcom and GLORYS grid
                if not error_occured:
                    plt.figure()
                    plt.triplot(self.x, self.y, self.tri, c = 'k', label = 'FVCOM', zorder = 100)
                    error_occured = True
                    
                    # Plot all ROMS land points in the vicinity
                    plt.plot(self.data_points[:, 0], self.data_points[:, 1], 'b.', zorder = 5)
                    plt.axis('equal')
                
                    # Plot all ROMS land points in the vicinity
                    plt.plot(
                        self.data_points[self.GLORYS.land_mask.ravel(), 0], 
                        self.data_points[self.GLORYS.land_mask.ravel(), 1],
                        'g.', 
                        zorder = 5,
                        label = f'GLORYS land'
                    )

                # Plot GLORYS land points intersecting with FVCOM
                int_land = self.data_points[getattr(self, f'{field}_index')[indices]]
                plt.plot(int_land[:,0], int_land[:,1], 'r.', label = f'{field} points', zorder = 10)
        plt.show(block = False)

        # After all fields have been plotted
        # ----
        if error_occured:
            plt.axis('equal')
            plt.title('Points in GLORYS land mask intersecting with FVCOM mesh')
            plt.legend(loc = 'upper right')
            raise LandError('GLORYS land intersects with your FVCOM experiment in the nestingzone, see the figure and adjust the mesh.')

class LandError(Exception): pass
class DomainError(Exception): pass