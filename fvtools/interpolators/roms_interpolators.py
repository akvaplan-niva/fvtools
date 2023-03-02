import numpy as np
import matplotlib.pyplot as plt
from fvtools.grid.roms_grid import CropRho, CropU, CropV
from .nearest4 import N4
from dataclasses import dataclass

class N4_interpolation(CropRho, CropU, CropV):
    '''
    A class used to store central results from the RomsNesting classes
    '''
    def load(self, path, uv = True):
        '''
        Store interpolation coefficients in .npy file
        '''
        d = np.load(path, allow_pickle = True).item()
        for key in d.keys():
            setattr(self, key, d[key])

    def dump(self, uv = True):
        '''
        Store interpolation coefficients in .npy file
        '''
        d = {}
        for field in ['cropped_dsigma_u', 'cropped_dsigma_v', 'fv_rho_mask', 'cell_utm_angle', 'siglev', 'weight_node', 'weight_cell',
                      'rho_index', 'rho_coef', 'vi_ind1_rho', 'vi_ind2_rho', 'vi_weigths1_rho', 'vi_weigths2_rho']:
            d[field] = getattr(self, field)

        if uv:
            for field in ['fv_u_mask', 'fv_v_mask', 'angle', 'u_index', 'v_index', 'u_coef', 'v_coef', 'vi_ind1_u', 'vi_ind2_u', 
                          'vi_weigths1_u', 'vi_weigths2_u', 'vi_ind1_v', 'vi_ind2_v', 'vi_weigths1_v', 'vi_weigths2_v']:
                d[field] = getattr(self, field)
        
        np.save('nearest4roms.npy', d)

class N4ROMS(N4):
    '''
    Object with indices and coefficients for ROMS to FVCOM interpolation using bilinear coefficients
    '''
    # Mesh details
    # ----
    @property
    def xc(self):
        return np.mean(self.x[self.tri], axis = 1)
    
    @property
    def yc(self):
        return np.mean(self.y[self.tri], axis = 1)

    # Roms angle relative to lat/lon system
    # ----
    @property
    def angle(self):
        return np.mean(np.sum(self.ROMS.angle[self.ROMS.fv_rho_mask][self.rho_index] * self.rho_coef, axis=1)[self.tri], axis=1)

    # Points we need to find nearest 4
    # ----
    @property
    def psi_points(self):
        return np.array([self.ROMS.cropped_x_psi, self.ROMS.cropped_y_psi]).transpose()

    @property
    def rho_points(self):
        return np.array([self.ROMS.cropped_x_rho, self.ROMS.cropped_y_rho]).transpose()
    
    @property
    def u_points(self):
        return np.array([self.ROMS.cropped_x_u, self.ROMS.cropped_y_u]).transpose()

    @property
    def v_points(self):
        return np.array([self.ROMS.cropped_x_v, self.ROMS.cropped_y_v]).transpose()

    @property
    def fv_nodes(self):
        return np.array([self.x, self.y]).transpose()

    @property
    def fv_cells(self):
        return np.array([self.xc, self.yc]).transpose()

    # Call N4s methods to find the nearest4
    # ----
    def nearest4(self, M = None):
        '''
        Create nearest four indices and weights for all of the fields
        '''
        # Compute interpolation coefficients for nearest4 interpolation
        try:
            self.rho_index, self.rho_coef = self.get_interpolation_matrices(xy_source = self.rho_points, 
                                                                            xy_source_center = self.psi_points, 
                                                                            xy_fvcom = self.fv_nodes,
                                                                            widget_title = 'rho'
                                                                            )
            if self.uv:
                self.u_index, self.u_coef = self.get_interpolation_matrices(xy_source = self.u_points,
                                                                            xy_source_center = self.v_points,
                                                                            xy_fvcom = self.fv_cells,
                                                                            widget_title = 'u'
                                                                            )

                self.v_index, self.v_coef = self.get_interpolation_matrices(xy_source = self.v_points,
                                                                            xy_source_center = self.u_points,
                                                                            xy_fvcom = self.fv_cells,
                                                                            widget_title = 'v'
                                                                            )

        except ValueError:
            self.domain_exception_plot(self.psi_points)
            raise DomainError('Your FVCOM domain is outside of the ROMS domain, it needs to be changed.')

        # Check if your mesh covers ROMS land, if so kill the routine and force the user to change the grid
        self.check_if_ROMS_land_in_FVCOM_mesh(M)

    # Quality control. We don't want to use land points, since that would mean that we extrapolate the velocity field from ROMS.
    # ----
    def check_if_ROMS_land_in_FVCOM_mesh(self, M = None):
        '''
        Check if FVCOM covers ROMS land, if so return
        '''
        if not self.land_check: # return if told to not care about land
            return

        error_occured = False
        for field in ['rho','u','v']:
            indices = getattr(self.ROMS, f'Land_{field}')[getattr(self, f'{field}_index')]
            if indices.any():
                if not error_occured:
                    plt.figure()
                    if M is not None:
                        M.plot_grid()
                    else:
                        plt.plot(self.x, self.y, 'r.', label = 'FVCOM')
                    error_occured = True


                # Plot all ROMS land points in the vicinity
                plt.plot(getattr(self.ROMS, f'cropped_x_{field}')[getattr(self.ROMS, f'Land_{field}')], 
                         getattr(self.ROMS, f'cropped_y_{field}')[getattr(self.ROMS, f'Land_{field}')], 'k.', zorder = 5)

                # Plot ROMS land points intersecting with FVCOM
                x_roms = getattr(self.ROMS, f'cropped_x_{field}').ravel()[getattr(self, f'{field}_index')[indices]]
                y_roms = getattr(self.ROMS, f'cropped_y_{field}').ravel()[getattr(self, f'{field}_index')[indices]]
                plt.scatter(x_roms, y_roms, label = f'{field} points', zorder = 10)

        # After all fields have been plotted
        # ----
        if error_occured:
            plt.axis('equal')
            plt.title('Points in ROMS land mask intersecting with FVCOM mesh')
            plt.legend(loc = 'upper right')
            raise LandError('ROMS intersects with your FVCOM experiment in the nestingzone, see the figure and adjust the mesh.')

    def domain_exception_plot(self, ROMS_points):
        '''
        Plot that illustrates where FVCOM extends beyond ROMS
        '''
        plt.plot(ROMS_points[:, 0], ROMS_points[:, 1], 'r.', label = 'ROMS')
        plt.plot(self.x, self.y, 'b.', label = 'FVCOM')
        plt.legend()
        plt.axis('equal')
        plt.show(block=False)

    def dump(self):
        '''
        Dumps important fields to a lighter object that (hopefully) reduces the nesting routines memory needs
        '''
        # Copy attributes we need later from mother class to copy class
        smallerN4 = N4_interpolation()
        smallerN4 = self._set_attributes_to_dump(smallerN4, self.ROMS, ['cropped_dsigma_u', 'cropped_dsigma_v', 'fv_rho_mask'])
        smallerN4 = self._set_attributes_to_dump(smallerN4, self.FV, ['cell_utm_angle'])
        smallerN4 = self._set_attributes_to_dump(smallerN4, self, ['rho_index', 'rho_coef'])

        try:
            smallerN4 = self._set_attributes_to_dump(smallerN4, self, ['vi_ind1_rho', 'vi_ind2_rho', 'vi_weigths1_rho', 'vi_weigths2_rho']) 
        except:
            pass

        try:
            # Only needed and accessible for nesting files, this can/should be rewritten to be more elegant...
            smallerN4 = self._set_attributes_to_dump(smallerN4, self.FV, ['siglev', 'weight_node', 'weight_cell'])
        except:
            pass

        if self.uv:
            smallerN4 = self._set_attributes_to_dump(smallerN4, self.ROMS, ['fv_u_mask', 'fv_v_mask'])
            smallerN4 = self._set_attributes_to_dump(smallerN4, self, ['angle', 'u_index', 'v_index', 'u_coef', 'v_coef'])
            try:
                smallerN4 = self._set_attributes_to_dump(smallerN4, self, ['vi_ind1_u', 'vi_ind2_u', 'vi_weigths1_u', 'vi_weigths2_u'])
                smallerN4 = self._set_attributes_to_dump(smallerN4, self, ['vi_ind1_v', 'vi_ind2_v', 'vi_weigths1_v', 'vi_weigths2_v'])
            except:
                pass
        return smallerN4

    def _set_attributes_to_dump(self, smallerN4, source, fields):
        for field in fields:
            setattr(smallerN4, field, getattr(source, field))
        return smallerN4

class LinearInterpolation:
    '''
    Linearly interpolate data from ROMS to FVCOM grid points
    - Bi-linear in the horizontal
    - Linear (or nearest neighbor if below bottom) in the vertical
    '''
    def horizontal_interpolation(self, timestep, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']):
        '''
        bi-linear interpolation from one ROMS point to another
        '''
        zlen = timestep.salt.shape[-1]
        if 'u' in variables:
            timestep.u    = np.sum(timestep.u[self.N4.u_index, :]*np.repeat(self.N4.u_coef[:, :, np.newaxis], zlen, axis=2), axis=1)

        if 'v' in variables:
            timestep.v    = np.sum(timestep.v[self.N4.v_index, :]*np.repeat(self.N4.v_coef[:, :, np.newaxis], zlen, axis=2), axis=1)

        if 'ua' in variables:
            timestep.ua   = np.sum(timestep.ua[self.N4.u_index]*self.N4.u_coef, axis=1)

        if 'va' in variables:
            timestep.va   = np.sum(timestep.va[self.N4.v_index]*self.N4.v_coef, axis=1)

        if 'zeta' in variables:
            timestep.zeta = np.sum(timestep.zeta[self.N4.rho_index]*self.N4.rho_coef, axis=1)

        if 'temp' in variables:
            timestep.temp = np.sum(timestep.temp[self.N4.rho_index, :]*np.repeat(self.N4.rho_coef[: ,:, np.newaxis], zlen, axis=2), axis=1)

        if 'salt' in variables:
            timestep.salt = np.sum(timestep.salt[self.N4.rho_index, :]*np.repeat(self.N4.rho_coef[:, :, np.newaxis], zlen, axis=2), axis=1)
            
        return timestep

    def vertical_interpolation(self, timestep, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']):
        '''
        Linear vertical interpolation.
        '''
        if 'salt' in variables:
            salt = np.flip(timestep.salt, axis=1).T
            timestep.salt = salt[self.N4.vi_ind1_rho, range(0, salt.shape[1])] * self.N4.vi_weigths1_rho\
                        + salt[self.N4.vi_ind2_rho, range(0, salt.shape[1])] * self.N4.vi_weigths2_rho

        if 'temp' in variables:
            temp = np.flip(timestep.temp, axis=1).T
            timestep.temp = temp[self.N4.vi_ind1_rho, range(0, temp.shape[1])] * self.N4.vi_weigths1_rho \
                        + temp[self.N4.vi_ind2_rho, range(0, temp.shape[1])] * self.N4.vi_weigths2_rho

        if 'u' in variables:
            u = np.flip(timestep.u, axis=1).T
            timestep.u = u[self.N4.vi_ind1_u, range(0, u.shape[1])] * self.N4.vi_weigths1_u + \
                     + u[self.N4.vi_ind2_u, range(0, u.shape[1])] * self.N4.vi_weigths2_u

        if 'v' in variables:
            v = np.flip(timestep.v, axis=1).T
            timestep.v = v[self.N4.vi_ind1_v, range(0, v.shape[1])] * self.N4.vi_weigths1_v + \
                     + v[self.N4.vi_ind2_v, range(0, v.shape[1])] * self.N4.vi_weigths2_v
        return timestep

class LandError(Exception): pass
class DomainError(Exception): pass