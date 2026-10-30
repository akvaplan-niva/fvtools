import numpy as np
import matplotlib.pyplot as plt
from fvtools.grid.roms.cropper_roms import CropRho, CropU, CropV
from .nearest4 import N4

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

class N4nestdepth:
    '''
    the depth we interpolate to depends on the horizontal interpolation scheme
    '''
    @property
    def fvcom_rho_dpt(self):
        return np.sum(self.ROMS.h_rho[self.ROMS.fv_rho_mask][self.rho_index] * self.rho_coef, axis=1)

    @property
    def fvcom_u_dpt(self):
        return np.sum(self.ROMS.h_u[self.ROMS.fv_u_mask][self.u_index] *self.u_coef, axis=1)

    @property
    def fvcom_v_dpt(self):
        return np.sum(self.ROMS.h_v[self.ROMS.fv_v_mask][self.v_index] *self.v_coef, axis=1)

class N4ROMS(N4):
    '''
    Object with indices and coefficients for ROMS to FVCOM interpolation using bilinear coefficients
    '''
    mother = 'ROMS'

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
            self.rho_index, self.rho_coef = self.get_interpolation_matrices(
                xy_source=self.rho_points, source_mask=self.ROMS.fv_rho_mask,
                xy_fvcom=self.fv_nodes, widget_title='rho')
            if self.uv:
                self.u_index, self.u_coef = self.get_interpolation_matrices(
                    xy_source=self.u_points, source_mask=self.ROMS.fv_u_mask,
                    xy_fvcom=self.fv_cells, widget_title='u')

                self.v_index, self.v_coef = self.get_interpolation_matrices(
                    xy_source=self.v_points, source_mask=self.ROMS.fv_v_mask,
                    xy_fvcom=self.fv_cells, widget_title='v')

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
        for field in (['rho', 'u', 'v'] if self.uv else ['rho']):
            indices = getattr(self.ROMS, f'Land_{field}').ravel()[getattr(self, f'{field}_index')]
            if indices.any():
                # If so, we only need to plot the background once, hence "if not error_occured"
                if not error_occured:
                    plt.figure()
                    plt.triplot(self.x, self.y, self.tri, c = 'k', marker = 's', label = 'FVCOM', zorder = 100)
                    error_occured = True
                    
                    # Plot all ROMS land points in the vicinity
                    plt.plot(
                        getattr(self, f'{field}_points')[:, 0], 
                        getattr(self, f'{field}_points')[:, 1], 
                        'b.', 
                        marker = 's', zorder = 5, label = 'ROMS ocean'
                        )
                    plt.axis('equal')
                
                    # Plot all ROMS land points in the vicinity
                    plt.plot(
                        getattr(self, f'{field}_points')[getattr(self.ROMS, f'Land_{field}').ravel(), 0], 
                        getattr(self, f'{field}_points')[getattr(self.ROMS, f'Land_{field}').ravel(), 1],
                        'g.', 
                        marker = 's',
                        zorder = 5,
                        label = f'ROMS land'
                    )

                # Plot ROMS land points intersecting with FVCOM
                int_land = getattr(self, f'{field}_points')[getattr(self, f'{field}_index')[indices]]
                plt.plot(int_land[:,0], int_land[:,1], 'r.', marker = 's', label = f'{field} points', zorder = 10)

        # After all fields have been plotted
        # ----
        if error_occured:
            plt.axis('equal')
            plt.title('Points in ROMS land mask intersecting with FVCOM mesh')
            plt.legend(loc = 'upper right')
            raise LandError('ROMS land intersect with your FVCOM experiment in the nestingzone. See the figure and adjust the mesh.')

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

class N4ROMSNESTING(N4ROMS, N4nestdepth):
    def __init__(self, ROMS, x = None, y = None, tri = None, uv = False, land_check=True):
        '''
        Initialize empty attributes
        '''
        self.x, self.y = x, y
        self.tri = tri
        self.uv = uv
        self.land_check = land_check
        self.ROMS = ROMS

# Phase 5: bounded-memory interpolation of already-downloaded ROMS arrays.
# A smaller block lowers peak RAM usage; it does not trigger an additional read.
_ROMS_INTERPOLATION_BLOCK_SIZE = 4096


def _horizontal_interpolate_blockwise(data, indices, coefficients):
    """Interpolate existing 1-D or (point, sigma) data with bounded temporaries.

    np.sum retains its original four-neighbour reduction and operation order.
    Use normal broadcasting rather than repeating coefficients in sigma.
    Masked input remains masked output, including a mask in any block.
    """
    npoints = indices.shape[0]
    is_profile = data.ndim == 2

    def calculate(first, last):
        if is_profile:
            return np.sum(data[indices[first:last], :] *
                          coefficients[first:last, :, None], axis=1)
        return np.sum(data[indices[first:last]] *
                      coefficients[first:last], axis=1)

    if npoints <= _ROMS_INTERPOLATION_BLOCK_SIZE:
        return calculate(0, npoints)

    result = None
    for first in range(0, npoints, _ROMS_INTERPOLATION_BLOCK_SIZE):
        last = min(first + _ROMS_INTERPOLATION_BLOCK_SIZE, npoints)
        part = calculate(first, last)
        if result is None:
            shape = (npoints,) + part.shape[1:]
            factory = np.ma.empty if np.ma.isMaskedArray(part) else np.empty
            result = factory(shape, dtype=part.dtype)
        result[first:last] = part
    return result


def _vertical_interpolate_blockwise(data, index1, index2, weight1, weight2):
    """Original vertical gather and weighted sum, sliced by FVCOM point.

    Input has shape (point, ROMS sigma); result (FVCOM sigma, point).
    Only local arrays are accessed, so no new THREDDS/NetCDF requests occur.
    """
    profile = np.flip(data, axis=1).T
    npoints = profile.shape[1]

    def calculate(first, last):
        columns = np.arange(first, last)
        return (profile[index1[:, first:last], columns] * weight1[:, first:last]
                + profile[index2[:, first:last], columns] * weight2[:, first:last])

    if npoints <= _ROMS_INTERPOLATION_BLOCK_SIZE:
        return calculate(0, npoints)

    result = None
    for first in range(0, npoints, _ROMS_INTERPOLATION_BLOCK_SIZE):
        last = min(first + _ROMS_INTERPOLATION_BLOCK_SIZE, npoints)
        part = calculate(first, last)
        if result is None:
            shape = (part.shape[0], npoints)
            factory = np.ma.empty if np.ma.isMaskedArray(part) else np.empty
            result = factory(shape, dtype=part.dtype)
        result[:, first:last] = part
    return result


class LinearInterpolation:
    '''
    Linearly interpolate data from ROMS to FVCOM grid points
    - Bi-linear in the horizontal
    - Linear (or nearest neighbor if below bottom) in the vertical
    '''
    def horizontal_interpolation(self, timestep, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']):
        """Bilinear ROMS-to-FVCOM interpolation using bounded local blocks."""
        for name, coordinate in (('u', 'u'), ('v', 'v'),
                                 ('ua', 'u'), ('va', 'v'),
                                 ('zeta', 'rho'), ('temp', 'rho'), ('salt', 'rho')):
            if name in variables:
                setattr(timestep, name, _horizontal_interpolate_blockwise(
                    getattr(timestep, name),
                    getattr(self.N4, f'{coordinate}_index'),
                    getattr(self.N4, f'{coordinate}_coef')))
        return timestep

    def vertical_interpolation(self, timestep, variables = ['salt', 'temp', 'zeta', 'u', 'v', 'ua', 'va']):
        """Linear vertical interpolation, with bounded temporary arrays."""
        for name, coordinate in (('salt', 'rho'), ('temp', 'rho'),
                                 ('u', 'u'), ('v', 'v')):
            if name in variables:
                setattr(timestep, name, _vertical_interpolate_blockwise(
                    getattr(timestep, name),
                    getattr(self.N4, f'vi_ind1_{coordinate}'),
                    getattr(self.N4, f'vi_ind2_{coordinate}'),
                    getattr(self.N4, f'vi_weigths1_{coordinate}'),
                    getattr(self.N4, f'vi_weigths2_{coordinate}')))
        return timestep


class LandError(Exception): pass
class DomainError(Exception): pass
