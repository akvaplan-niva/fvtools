import numpy as np

from fvtools.interpolators.nearest4 import N4 # base-class for nearest 4 interpolation & the method used to find nearest 4
from fvtools.util.kdtree import KDTree
from functools import cached_property

class DomainMasks:
    '''
    We reduce the data we handle to reduce memory use
    '''
    offset = 40_000
    @cached_property
    def fv_domain_mask(self):
        '''
        Reduces the number of AROME points we take in to consideration when computing the interpolation coefficients
        '''
        return self.COARSE.crop_grid(xlim=[self.FVCOM_grd.x.min() - self.offset, self.FVCOM_grd.x.max() + self.offset],
                                     ylim=[self.FVCOM_grd.y.min() - self.offset, self.FVCOM_grd.y.max() + self.offset])

    @cached_property
    def fv_domain_mask_ex(self):
        '''
        crops the ex-domain to the smaller arome domain found in other files
        '''
        return self.COARSE.crop_extended(xlim=[self.FVCOM_grd.x.min() - self.offset, self.FVCOM_grd.x.max() + self.offset],
                                         ylim=[self.FVCOM_grd.y.min() - self.offset, self.FVCOM_grd.y.max() + self.offset])

    @cached_property
    def xy_center_mask(self):
        '''
        mask centre-poitns (used to find nearest4) to the fv_domain_mask
        '''
        return np.logical_and(self.fv_domain_mask[1:, 1:], self.fv_domain_mask[:-1, :-1])

    @property
    def LandMask(self):
        '''
        1 = ocean, 0 = land
        '''
        return self.COARSE.land_mask[self.fv_domain_mask].astype(int)

class N4Coefficients(N4, DomainMasks):
    '''
    Object with indices and coefficients for AROME to FVCOM interpolation
    - note: radiation fields will only use ocean points, if land in radiation square, we will mask it.
            squares with full-land coverage will use nearest ocean neighbor
    '''
    def __init__(self, FVCOM_grd, COARSE_grd):
        '''
        Docstring for __init__
        
        :param FVCOM_grd: An initialized FVCOM_grid object
        :param COARSE_grd: An initialized coarse atmopsheric model grid object, either AROME_grid or ERA5 grid
        '''
        self.FVCOM_grd = FVCOM_grd
        self.COARSE    = COARSE_grd

    @property
    def xy_coarse(self):
        return np.array([self.COARSE.x[self.fv_domain_mask], self.COARSE.y[self.fv_domain_mask]]).transpose()
    
    @property
    def xy_coarse_center(self):
        return np.array([self.COARSE.x_center[self.xy_center_mask], self.COARSE.y_center[self.xy_center_mask]]).transpose()
    
    @property
    def fv_nodes(self):
        return np.array([self.FVCOM_grd.x, self.FVCOM_grd.y]).transpose()

    @property
    def fv_cells(self):
        return np.array([self.FVCOM_grd.xc, self.FVCOM_grd.yc]).transpose()
    
    @property
    def cell_utm_angle(self):
        return self.FVCOM_grd.cell_utm_angle

    def compute_nearest4(self):
        '''
        Create nearest four indices and weights for all of the fields
        '''
        # Check that the coarse grid covers the FVCOM grid
        print('- Compute interpolation coefficients')
        self.nindex, self.ncoef = self.get_interpolation_matrices(xy_source = self.xy_coarse,
                                                                  source_mask = self.fv_domain_mask,
                                                                  xy_fvcom = self.fv_nodes,
                                                                  widget_title = 'node')

        self.cindex, self.ccoef = self.get_interpolation_matrices(xy_source = self.xy_coarse,
                                                                  source_mask = self.fv_domain_mask,
                                                                  xy_fvcom = self.fv_cells,
                                                                  widget_title = 'cell')

        # We can not use radiation fields from land points
        print('  - Compute interpolation indices and coefficients without land points')
        self.nindex_ocean, self.ncoef_ocean = self._remove_land_points(self.nindex, self.ncoef)
        self.cindex_ocean, self.ccoef_ocean = self._remove_land_points(self.cindex, self.ccoef)

    def _remove_land_points(self, index, coef):
        '''
        Identify land points, these must be removed for the radiation field interpolation
        '''
        newindex = np.copy(index)
        newcoef = np.copy(coef)

        # N4Coefficients.LandMask uses 1=sea and 0=land.
        newcoef[self.LandMask[newindex] == 0] = 0.0
        total = np.sum(newcoef, axis=1)
        has_ocean = total > 0
        # Never divide by zero when all four corners are land.
        newcoef[has_ocean] /= total[has_ocean, None]

        all_land = ~has_ocean
        if np.any(all_land):
            nearest = self._find_nearest_ocean_neighbor(
                index[all_land], coef[all_land])
            # Repeat the same nearest sea point four times, preserving the
            # existing nearest-ocean fallback without masked / NaN weights.
            newindex[all_land, :] = nearest[:, None]
            newcoef[all_land, :] = 0.25
        return newindex, newcoef

    def _find_nearest_ocean_neighbor(self, index, coef):
        '''
        replace land point with nearest ocean neighbor
        '''
        ocean_indices = np.flatnonzero(self.LandMask == 1)
        if not ocean_indices.size:
            raise ValueError(
                'No sea points in the cropped atmospheric grid. Check the '
                'ERA5 land/sea mask (lsm: 0=sea, 1=land), grid extent and '
                'DomainMasks.offset; ocean-only variables cannot be interpolated.'
            )
        # Positions of only the targets requiring land fallback.
        fvcom_x = np.sum(self.xy_coarse[index, 0] * coef, axis=1)
        fvcom_y = np.sum(self.xy_coarse[index, 1] * coef, axis=1)
        ocean_tree = KDTree(self.xy_coarse[ocean_indices])
        _, nearest = ocean_tree.query(np.column_stack((fvcom_x, fvcom_y)))
        return ocean_indices[np.atleast_1d(nearest)]

    def compute_cell_arome_angle(self, latlon):
        '''
        Angle (radians, counter-clockwise) from the model's x-axis, or from true east if latlon, to the AROME
        grid's x-axis at each FVCOM cell. Winds along the AROME grid axes are turned by this angle.

        Uses the same cell coefficients as the winds (call compute_nearest4 first), and interpolates unit
        vectors rather than angles.
        '''
        angle = self.COARSE.x_axis_angle_east if latlon else self.COARSE.x_axis_angle
        angle = np.asarray(angle)[self.fv_domain_mask]
        self.cell_arome_angle = np.arctan2(np.sum(np.sin(angle)[self.cindex] * self.ccoef, axis=1),
                                           np.sum(np.cos(angle)[self.cindex] * self.ccoef, axis=1))

    def dump(self):
        '''
        Not sure if this is needed, but I am scared of using much more memory than needbe when parallellizing
        '''
        smallerN4 = N4_interpolation()
        smallerN4 = self._set_attributes_to_dump(smallerN4, self, ['fv_domain_mask', 'nindex', 'cindex', 'nindex_ocean'])
        smallerN4 = self._set_attributes_to_dump(smallerN4, self, ['ncoef', 'ccoef', 'ncoef_ocean', 'cell_arome_angle'])
        smallerN4 = self._set_attributes_to_dump(smallerN4, self.FVCOM_grd, ['cell_utm_angle'])
        if hasattr(self, 'fv_domain_mask_ex'):
            smallerN4 = self._set_attributes_to_dump(smallerN4, self, ['fv_domain_mask_ex'])
        return smallerN4

    def _set_attributes_to_dump(self, smallerN4, source, fields):
        '''
        Used to set attributes of the smaller version of the N4 class
        '''
        for field in fields:
            setattr(smallerN4, field, getattr(source, field))
        return smallerN4

class N4_interpolation:
    '''
    containing the fields we need for the interpolation methods
    '''
    fv_domain_mask: np.array
    cell_utm_angle: np.array
    cell_arome_angle: np.array
    nindex: np.array
    cindex: np.array
    nindex_rad: np.array
    ncoef: np.array
    ccoef: np.array
    ncoef_rad: np.array
    fv_domain_mask_ex: np.array = np.empty(0)