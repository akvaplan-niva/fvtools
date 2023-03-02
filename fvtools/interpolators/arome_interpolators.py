import numpy as np

from fvtools.interpolators.nearest4 import N4 # base-class for nearest 4 interpolation & method to find nearest 4
from scipy.spatial import cKDTree as KDTree
from functools import cached_property


# Script storing methods used to interpolate data from AROME grid to the FVCOM mesh (currently just supportinh nearest4 interpolation)
class AromeDomainMasks:
    '''
    We reduce the data we handle to reduce memory use
    '''
    @cached_property
    def fv_domain_mask(self):
        '''
        Reduces the number of AROME points we take in to consideration when computing the interpolation coefficients
        '''
        return self.AROME_grd.crop_grid(xlim=[self.FVCOM_grd.x.min() - 7500, self.FVCOM_grd.x.max() + 7500],
                                        ylim=[self.FVCOM_grd.y.min() - 7500, self.FVCOM_grd.y.max() + 7500])

    @cached_property
    def fv_domain_mask_ex(self):
        '''
        crops the ex-domain to the smaller arome domain found in other files
        '''
        return self.AROME_grd.crop_extended(xlim=[self.FVCOM_grd.x.min() - 7500, self.FVCOM_grd.x.max() + 7500],
                                            ylim=[self.FVCOM_grd.y.min() - 7500, self.FVCOM_grd.y.max() + 7500])

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
        return self.AROME_grd.landmask[self.fv_domain_mask].astype(int)

class N4AROME(N4, AromeDomainMasks):
    '''
    Object with indices and coefficients for AROME to FVCOM interpolation
    - note: radiation fields will only use ocean points, if land in radiation square, we will mask it.
            squares with full-land coverage will use nearest ocean neighbor
    '''
    def __init__(self, FVCOM_grd, AROME_grd):
        self.FVCOM_grd = FVCOM_grd
        self.AROME_grd = AROME_grd

    @property
    def xy_arome(self):
        return np.array([self.AROME_grd.x[self.fv_domain_mask], self.AROME_grd.y[self.fv_domain_mask]]).transpose()
    
    @property
    def xy_arome_center(self):
        return np.array([self.AROME_grd.x_center[self.xy_center_mask], self.AROME_grd.y_center[self.xy_center_mask]]).transpose()
    
    @property
    def fv_nodes(self):
        return np.array([self.FVCOM_grd.x, self.FVCOM_grd.y]).transpose()

    @property
    def fv_cells(self):
        return np.array([self.FVCOM_grd.xc, self.FVCOM_grd.yc]).transpose()    

    def load_nearest4(self, infile):
        '''
        Load an already computed nearest4.
        '''
        nearest4 = np.load(infile, allow_pickle=True).item()
        for field in ['nindex', 'ncoef', 'cindex', 'ccoef', 'nindex_rad', 'ncoef_rad', 'cindex_rad', 'ccoef_rad']:
            setattr(self, field, nearest4[field])

    def compute_nearest4(self, nearest4filename = 'nearest4arome.npy'):
        '''
        Create nearest four indices and weights for all of the fields
        '''
        assert all(self.xy_arome[:, 0] == self.AROME_grd.xex[self.fv_domain_mask_ex]), f"(parts) of your FVCOM grid seems to be outside of the AROME domain"

        print('  - Compute interpolation coefficients')
        self.nindex, self.ncoef = self.get_interpolation_matrices(xy_source = self.xy_arome,
                                                                  xy_source_center = self.xy_arome_center,
                                                                  xy_fvcom = self.fv_nodes,
                                                                  widget_title = 'node')

        self.cindex, self.ccoef = self.get_interpolation_matrices(xy_source = self.xy_arome,
                                                                  xy_source_center = self.xy_arome_center,
                                                                  xy_fvcom = self.fv_cells,
                                                                  widget_title = 'cell')

        # We can not use radiation fields from land points
        print('  - Adjust radiation interpolation to avoid using land points')
        self.nindex_rad, self.ncoef_rad = self._remove_land_points(self.nindex, self.ncoef)
        self.cindex_rad, self.ccoef_rad = self._remove_land_points(self.cindex, self.ccoef)

        print(f'  - Save coefficients and indices for later: {nearest4filename}')
        out = {}
        for field in ['nindex', 'ncoef', 'cindex', 'ccoef', 'nindex_rad', 'ncoef_rad', 'cindex_rad', 'ccoef_rad']:
            out[field] = getattr(self, field)
        np.save(nearest4filename, out)

    def _remove_land_points(self, index, coef):
        '''
        Identify land points, these must be removed for the radiation field interpolation
        '''
        newindex = np.copy(index)
        newcoef  = np.copy(coef)

        # Set weight of land points to zero and re-normalize (use nearest3)
        landbool = self.LandMask[newindex]==0
        newcoef[landbool] = 0
        newcoef = newcoef/np.sum(newcoef,axis=1)[:,None]

        # Find the nearest neighbour where all points are landpoints
        points_on_arome_land = np.where(np.isnan(newcoef[:,0]))[0]  # points completely covered by arome land
        nearest_arome_ocean  = self._find_nearest_ocean_neighbor(index, coef)

        # Overwrite indices at points completely covered by arome land
        newindex[points_on_arome_land, :] = nearest_arome_ocean[points_on_arome_land][:, None]
        newcoef[points_on_arome_land, :]  = 0.25 # just weight the same point = 0.25 for 4 times, because why not...

        return newindex, newcoef

    def _find_nearest_ocean_neighbor(self, index, coef):
        '''
        replace land point with nearest ocean neighbor
        '''
        # Points we need to change
        fvcom_x = np.sum(self.xy_arome[index, 0]*coef, axis=1)
        fvcom_y = np.sum(self.xy_arome[index, 1]*coef, axis=1)

        # Create a tree referencing ocean points
        ocean_tree = KDTree(np.array([self.xy_arome[self.LandMask==1, 0], self.xy_arome[self.LandMask==1, 1]]).transpose())
        arome_tree = KDTree(self.xy_arome)

        # Nearest ocean point
        _, _nearest_ocean = ocean_tree.query(np.array([fvcom_x, fvcom_y]).transpose())
        nearest_ocean_x = self.xy_arome[self.LandMask==1, 0][_nearest_ocean]
        nearest_ocean_y = self.xy_arome[self.LandMask==1, 1][_nearest_ocean]

        # With same indexing as the rest of AROME
        _, nearest_arome_ocean = arome_tree.query(np.array([nearest_ocean_x, nearest_ocean_y]).transpose())
        return nearest_arome_ocean

    def dump(self):
        '''
        Not sure if this is needed, but I am scared of using much more memory than needbe when paralellizing
        '''
        smallerN4 = N4_interpolation()
        smallerN4 = self._set_attributes_to_dump(smallerN4, self, ['fv_domain_mask', 'nindex', 'cindex', 'nindex_rad'])
        smallerN4 = self._set_attributes_to_dump(smallerN4, self, ['ncoef', 'ccoef', 'ncoef_rad'])
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
    nindex: np.array
    cindex: np.array
    nindex_rad: np.array
    ncoef: np.array
    ccoef: np.array
    ncoef_rad: np.array
    fv_domain_mask_ex: np.array = np.empty(0)