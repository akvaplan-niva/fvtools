from functools import cached_property
import numpy as np

class CropRho:
    '''
    We always deal with a cropped verison of the ROMS grid in the nesting- restart- and movie routines.
    These classes are used to crop the roms domain to cover xbounds, ybounds (xy-coordinates), and provides metrics (m_ri, x_ri) etc,
    to be used when we crop data on-the-fly while downloading.
    '''
    def crop_rho(self):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_rho >= np.min(self.xbounds)-self.offset, self.x_rho <= np.max(self.xbounds)+self.offset)
        ind2 = np.logical_and(self.y_rho >= np.min(self.ybounds)-self.offset, self.y_rho <= np.max(self.ybounds)+self.offset)
        return np.logical_and(ind1, ind2)

    @property
    def fv_rho_mask(self):
        if not hasattr(self, '_fv_rho_mask'):
            self._fv_rho_mask = self.crop_rho()
        return self._fv_rho_mask

    @fv_rho_mask.setter
    def fv_rho_mask(self, var):
        self._fv_rho_mask = var

    @cached_property
    def cropped_rho_mask(self):
        '''Cropped version of the rho-mask that we use when processing the data downloaded from thredds'''
        return self.fv_rho_mask[self.m_ri:self.x_ri+1, self.m_rj:self.x_rj+1]

    @cached_property
    def m_ri(self):
        '''min i-index for rho-points'''
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return min(rho_i)

    @cached_property
    def x_ri(self):
        '''max i-index for rho points'''
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return max(rho_i)

    @cached_property
    def m_rj(self):
        '''min j-index for rho-points'''
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return min(rho_j)

    @cached_property
    def x_rj(self):
        '''max j-index for rho points'''
        rho_i, rho_j = np.where(self.fv_rho_mask)
        return max(rho_j)

class CropU:
    def crop_u(self):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_u >= np.min(self.xbounds)-self.offset, self.x_u <= np.max(self.xbounds)+self.offset)
        ind2 = np.logical_and(self.y_u >= np.min(self.ybounds)-self.offset, self.y_u <= np.max(self.ybounds)+self.offset)
        return np.logical_and(ind1, ind2)

    @property
    def fv_u_mask(self):
        if not hasattr(self, '_fv_u_mask'):
            self._fv_u_mask = self.crop_u()
        return self._fv_u_mask

    @fv_u_mask.setter
    def fv_u_mask(self, var):
        self._fv_u_mask = var

    @cached_property
    def cropped_u_mask(self):
        return self.fv_u_mask[self.m_ui:self.x_ui+1, self.m_uj:self.x_uj+1]

    @cached_property
    def m_ui(self):
        u_i, u_j = np.where(self.fv_u_mask)
        return min(u_i)

    @cached_property
    def x_ui(self):
        u_i, u_j = np.where(self.fv_u_mask)
        return max(u_i)

    @cached_property
    def m_uj(self):
        u_i, u_j = np.where(self.fv_u_mask)
        return min(u_j)

    @cached_property
    def x_uj(self):
        u_i, u_j = np.where(self.fv_u_mask)
        return max(u_j)

class CropV:
    def crop_v(self):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_v >= np.min(self.xbounds)-self.offset, self.x_v <= np.max(self.xbounds)+self.offset)
        ind2 = np.logical_and(self.y_v >= np.min(self.ybounds)-self.offset, self.y_v <= np.max(self.ybounds)+self.offset)
        return np.logical_and(ind1, ind2)

    @property
    def fv_v_mask(self):
        if not hasattr(self, '_fv_v_mask'):
            self._fv_v_mask = self.crop_v()
        return self._fv_v_mask

    @fv_v_mask.setter
    def fv_v_mask(self, var):
        self._fv_v_mask = var

    @cached_property
    def cropped_v_mask(self):
        return self.fv_v_mask[self.m_vi:self.x_vi+1, self.m_vj:self.x_vj+1]

    @cached_property
    def m_vi(self):
        v_i, v_j = np.where(self.fv_v_mask)
        return min(v_i)

    @cached_property
    def x_vi(self):
        v_i, v_j = np.where(self.fv_v_mask)
        return max(v_i)

    @cached_property
    def m_vj(self):
        v_i, v_j = np.where(self.fv_v_mask)
        return min(v_j)

    @cached_property
    def x_vj(self):
        v_i, v_j = np.where(self.fv_v_mask)
        return max(v_j)

class ROMSCropper(CropU, CropV, CropRho):
    '''
    Class that uses the crop-properties to fit arrays to the desired subdomain
    '''
    @property
    def cropped_dsigma_v(self):
        '''
        sigma layer thickness of each ROMS layer
        '''
        return self.dsigma_v[:, self.m_vi:self.x_vi+1, self.m_vj:self.x_vj+1]
    
    @property
    def cropped_dsigma_u(self):
        '''
        sigma layer thickness of each ROMS layer
        '''
        return self.dsigma_u[:, self.m_ui:self.x_ui+1, self.m_uj:self.x_uj+1]

    # Cropped roms coordinates as (n,) arrays.
    @property
    def cropped_x_rho(self):
        return self.x_rho[self.fv_rho_mask]

    @property
    def cropped_y_rho(self):
        return self.y_rho[self.fv_rho_mask]

    @property
    def cropped_x_u(self):
        return self.x_u[self.fv_u_mask]

    @property
    def cropped_y_u(self):
        return self.y_u[self.fv_u_mask]

    @property
    def cropped_x_v(self):
        return self.x_v[self.fv_v_mask]

    @property
    def cropped_y_v(self):
        return self.y_v[self.fv_v_mask]

    # Land-mask
    @property
    def Land_rho(self):
        '''rho mask cropped to fit with the mesh we're interpolating to'''
        return self.rho_mask[self.fv_rho_mask]

    @property
    def Land_u(self):
        '''u mask cropped to fit with the mesh we're interpolating to'''
        return self.u_mask[self.fv_u_mask]

    @property
    def Land_v(self):
        '''v mask cropped to fit with the mesh we're interpolating to'''
        return self.v_mask[self.fv_v_mask]

    @property
    def cropped_x_psi(self):
        umask = np.logical_and(self.fv_u_mask[1:,:], self.fv_u_mask[:-1,:])
        vmask = np.logical_and(self.fv_v_mask[:,1:], self.fv_v_mask[:,:-1])
        psi_mask = np.logical_and(umask,vmask)
        return (self.x_u[1:,:] + self.x_u[:-1,:])[psi_mask]/2

    @property
    def cropped_y_psi(self):
        umask = np.logical_and(self.fv_u_mask[1:,:], self.fv_u_mask[:-1,:])
        vmask = np.logical_and(self.fv_v_mask[:,1:], self.fv_v_mask[:,:-1])
        psi_mask = np.logical_and(umask,vmask)
        return (self.y_v[:,1:] + self.y_v[:,:-1])[psi_mask]/2

    @property
    def cropped_x_rho_grid(self):
        '''Exclusively used in the ROMS movie maker'''
        return self.x_rho[self.m_ri:(self.x_ri+1), self.m_rj:(self.x_rj+1)]

    @property
    def cropped_y_rho_grid(self):
        '''Exclusively used in the ROMS movie maker'''
        return self.y_rho[self.m_ri:(self.x_ri+1), self.m_rj:(self.x_rj+1)]