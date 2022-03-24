import numpy as np
from netCDF4 import Dataset


def add_vertical_interpolation2N4(N4,coords = ['rho', 'u', 'v']):
    '''Adds matrices for vertical interpolation to N4 object.'''    
    for c in coords:
        z_roms = getattr(N4.ROMS_grd, 'z_' + c)                   # z at all points in ROMS grid
        z_roms = z_roms[getattr(N4, 'fv_' + c + '_mask'), :]      # z cropped
        z_roms = np.sum(z_roms[getattr(N4, c + '_index'), :] 
                        * getattr(N4, c + '_coef')[:, :, None],
                        axis=1)                                   # z interpolated to fvcom-points

        z_roms = np.flip(z_roms, axis=1) 
        
        if c == 'rho':
            try:
                z_fvcom = getattr(N4.NEST, 'siglay') * N4.fvcom_rho_dpt[:, None]
            except:
                z_fvcom = getattr(N4.NEST, 'siglay') * N4.fvcom_rho_dpt[:]
        else:
            z_fvcom = getattr(N4.NEST, 'siglay_center')  \
                    * (0.5*N4.fvcom_u_dpt[:, None] + 0.5*N4.fvcom_v_dpt[:, None])
         

        ind1, ind2, weigths1, weigths2 = calc_interp_matrices(-z_roms.T, -z_fvcom.T)

        
        setattr(N4, 'vi_ind1_' + c, ind1) 
        setattr(N4, 'vi_ind2_' + c, ind2)
        setattr(N4, 'vi_weigths1_' + c, weigths1) 
        setattr(N4, 'vi_weigths2_' + c, weigths2) 
    
    return N4

# For data from nearest neighbor interpolation
def add_vertical_interpolation2N1(N1, coords = ['u','v','rho']):
    '''
    Adds matrices for vertical interpolation to the N1 object.
    '''
    for c in coords:
        z_roms  = getattr(N1.ROMS_grd, 'z_' + c)              # z at all points in ROMS grid
        z_roms  = z_roms[getattr(N1, 'fv_' + c + '_mask'), :] # z cropped
        z_roms  = z_roms[getattr(N1, 'Land_'+c)==0]
        z_roms  = z_roms[getattr(N1, c + '_index'), :]      # z at the points nearest the FVCOM points
    
        z_roms  = np.flip(z_roms, axis=1) 
        #z_fvcom = getattr(N1.NEST, 'siglay') * N1.fvcom_rho_dpt[:]
        if c == 'rho':
            try:
                z_fvcom = getattr(N1.NEST, 'siglay').T * N1.fvcom_rho_dpt[:, None]
            except:
                z_fvcom = getattr(N1.NEST, 'siglay').T * N1.fvcom_rho_dpt[:]

        else:
            try:
                z_fvcom = getattr(N1.NEST, 'siglay_c').T  \
                          * (0.5*N1.fvcom_u_dpt[:, None] + 0.5*N1.fvcom_v_dpt[:, None])
            except:
                z_fvcom = getattr(N1.NEST, 'siglay_c').T  \
                          * (0.5*N1.fvcom_u_dpt[:] + 0.5*N1.fvcom_v_dpt[:])

        ind1, ind2, weigths1, weigths2 = calc_interp_matrices(-z_roms.T, -z_fvcom)
        
        setattr(N1, 'vi_ind1_' + c, ind1)
        setattr(N1, 'vi_ind2_' + c, ind2)
        setattr(N1, 'vi_weigths1_' + c, weigths1)
        setattr(N1, 'vi_weigths2_' + c, weigths2)
    
    return N1

def calc_interp_matrices(z1, z2):
    '''Calculate matrices for linear interpolation.
    
    Parameters
    ----------
    x: array (m x n) 
       The x-coordinate of the data points.

    xi: array (k x n) 
        The coordinates at which to evaluate the interpolated values. 

    Returns
    -------
    Indices of the data on both sides of xi. 
    ind1: First array of indices.
    ind2: Second array of indices.

    Weigths to multiply with the values given by indices.
    weigths1: First array of weigths.
    weigths2: Second array of indices.


    Notes
    -----
    For values in xi outside the range of x, the highest/lowest values 
    in x are used. No extrapolation is carried out.  


    Example
    --------
    x = np.arange(0, 2*np.pi, 0.5)
    y = np.sin(x)
    
    # Interpolate to twice the resolution
    new_x = np.arange(0, 2*np.pi, 0.25)
    ind1, ind2, weigths1, weigths2 = calc_interp_indices(x, new_x)
    new_y = y[ind1]*weigths1 + y[ind2]*weigths2 
    '''
   
    if len(z2.shape) == 1:
        z2 = z2[:, None]
    if len(z1.shape) == 1:
        z1 = z1[:, None]

    indices1 = np.zeros(z2.shape)
    indices2 = np.zeros(z2.shape)
    weigths1 = np.zeros(z2.shape)
    weigths2 = np.zeros(z2.shape)
  
    for n in range(0, z2.shape[-1]):

        z1_n = z1[:, n]
        z2_n = z2[:, n] 
        
        absdiff = np.abs((z2_n[:, None] - z1_n[None, :]))

        ind1 = np.argmin(absdiff, axis=1)
        ind2 = np.zeros(len(ind1))        
        w1 = np.zeros(len(ind1))
        w2 = np.zeros(len(ind2))
        
        # If the shallowest depth in z2_n is shallower than the shallowest depth in z1_n,
        # use the shallowest z2_n value. 
        shallow_ind = z2_n <= z1_n[0]
        w1[shallow_ind] = 1
        
        # If the deepest depth in z2_n is deeper than the deepest depth in z1_n,
        # use the deepest z2_n value. 
        deep_ind = z2_n >= z1_n[-1]
        w1[deep_ind] = 1
        
        # If a value in z2_n is equal to a value in z1_n, no interpolation is necessary,
        # and z1_n value is used
        zero_ind =  z2_n-z1_n[ind1] == 0
        w1[zero_ind] = 1
        
        
        # Handle values in between
        between_ind = (~shallow_ind) & (~deep_ind) & (~zero_ind)
        smaller_ind = (z1_n[ind1] > z2_n) & between_ind
        ind2[smaller_ind] = ind1[smaller_ind] - 1
        ind2 = ind2.astype('int')
        
        larger_ind = (z1_n[ind1] < z2_n) & between_ind
        ind2[larger_ind] = ind1[larger_ind] + 1
        
        dz1 = z1_n[ind1[smaller_ind]] - z1_n[ind2[smaller_ind]]
        dz2 = z1_n[ind2[larger_ind]] - z1_n[ind1[larger_ind]]
        
        w2[smaller_ind] = (z1_n[ind1[smaller_ind]] - z2_n[smaller_ind]) / dz1
        w2[larger_ind] = (z2_n[larger_ind] - z1_n[ind1[larger_ind]]) / dz2
        
        w1[smaller_ind] = (z2_n[smaller_ind] - z1_n[ind2[smaller_ind]]) / dz1
        w1[larger_ind] = (z1_n[ind2[larger_ind]] - z2_n[larger_ind]) / dz2


        indices1[:, n] = ind1
        indices2[:, n] = ind2
        weigths1[:, n] = w1
        weigths2[:, n] = w2

    indices1 = indices1.astype('int')
    indices2 = indices2.astype('int')

    return indices1, indices2, weigths1, weigths2 


def calc_uv_bar(ncfile, ngrd, M):
    '''Calculate vertical average of velocity (ubar, vbar) and write to ncfile.'''
    h_uv = np.squeeze(M.h_uv[ngrd.cid])
    siglevz = ngrd.siglev_center * h_uv[:, None] 
    dz = np.abs(np.diff(siglevz, axis=1))

    nc = Dataset(ncfile, 'r+')
    ubar = nc.variables['ua']
    vbar = nc.variables['va']

    number_of_timesteps = ubar.shape[0]

    for n in range(0, number_of_timesteps):
        u = nc.variables['u'][n, :, :].T
        ubar[n, :] = np.sum(u*dz, axis=1) / h_uv

        v = nc.variables['v'][n, :, :].T
        vbar[n, :] = np.sum(v*dz, axis=1) / h_uv

    nc.close()

 

