import numpy as np
from numba import njit

def add_vertical_interpolation2N4(N4, coords = ['rho', 'u', 'v'], widget_tile = ''):
    '''
    Adds matrices for vertical interpolation to N4 object.
    - This is in fact a bit tedious, would be better to just give the fields, not getattr them within the routine
    '''    
    for c in coords:
        print(f'  - Finding vertical {c} weights:')
        z_roms = getattr(N4.ROMS, f'z_{c}')                  # z at all points in ROMS grid
        z_roms = z_roms[getattr(N4.ROMS, f'fv_{c}_mask'), :] # z cropped

        # z interpolated to fvcom-points
        z_roms = np.sum(z_roms[getattr(N4, f'{c}_index'), :] * getattr(N4, f'{c}_coef')[:, :, None], axis=1)
        z_roms = np.array(np.flip(z_roms, axis=1))

        # Find FVCOM depth
        if c == 'rho':
            z_fvcom = np.array(getattr(N4.FV, 'siglay') * N4.fvcom_rho_dpt[:, None])
        else:
            z_fvcom = np.array(getattr(N4.FV, 'siglay_center') * (0.5*N4.fvcom_u_dpt[:, None] + 0.5*N4.fvcom_v_dpt[:, None]))
         
        z_roms, z_fvcom = prepare_z_dimensions(z_roms.T, z_fvcom.T)
        ind1, ind2, weigths1, weigths2 = calc_interp_matrices(-z_roms, -z_fvcom)
        
        setattr(N4, f'vi_ind1_{c}', ind1) 
        setattr(N4, f'vi_ind2_{c}', ind2)
        setattr(N4, f'vi_weigths1_{c}', weigths1) 
        setattr(N4, f'vi_weigths2_{c}', weigths2)     
    return N4

def prepare_z_dimensions(z_roms, z_fvcom):
    '''
    So that it does not need to be done within the numba compiled routine
    '''
    if len(z_roms.shape) == 1:
        z_roms = z_roms[:, None]
    if len(z_fvcom.shape) == 1:
        z_fvcom = z_fvcom[:, None]
    return z_roms, z_fvcom
    
@njit
def calc_interp_matrices(z1, z2):
    '''
    Calculate matrices for linear interpolation.
    
    Parameters
    ----------
    z1: array (m x n) 
       The x-coordinate of the data points.

    z2: array (k x n) 
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
    indices1 = np.zeros(z2.shape, dtype = np.int64)
    indices2 = np.zeros(z2.shape, dtype = np.int64)
    weigths1 = np.zeros(z2.shape, dtype = np.float64)
    weigths2 = np.zeros(z2.shape, dtype = np.float64)

    for n in range(0, z2.shape[-1]):
        # look at the depth in this FVCOM grid point
        z1_n = z1[:, n]
        z2_n = z2[:, n] 
        
        # Difference between each sigma layer
        absdiff = numba_absdiff(z2_n, z1_n)
        ind1    = np.argmin(absdiff, axis=1)
        ind1, ind2, w1, w2 = find_inds_and_weights(ind1, z1_n, z2_n)

        indices1[:, n] = ind1
        indices2[:, n] = ind2
        weigths1[:, n] = w1
        weigths2[:, n] = w2

    return indices1, indices2, weigths1, weigths2

@njit
def numba_absdiff(z2_n, z1_n):
    '''
    Find the absolute difference between mother- and child mother depth levels at each fvcom points.
    Significantly faster than the broadcasting operation z2_n[:,None]-z1_n[None,:]
    '''
    out = np.zeros((z2_n.shape[0], z1_n.shape[0]))
    for i, first_depth in enumerate(z2_n):
        for j, second_depth in enumerate(z1_n):
            out[i, j] = np.abs(first_depth - second_depth)
    return out

@njit
def find_inds_and_weights(ind1, z1_n, z2_n):
    '''
    Find the nearest sigma layer(s) and find interpolation weights
    '''
    ind2 = np.zeros(len(ind1), dtype = np.int64)        
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
    zero_ind = z2_n-z1_n[ind1] == 0
    w1[zero_ind] = 1
    
    # Handle values in between
    between_ind = (~shallow_ind) & (~deep_ind) & (~zero_ind)
    smaller_ind = (z1_n[ind1] > z2_n) & between_ind
    ind2[smaller_ind] = ind1[smaller_ind] - 1
    
    larger_ind = (z1_n[ind1] < z2_n) & between_ind
    ind2[larger_ind] = ind1[larger_ind] + 1
    
    dz1 = z1_n[ind1[smaller_ind]] - z1_n[ind2[smaller_ind]]
    dz2 = z1_n[ind2[larger_ind]] - z1_n[ind1[larger_ind]]
    
    w2[smaller_ind] = (z1_n[ind1[smaller_ind]] - z2_n[smaller_ind]) / dz1
    w2[larger_ind] = (z2_n[larger_ind] - z1_n[ind1[larger_ind]]) / dz2
    
    w1[smaller_ind] = (z2_n[smaller_ind] - z1_n[ind2[smaller_ind]]) / dz1
    w1[larger_ind] = (z1_n[ind2[larger_ind]] - z2_n[larger_ind]) / dz2
    return ind1, ind2, w1, w2