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
    
@njit(cache=True)
def calc_interp_matrices(z1, z2):
    """Fast, allocation-light equivalent of the legacy nearest-level interpolation.

    Inputs are (ROMS levels, points) and (FVCOM levels, points), with
    shallow-to-deep nonnegative depths; output shapes follow FVCOM levels.
    The nearest ROMS level is chosen using strict '<' to preserve np.argmin
    tie-breaking (first occurrence). No extrapolation, repeated levels OK.
    """
    if z1.ndim != 2 or z2.ndim != 2 or z1.shape[1] != z2.shape[1] or z1.shape[0] == 0:
        raise ValueError('Vertical interpolation requires two 2-D arrays with matching columns and at least one ROMS level')
    indices1 = np.zeros(z2.shape, dtype=np.int64)
    indices2 = np.zeros(z2.shape, dtype=np.int64)
    weights1 = np.zeros(z2.shape, dtype=np.float64)
    weights2 = np.zeros(z2.shape, dtype=np.float64)
    nlevels = z1.shape[0]
    nfvlayers = z2.shape[0]
    npoints = z1.shape[1]

    for n in range(npoints):
        for j in range(nlevels):
            if not np.isfinite(z1[j, n]):
                raise ValueError('Vertical interpolation depths must be finite')
        for k in range(nfvlayers):
            if not np.isfinite(z2[k, n]):
                raise ValueError('Vertical interpolation depths must be finite')
        for j in range(1, nlevels):
            if z1[j, n] < z1[j - 1, n]:
                raise ValueError('ROMS vertical levels must be sorted shallow-to-deep')

        first = z1[0, n]
        last = z1[nlevels - 1, n]
        for k in range(nfvlayers):
            target = z2[k, n]
            nearest = 0
            nearest_dist = abs(target - first)
            for j in range(1, nlevels):
                dist = abs(target - z1[j, n])
                if dist < nearest_dist:  # np.argmin breaks ties at first index
                    nearest = j
                    nearest_dist = dist
            indices1[k, n] = nearest
            z_nearest = z1[nearest, n]
            if target <= first or target >= last or target == z_nearest:
                weights1[k, n] = 1.0
            elif z_nearest > target:
                other = nearest - 1
                while other >= 0 and z1[other, n] == z_nearest:
                    other -= 1
                if other < 0:
                    raise ValueError('Cannot bracket FVCOM depth with ROMS sigma levels')
                indices2[k, n] = other
                dz = z_nearest - z1[other, n]
                weights2[k, n] = (z_nearest - target) / dz
                weights1[k, n] = (target - z1[other, n]) / dz
            else:
                other = nearest + 1
                while other < nlevels and z1[other, n] == z_nearest:
                    other += 1
                if other == nlevels:
                    raise ValueError('Cannot bracket FVCOM depth with ROMS sigma levels')
                indices2[k, n] = other
                dz = z1[other, n] - z_nearest
                weights2[k, n] = (target - z_nearest) / dz
                weights1[k, n] = (z1[other, n] - target) / dz
    return indices1, indices2, weights1, weights2

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
    larger_ind = (z1_n[ind1] < z2_n) & between_ind
    # Skip equal-depth neighbours. A collapsed sigma layer must not cause a
    # zero denominator or NaN weights; keep the legacy nearest-level choice.
    for i in range(len(ind1)):
        if smaller_ind[i]:
            j = ind1[i] - 1
            while j >= 0 and z1_n[j] == z1_n[ind1[i]]:
                j -= 1
            if j < 0:
                raise ValueError('Cannot bracket FVCOM depth with ROMS sigma levels')
            ind2[i] = j
        elif larger_ind[i]:
            j = ind1[i] + 1
            while j < len(z1_n) and z1_n[j] == z1_n[ind1[i]]:
                j += 1
            if j == len(z1_n):
                raise ValueError('Cannot bracket FVCOM depth with ROMS sigma levels')
            ind2[i] = j
    
    dz1 = z1_n[ind1[smaller_ind]] - z1_n[ind2[smaller_ind]]
    dz2 = z1_n[ind2[larger_ind]] - z1_n[ind1[larger_ind]]
    
    w2[smaller_ind] = (z1_n[ind1[smaller_ind]] - z2_n[smaller_ind]) / dz1
    w2[larger_ind] = (z2_n[larger_ind] - z1_n[ind1[larger_ind]]) / dz2
    
    w1[smaller_ind] = (z2_n[smaller_ind] - z1_n[ind2[smaller_ind]]) / dz1
    w1[larger_ind] = (z1_n[ind2[larger_ind]] - z2_n[larger_ind]) / dz2
    return ind1, ind2, w1, w2
