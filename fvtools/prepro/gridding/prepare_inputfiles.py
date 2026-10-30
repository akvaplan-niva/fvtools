# Create .dat files for bathymetry coriolis etc.
import numpy as np

def write_FVCOM_bath(M, filename = 'smoothed_bathymetry_dep.dat', reference='cart'):
    '''1
    Write a FVCOM readable batymetry file
    -------------------------------------
    - Generates an ascii FVCOM 4.x format bathymetry from a Mesh object
    
    Input
    ----
    - M: Any object containing lat, lon, x, y and h variables.
    - filename = FVCOM_bathymetry file name.
    '''
    
    # Choose coordinate reference
    # ----
    if reference=='cart':
        x = M.x
        y = M.y
    else:
        x = M.lon
        y = M.lat

    # ----
    f = open(filename, 'w')
    f.write('Node Number = ' + str(len(x))+'\n')
    for xn, yn, h in zip(x, y, M.h):
        line = '{0:.6f}'.format(xn) + ' ' + '{0:.6f}'.format(yn) + ' ' + '{0:.6f}'.format(h)+'\n'
        f.write(line)
    f.close()
    print(f'- Stored the bathymetry to: {filename}')
