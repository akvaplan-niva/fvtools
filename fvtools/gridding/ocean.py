# Work need to be done to work in python3!
import numpy as np
from coast import read_map

def write_boundary(mapfile, res = None):
    """
    Write a boundary file of the same format as coastline boundaries
    """
    # Interpret the polygon as a boundary ("mainland") file
    x,y,npol = read_map(mapfile)
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    write_smeshing_input(x, y, res)

def write_smeshing_input(x,y,res):
    """
    Write a boundary.txt file
    """
    fid  = open('./input/boundary.txt', 'w')
    fid.write(str(len(x))+'\n')
    for xp,yp in zip(x,y):
        line = str(xp) + ' ' + str(yp) + ' ' + str(res) + '\n'
        fid.write(line)
    fid.close()
