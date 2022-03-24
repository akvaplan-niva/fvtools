import os
# ------------------------------------------------------------------------------
import gridding.coast as coast
import numpy as np
import pandas as pd

kyst = coast.prepare_gridding('input/boundary.txt',
                              'input/islands.txt',
                              'appdata/Polygon.txt',
                              'PolyParameters.txt',
                              np.arange(0,57))

par = pd.read_csv('PolyParameters.txt', sep=';')

min_depth = 5.0
kyst = coast.topography(kyst, par, min_depth, topofile = 'topofile.txt')

nlev = 35
du = 2.5
dl = 0.5
sigma = coast.sigma_tanh(nlev, dl, du)
obcres = 800.
rx1max = 6.
kyst = coast.resolution(kyst, obcres, topores = True, sigma = sigma, rx1max=rx1max, min_depth=min_depth)

coast.write_to_file(kyst, 'input')
