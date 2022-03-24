#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

import os
#import cPickle as pickle
import numpy as np

import pyproj
from netCDF4 import Dataset

class WRF_grid():
    """Object containing grid information about WRF atmospheric model grid."""

    def __init__(self, 
                 pathToFile='/work/hdj002/WRF3km/wrf_3km_norkyst800_2014.nc'):
        """Read grid coordinates from nc-file."""
        
        self.ncfile = pathToFile
        self.name = pathToFile.split('/')[-1].split('.')[0]
        ncdata = Dataset(pathToFile, 'r')
        self.lon = ncdata.variables.get('XLONG')[:]
        self.lat = ncdata.variables.get('XLAT')[:]
        
        UTM33W = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
        self.x, self.y = UTM33W(self.lon, self.lat, inverse=False)

    def crop_grid(self, xlim, ylim):
        """Find indices of grid points inside specified domain""" 
        
        ind1 = np.logical_and(self.x >= xlim[0], self.x <= xlim[1])
        ind2 = np.logical_and(self.y >= ylim[0], self.y <= ylim[1])
        
        return np.logical_and(ind1, ind2)
       



    def save(self, name=None):
        """Save WRF_object to file"""
        if name is None:
            name = self.name

        pickle.dump(self, open(name, "wb")) 






