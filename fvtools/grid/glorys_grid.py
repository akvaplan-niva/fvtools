import xarray as xr
import numpy as np
from .base import BaseGrid

class GLORYS(BaseGrid):
    def __init__(self, ncglorys):
        '''
        Initialize grid variables for the GLORYS model
        '''
        # We read from the grid. Glorys data is gridded to z-levels, so we need to 
        with xr.open_dataset(ncglorys) as df:
            self.depth = df.depth.data # load depth vector, same for all grid cells
            self.lon, self.lat = np.meshgrid(df.longitude.data, df.latitude.data) # meshgrid lon,lat
            self.zlevels = np.tile(df.depth.data[:, None, None], (1, *self.lon.shape)) # make a depth matrix for all points
            self.land_3d = np.isnan(df.so.isel(time = 0)) # Identify which gridpoints/depths do not cover the ocean
            
    @property
    def depth_below_geoid(self):
        '''
        Water depth, depth below mean sea surface etc.
        '''
        dpt = np.copy(self.zlevels)
        dpt[self.land_3d] = np.nan
        return np.nanmax(dpt, axis = 0)

    @property
    def land_mask(self):
        '''
        Mask for land
        - True where there is land
        '''
        return np.isnan(self.depth_below_geoid)
    
    @property
    def h_rho(self):
        '''
        Use same name convention as ROMS so that we can use MatchTopo without too much fuss.
        '''
        return self.depth_below_geoid