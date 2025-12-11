import xarray as xr
import numpy as np
from .base import BaseGrid

class ERA5(BaseGrid):
    def __init__(self, ncera5, Proj = None):
        '''
        Initialize grid variables for ERA5
        - ncrera5: path to era5 netcdf file
        - Proj: Projection object, use the one associated with the FVCOM_grid object you have initialized
        '''
        # We read from the grid. Glorys data is gridded to z-levels, so we need to
        self.Proj = Proj 
        with xr.open_dataset(ncera5) as df:
            self.lon, self.lat = np.meshgrid(df.lon, df.lat) # meshgrid lon,lat
            land_mask = df.lsm.isel(time=0).data # Assuming that wetting/drying isn't too dramatic in ERA5
            self.land_mask = land_mask > 0
            self.x, self.y = self.Proj(self.lon, self.lat)