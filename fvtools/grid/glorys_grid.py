import xarray as xr
import numpy as np
from .common.base import BaseGrid

class GLORYS(BaseGrid):
    def __init__(self, ncglorys):
        """Accept an xarray Dataset or NetCDF path, without retaining file handles.

        Passing the already-open, sorted CMEMS Dataset ensures grid mask and
        timestep arrays have identical latitude/longitude ordering.
        """
        if isinstance(ncglorys, xr.Dataset):
            self._load_dataset_grid(ncglorys)
        else:
            with xr.open_dataset(ncglorys) as df:
                self._load_dataset_grid(df)

    def _load_dataset_grid(self, df):
        self.depth = np.asarray(df.depth.values, dtype=np.float64)
        self.lon, self.lat = np.meshgrid(
            np.asarray(df.longitude.values), np.asarray(df.latitude.values))
        self.zlevels = np.broadcast_to(
            self.depth[:, None, None],
            (len(self.depth), *self.lon.shape))
        self.land_3d = ~np.isfinite(np.asarray(df.so.isel(time=0).values))
        if self.land_3d.shape != self.zlevels.shape:
            raise ValueError('GLORYS salinity mask does not match its (depth,lat,lon) grid')
            
    @property
    def depth_below_geoid(self):
        """Deepest wet GLORYS z-level, or NaN on dry points."""
        wet_levels = np.where(~self.land_3d, self.zlevels, -np.inf)
        deepest = wet_levels.max(axis=0)
        return np.where(np.isfinite(deepest), deepest, np.nan)

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