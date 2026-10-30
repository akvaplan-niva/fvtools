import numpy as np

class BaseGrid:
    '''
    Contains standard structured grid methods
    '''
    @property
    def lon_center(self):
        '''
        Longitude in the center of lat/lon cells
        '''
        return (self.lon[:-1,:-1] + self.lon[1:,1:])/2
    
    @property
    def lat_center(self):
        '''
        Latitude in the center of lat/lon cells
        '''
        return (self.lat[:-1,:-1] + self.lat[1:,1:])/2
    
    @property
    def lonlat_center(self):
        '''
        center coordinate as (n,2) array
        '''
        return np.array([self.lon_center.ravel(), self.lat_center.ravel()]).T
    
    @property
    def lonlat_data(self):
        '''
        data coordinate as (n,2) array
        '''
        return np.array([self.lon.ravel(), self.lat.ravel()]).T

    @property
    def x_center(self):
        x, _ = self.Proj(self.lon_center, self.lat_center)
        return x

    @property
    def y_center(self):
        _, y = self.Proj(self.lon_center, self.lat_center)
        return y
    
    def crop_grid(self, xlim, ylim):
        """Find indices of grid points inside specified domain"""
        ind1 = np.logical_and(self.x >= xlim[0], self.x <= xlim[1])
        ind2 = np.logical_and(self.y >= ylim[0], self.y <= ylim[1])
        return np.logical_and(ind1, ind2)