# Object we use to load an AROME grid, and subsequently interpolate data from the AROME grid to the FVCOM mesh
import pyproj
import numpy as np
from functools import cached_property
from netCDF4 import Dataset

class AROMEbase:
    '''
    When dealing with old AROME data, we need to access data stored on several grids.
    '''
    @property
    def x_center(self):
        return (self.x[1:,1:]+self.x[1:,:-1])/2

    @property
    def y_center(self):
        return (self.y[1:,1:]+self.y[:-1,1:])/2

    def crop_grid(self, xlim, ylim):
        """Find indices of grid points inside specified domain"""

        ind1 = np.logical_and(self.x >= xlim[0], self.x <= xlim[1])
        ind2 = np.logical_and(self.y >= ylim[0], self.y <= ylim[1])
        return np.logical_and(ind1, ind2)

    def crop_extended(self, xlim, ylim):
        """ 
        The use of this will _only_ be correct if the the pp- and extended files share the same grid, but
        one of them are a cropped version of the other!
        """
        ind1 = np.logical_and(self.xex >= xlim[0], self.xex <= xlim[1])
        ind2 = np.logical_and(self.yex >= ylim[0], self.yex <= ylim[1])
        return np.logical_and(ind1, ind2)

class AROME_grid(AROMEbase):
    '''
    Object containing grid information about AROME atmospheric model grid.
    '''

    def __init__(self):
        """
        Read grid coordinates from nc-files.
        A bit of a mess since pp files uses a snipped version of the extracted grid.
        """
        # Assuming that the grid is unchanged, will probably lead to crashes in the future - but those are
        # problems we need to think about anyways, and in such we welcome them...
        # --------------
        year     = '2020'#str(date.year)
        month    = '01'#'{:02d}'.format(date.month)
        day      = '30'#'{:02d}'.format(date.day)
        basepath = f'https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month}/{day}'
        pathToEX = f'{basepath}/meps_mbr0_extracted_backup_2_5km_{year}{month}{day}T00Z.nc'
        pathToPP = f'{basepath}/meps_mbr0_pp_2_5km_{year}{month}{day}T00Z.nc'

        self.ncpp  = pathToPP
        self.ncex  = pathToEX
        self.name  = pathToPP.split('/')[-1].split('.')[0]
        self.naex  = pathToEX.split('/')[-1].split('.')[0]

        ncdata = Dataset(pathToPP, 'r')
        self.lonpp = ncdata.variables.get('longitude')[:]
        self.latpp = ncdata.variables.get('latitude')[:]
        ncdata.close()

        ncdata     = Dataset(pathToEX, 'r')
        self.lonex = ncdata.variables.get('longitude')[:]
        self.latex = ncdata.variables.get('latitude')[:]
        ncdata.close()

        UTM33W = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
        self.x, self.y = UTM33W(self.lonpp, self.latpp, inverse=False)
        self.xex, self.yex = UTM33W(self.lonex, self.latex, inverse=False)

    @cached_property
    def landmask(self):
        '''
        MetCoOp fractional landmask. Let "1" indicate ocean. We only use this to avoid
        land radiation to the ocean model. Should be avoided and replaced with an albedo adjustment at
        for shortwave. Not sure if longwave behaves similarly.
        '''
        _ncdata = Dataset(self.ncpp, 'r')
        _landfraction = (_ncdata.variables.get('land_area_fraction')[:]-1.0)*-1.0
        _ncdata.close
        return _landfraction

class newAROME_grid(AROMEbase):
    """
    Arome included new partners in February 2020. The computational domain was extended in the process.
    This class reads grids from that period.
    """
    def __init__(self):
        """
        Read grid coordinates from nc-files.
        A bit of a mess since pp files uses a snipped version of the extracted grid.
        """
        self.nc    = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive/2020/10/30/meps_det_2_5km_20201030T00Z.nc'
        self.name  = self.nc.split('/')[-1].split('.')[0]
        self.na    = self.nc.split('/')[-1].split('.')[0]
        self.ncdata = Dataset(self.nc, 'r')
        self.lon = ncdata.variables.get('longitude')[:]
        self.lat = ncdata.variables.get('latitude')[:]

        UTM33W = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
        self.x, self.y = UTM33W(self.lon, self.lat, inverse=False)
        super.__init__()

    @property
    def xex(self):
        return self.x

    @property
    def yex(self):
        return self.y

    @cached_property
    def landmask(self):
        ncdata = Dataset(self.nc)
        _landfraction = (ncdata.variables.get('land_area_fraction')[0,0,:]-1.0)*-1.0
        _ncdata.close()
        return _landfraction