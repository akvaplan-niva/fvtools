# ---------------------------------------------------
#       Write ROMS ocean points to a shapefile
# ---------------------------------------------------
import numpy as np
import fiona as fi
import pandas as pd
import geopandas as gpd
import sys
import pyproj
from netCDF4 import Dataset
from scipy.spatial import KDTree
# ----------------------------------------------------

def main(latlim, lonlim, mother):
    '''
    Reads the mesh, identifies ocean points in a lat/lon
    range, and writes to a SMS readable shapefile in the ocean_points
    folder that appears in the working directory if the script ran
    successfully.

    ---

    Give it the latlim = [south, north], lonlim = [west, east]
    and mother = 'NS' (NorShelf) or 'NK' (NorKyst)
    '''
    
    # Get the grid metrics
    # ----
    grid    = get_grid(mother)

    # Crop the grid and return points to be written
    # ----
    cropped = crop_grid(grid, latlim, lonlim)

    # Write points to shapefile
    # ----
    write_shapefile(cropped)

    print('Done.')

def get_grid(mother):
    '''
    Get grid coordinates from a ROMS output file
    '''
    if mother == 'NK':
        path = '/tos-project1/NS9067K/apn_backup/ROMS/NK800_2019/'+\
               'norkyst_800m_his.nc4_2019092801-2019092900'

    elif mother == 'NS':
        path = 'https://thredds.met.no/thredds/dodsC/sea_norshelf_files/'+\
               'norshelf_qck_fc_20200305T00Z.nc'

    else:
        raise ValueError('Specify the mother grid as NK (norkyst) or NS (norshelf)!')

    # Load the netcdf
    d       = Dataset(path)
    
    # define vars to be extracted
    extract = ['lat_rho', 'lon_rho', 'lat_u', 'lon_u', 'lat_v', 'lon_v',\
               'mask_rho', 'mask_u', 'mask_v']
    
    new     = [field.replace('lat','y') for field in extract]
    new     = [field.replace('lon','x') for field in new]

    # initialize dict
    dumped  = {}

    # Loop over fields, dump them
    for this in extract:
        dumped[this] = d[this][:]

    return dumped

def crop_grid(grid, lats, lons):
    '''
    Crop the grid based on min/max lat and lon. Find the grid squares that do not have a direct
    connection to land.

    '''
    UTM33W     = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
    xlim, ylim = UTM33W(lons, lats,   inverse=False)
    grid       = get_utm(grid)

    # klargjør maskene
    # --> Litt usikker på om det er nødvendig med u og v? Antar at ROMS bruker no-splip
    #     grensebetingelse, så hvorfor ligger det ikke bare u og v = 0 i de punktene?
    mask_rho   = grid['mask_rho'][1:-1, 1:-1]
    mask_u1    = 0.5*(grid['mask_u'][:-2, :-1]  + grid['mask_u'][1:-1, :-1])
    mask_u2    = 0.5*(grid['mask_u'][1:-1, :-1] + grid['mask_u'][2:, :-1])

    mask_v1    = 0.5*(grid['mask_v'][:-1, :-2]  + grid['mask_v'][:-1, 1:-1])
    mask_v2    = 0.5*(grid['mask_v'][:-1, 1:-1]  + grid['mask_v'][:-1, 2:])

    # Generate the mask-grid
    x_rho      = grid['x_rho'][1:-1, 1:-1]
    y_rho      = grid['y_rho'][1:-1, 1:-1]

    x_u1        = grid['x_u'][:-2,:-1];  y_u1 = grid['y_u'][:-2,:-1]
    x_u2        = grid['x_u'][1:-1,:-1]; y_u2 = grid['y_u'][1:-1,:-1]
    x_u3        = grid['x_u'][2:,:-1];   y_u3 = grid['y_u'][2:,:-1]

    x_v1        = grid['x_v'][:-1,:-2];  y_v1 = grid['y_v'][:-1,:-2]
    x_v2        = grid['x_v'][:-1,1:-1]; y_v2 = grid['y_v'][:-1,1:-1]
    x_v3        = grid['x_v'][:-1,2:];   y_v3 = grid['y_v'][:-1,2:]    

    # Finner punkter der u,v og rho ikke grenser mot land
    mask       =  (mask_u1 == 1) & (mask_v1 == 1) & (mask_rho == 1) & \
                  (mask_u2 == 1) & (mask_v2 == 1)

    # crop the grid
    ind1       = np.logical_and(x_rho >= xlim[0], x_rho <= xlim[1])
    ind2       = np.logical_and(y_rho >= ylim[0], y_rho <= ylim[1])
    inds       = np.logical_and(ind1, ind2)

    x = []; y = []
    # u points
    x.extend(x_u1[inds][mask[inds]])
    x.extend(x_u2[inds][mask[inds]])
    x.extend(x_u3[inds][mask[inds]])

    y.extend(y_u1[inds][mask[inds]])
    y.extend(y_u2[inds][mask[inds]])    
    y.extend(y_u3[inds][mask[inds]])
    
    # v points
    x.extend(x_v1[inds][mask[inds]])
    x.extend(x_v2[inds][mask[inds]])
    x.extend(x_v3[inds][mask[inds]])

    y.extend(y_v1[inds][mask[inds]])
    y.extend(y_v2[inds][mask[inds]])
    y.extend(y_v3[inds][mask[inds]])

    # Keep unique points
    pts = np.array([x,y]).transpose()
    
    return np.unique(pts,axis=1)

def get_utm(grid):
    '''
    Recursively replace lat and lon with y and x
    '''
    UTM33W     = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
    fields     = ['u','v','rho']
    for field in fields:
        xg, yg           = UTM33W(grid['lon_'+field], grid['lat_'+field], inverse=False)
        grid['x_'+field] = xg
        grid['y_'+field] = yg
    
    return grid

def write_shapefile(vec):
    '''
    Reads a numpy array with [x,y] format, and writes as a shapefile
    '''
    df  = pd.DataFrame({'x': vec[:,0], 'y': vec[:,1]})
    gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.x, df.y))
    gdf.to_file('ocean_points')
