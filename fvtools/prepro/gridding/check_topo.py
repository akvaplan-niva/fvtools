# ----------------------------------------------------------
#   Check how dense the topography points are in a domain
#   Plot the data on top of a georeferenced image for easy
#   interpreation.
# ----------------------------------------------------------

import numpy as np
import pyproj
import utm
import contextily as ctx
import matplotlib.pyplot as plt

def main(lat, lon, dptfile):
    '''
    Load data from filename to check the data density in the range in
    lat and lon.

    input:
    - lat     = [min, max]
    - lon     = [min, max]
    - dptfile = '/cluster/home/hes001/Smeshing/Data/NNtopo.txt'
    '''

    # Load the data
    print('Loading data from the text file...')
    if dptfile.split('.')[-1] == 'npy':
        depth = load_numpybath(dptfile)
    else:
        depth = load_textbath(dptfile)

    # Crop the data to the lat, lon limits
    print('Cropping the data')
    x,y,z = crop_data(lat,lon,depth)

    # Prepare a georeferenced image
    show_positions(x,y,fname = dptfile.split('/')[-1])

def crop_data(lat, lon, data):
    '''
    Crop the grid to the pre-defined box
    '''
    # convert the lat lon to utm
    # ----
    UTM33W     = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
    xlim, ylim = UTM33W(lon, lat, inverse=False)

    # Load x and y
    # ----
    x_data     = data[:,0]
    y_data     = data[:,1]
    z_data     = data[:,2]

    # Find data indices in the range
    # ----
    ind1       = np.logical_and(x_data >= xlim[0], x_data <= xlim[1])
    ind2       = np.logical_and(y_data >= ylim[0], y_data <= ylim[1])
    inds       = np.logical_and(ind1, ind2)

    # Crop the vectors
    # ----
    x_c        = x_data[inds]
    y_c        = y_data[inds]
    z_c        = z_data[inds]

    return x_c, y_c, z_c

def show_positions(x,y,fname):
    '''
    Prepare the georeferenced image
    '''
    WebMerc = pyproj.Proj(init='epsg:3857')
    WGS84   = pyproj.Proj(init='epsg:4326')
    UTM33   = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
    
    # 1. convert to web merc
    xe,ns       = pyproj.transform(UTM33,WebMerc,x,y)
    
    # 2. do some contextily stuff
    print('Downloading georef')
    src         = ctx.providers.Wikimedia.url
    img, extent = ctx.bounds2img(s=min(ns),n=max(ns),e=max(xe),w=min(xe),source=src)

    # 3. Show where we have data
    print('Visualizing...')
    plt.figure()
    plt.imshow(img, extent=extent)
    plt.scatter(xe,ns,c='r')
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([]) 
    plt.title('Points where we have topography data in '+fname)
    plt.show()

def load_numpybath(dptfile):
    """
    Load the bathymetry from a big numpy file
    """
    try:
        depth_data = np.load(dptfile)
    except:
        depth_data = np.load(dptfile, allow_pickle = True)

    return depth_data
        
def load_textbath(dptfile):
    """
    Load the depth file
    """
    # Load raw data
    print('-> Load: '+dptfile)
    try:
        depth_data = np.loadtxt(dptfile, delimiter=',')
    except:
      try:
          depth_data = np.loadtxt(dptfile)
      except:
        try:
            depth_data = np.loadtxt(dptfile, delimiter=' ')
        except:
            depth_data = np.loadtxt(dptfile, skiprows = 1, delimiter = ',', usecols = [0,1,2])

    numpy_bath_name = dptfile.split('.txt')[0]+'.npy'
    print('- Storing the full bathymetry in: ' + numpy_bath_name)
    np.save(numpy_bath_name, depth_data)

    return depth_data