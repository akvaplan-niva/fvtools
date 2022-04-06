import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Proj
from scipy import spatial

def baty_langfjorden():
    '''Example creating and saving topo for Langfjorden'''
    tr = make_bathymetry('langfjorden_baty', 
                   ['norshelf_data'], 
                   lon_lim=[22, 23.3], 
                   lat_lim=[70, 70.24], 
                   ds=[2000],
                   add_coast=True)

    #plot_topo(tr)
    return tr


def make_bathymetry(save_name,  
                    other_datasets, 
                    lon_lim=None, 
                    lat_lim=None, 
                    ds=[2000], 
                    add_coast=True):
    '''Combine different bathymetry datasets in one dataset.'''
    
    baty = pd.DataFrame(columns=["x", "y", "z", "source"])
    baty = add_gridded50(baty, '/data/FVCOM/Setup_Files/bathymetry/gridded50_xyz', lon_lim, lat_lim)
    baty = add_primaer(baty, '/data/FVCOM/Setup_Files/bathymetry/primaer_data_xyz', lon_lim, lat_lim)

    
    for dataset, ds in zip(other_datasets, ds):
        print(dataset)
        print(ds)
        b = pd.read_csv(dataset)
        b = crop_data(b, lon_lim, lat_lim)
        nearest = find_nearest(baty.x, baty.y, b.x, b.y)
        notTooClose = nearest[0] >= ds
        b = b[notTooClose]
        baty = baty.append(b)
    
    if add_coast:
        print('Add coastline')
        baty = add_coastline(baty, '/data/FVCOM/Setup_Files/bathymetry/Norgeskyst.txt') 
    
    baty.to_csv(save_name, index=False)
    return baty




def add_gridded50(baty, 
                  datafolder, 
                  lon_lim, 
                  lat_lim):
    '''Add data in from gridded data set to to baty if they are inside given area.'''
    
    infolder = os.listdir(datafolder) # All content in folder
    xyz_files = [f for f in infolder if f.split('.')[-1]=='xyz'] # xyz-files in folder
    for f in xyz_files:
        print(f)
        data = pd.read_csv(os.path.join(datafolder, f), sep=' ', names=['x', 'y', 'z'])
        data = crop_data(data, lon_lim, lat_lim)
        baty = baty.append(data)

    baty['source'] = 'g'
    return baty



def add_primaer(baty, 
                datafolder, 
                lon_lim, 
                lat_lim,
                ds=50):
    '''Add data from primaer data in xyz files to baty if they are inside given area.'''
    
    infolder = os.listdir(datafolder) # All content in folder
    xyz_files = [f for f in infolder if f.split('.')[-1]=='xyz'] # xyz-files in folder
    for f in xyz_files:
        print(f)
        data = pd.read_csv(os.path.join(datafolder, f), sep=',')
        data = crop_data(data, lon_lim, lat_lim)
        if len(data)>0:
            nearest = find_nearest(baty.x, baty.y, data.x, data.y)
            notTooClose = nearest[0] >= ds
            data = data[notTooClose]
            baty = baty.append(data)

    return baty

def find_nearest(x, y, xi, yi):
    ''' '''
    tree = spatial.KDTree(np.array([x, y]).T)
    q = tree.query(np.array([xi, yi]).T)
    return q


def crop_data(data, 
              lon_lim, 
              lat_lim, 
              proj="+proj=utm +zone=33W, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs"):
    
    '''Remove data outside given limits.'''

    utm33 = Proj(proj)
    xlim, ylim = utm33(lon_lim, lat_lim)

    ind1 = (data.x>=xlim[0]) & (data.x<=xlim[1])
    ind2 = (data.y>=ylim[0]) & (data.y<=ylim[1])

    ind = ind1 & ind2
    if len(ind)==0:
        data = data[0:0]
    else:
        data = data[ind]
    return data


def plot_topo(topo):
    for source in topo.source.unique():
        plt.plot(topo.x[topo.source==source], topo.y[topo.source==source], '.')

    plt.axis('equal')
    plt.show()


def add_coastline(topo, coast_file):
    ''' Add zero to topofile at coastline points.'''
    xmin = topo.x.min()
    xmax = topo.x.max()
    ymin = topo.y.min()
    ymax = topo.y.max()

    x, y = np.loadtxt(coast_file, delimiter=' ', skiprows=2, unpack=True)
    ind = (np.mod(x, 1) != 0) & (np.mod(y, 1) != 0)
    x = x[ind]
    y = y[ind]
    
    ind1 = ((x>=xmin) & (x<=xmax)) & ((y>=ymin) & (y<=ymax))
    x = x[ind1]
    y = y[ind1]
    z = np.zeros(len(x))
    source = ["c"] * len(x)
    
    xy = pd.DataFrame({'x': x, 'y': y, 'z': z, 'source': source})

    topo = topo.append(xy)
    return topo
    
