# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.tri as tri
import matplotlib.image as mpimg
from PIL import Image


#from fvcom_grd import FVCOM_grid

#grd = FVCOM_grid()

def plot_field(x, y, tri, field, nodes=True):
    '''Plot FVCOM scalar field'''

    fig, ax = plt.subplots()
    patches = []
    
    num_polygons = tri.shape[0]
    num_sides = 3

    for n in range(num_polygons):
        polygon = Polygon(np.array([x[tri[n]], y[tri[n]]]).transpose(), True)
        patches.append(polygon)
    p = PatchCollection(patches, edgecolors=None)
    
    if nodes:
        colors = np.squeeze((field[tri[:,0]] + field[tri[:,1]] + field[tri[:,2]]) / 3.0) 
    else: 
        colors = field
    p.set_array(colors)
    ax.add_collection(p)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.axis('equal')
    plt.colorbar(p, ax=ax)
    plt.show()
    return p

def geoplot(filepath, offx=0.0, offy=0.0, plot = True):
    """
    Mandatory: 
    filepath
    Give the routine the name of the file, and it will return the georeference
    as well as the picture to plot.

    Optional:
    offx: Offset in x-direction (if you want to change the axis)
    offy: Offset in y-direction ( --||-- )
    plot: Set "plot=True" if you wish to let geoplot do the plotting for you
    """
    
    if filepath[-3:]=='jpg':
        filepath = filepath.split('.jpg')[0]

    if filepath[-3:]=='jgw':
        filepath = filepath.split('.jgw')[0]


    # Last inn selve bildet
    img   = mpimg.imread(filepath+'.jpg')
    

    # Last inn x-y koordinater, samt steglengde
    tmp   = open(filepath+'.jgw')
    tmp   = tmp.read().split('\n')
    A     = float(tmp[0])
    D     = float(tmp[1])
    B     = float(tmp[2])
    E     = float(tmp[3])
    xw    = float(tmp[4])
    yn    = float(tmp[5])

    # Finner antall pixler i hver retning
    #Dette sluttet å funke plutselig, men PIL ser ut til å gjøre jobben:    
    #    xpx   = len(img[0,:,0])
    #    ypx   = len(img[:,0,0])
    xpx,ypx=Image.open(filepath+'.jpg').size

    # Gjør transformasjonen for å få ut x-y på andre kanten
    x     = np.zeros((xpx, ypx))
    y     = np.zeros((xpx, ypx))

    xe    = A*xpx + B*ypx + xw
    ys    = D*xpx + E*ypx + yn

    # Lagrer koordinatene
    class coords: pass
    coords.xe = xe-offx
    coords.xw = xw-offx
    coords.ys = ys-offy
    coords.yn = yn-offy
    
    if plot:
        plt.imshow(img, extent = [coords.xw,coords.xe,coords.ys,coords.yn])
        plt.autoscale(False)
    return coords, img

def cvplot(nodes,fname):
    '''
    give nodenumber (as list) and name to file containing positions and nearby nodes/elements,
    and this will return polygons for the control volume at each node.

    for now: Show the control volume around some ocean-nodes (boundary nodes require more thinking)

    Example of use:
    from fvcom_plot import geoplot, cvplot
    geoplot('printout.jpg')
    cvplot([node1,node2,node3],'output_0001.nc')
    '''
    #nodes  = [40000,40001,40002]
    #fname  = '/tos-project1/NS9067K/apn_backup/FVCOM/Havard/Danielsnes/autumn/spinup/Danielsnes_0001.nc'

    # load grid metrics
    # -------------------
    d      = Dataset(fname)
    x      = d['x'][:]
    y      = d['y'][:]
    xc     = d['xc'][:]
    yc     = d['yc'][:]

    # load nearby surrounding nodes and elements
    # -------------------
    nbsn   = d['nbsn'][:].transpose()-1
    nbve   = d['nbve'][:].transpose()-1

    controls = {}

    # find the positions that correspond to each controlvolume in the selected nodes
    for node in nodes:
        # nodes surrounding node
        # -------
        nbsnhere = nbsn[node,np.where(nbsn[node,:]>0)][0]

        # elements surrounding node
        # -------
        elemshere = nbve[node,np.where(nbve[node,:]>0)][0]

        xmid      = (np.tile(x[node],len(nbsnhere))+x[nbsnhere])/2
        ymid      = (np.tile(y[node],len(nbsnhere))+y[nbsnhere])/2
        xcell     = xc[elemshere]
        ycell     = yc[elemshere]

        # connect xc and yc to draw the control volume
        # ------
        xcv       = []
        ycv       = []

        #plt.scatter(x,y)
        for i in range(len(xmid)):
            xcv.append(xmid[i])
            ycv.append(ymid[i])
    
            # find direction
            # ---
            if i < len(xmid)-1:
                xcvmid = (xmid[i]+xmid[i+1])/2
                ycvmid = (ymid[i]+ymid[i+1])/2
                dst    = np.sqrt((xcell-xcvmid)**2+(ycell-ycvmid)**2)
                ind    = np.argwhere(dst==dst.min())[0]

            xcv.append(xcell[ind])
            ycv.append(ycell[ind])

        controls[node] = zip(xcv,ycv)
    
    for node in nodes:
        xx,yy = zip(*controls[node])
        plt.fill(xx,yy,c='g')

    plt.axis('equal')
    return controls

# Windroses
def rose(u,v):
    '''
    Plot a rose plot based on FVCOM vector data
    '''
    from windrose import WindroseAxes

    speed     = np.sqrt(u**2+v**2)

    # Sketchy, but should be correct since:
    # ----
    # - arctan angle is relative to the x axis, not to the y-axis (shift by 90 degrees).
    # - windrose counts clockwise, arctan2 counts anti-clockwise
    # - windrose crashes if the angles are less than 0 or greater than 360
    direction = -np.arctan2(v,u)*(360/(2*np.pi))+360+90 # To get positive angles relative to north
    direction[np.where(direction>360)]+=-360
    
    ax = WindroseAxes.from_ax()
    ax.bar(direction, speed, normed=True)
    return ax

def geo_rose(point, u, v, width = 2000, geopath = None, title = None,
             maxvel = None, steps = None):
    '''
    Plot the windrose on top of a georeferenced image
    
    Example:
    point   = [d['xc'][cell], d['yc'][cell]]
    u       = d['u'][:, depth, cell]
    v       = d['v'][:, depth, cell]
    geopath = './printout.jpg'
    width   = 2000 [m], only used when geopath is None
    '''
    from windrose import WindroseAxes

    # Prepare the windrose data
    # ----
    speed      = np.sqrt(u**2+v**2)
    direction  = -np.arctan2(v,u)*(360/(2*np.pi))+360+90 # To get positive angles relative to north
    direction[np.where(direction>360)]+=-360

    # Get the georeferenced image
    # ----
    if geopath is not None:
        c, img     = geoplot(geopath, plot = False)
        extent     = [c.xw,c.xe,c.ys,c.yn]
        
    else:
        from fvtools.plot.geoplot import geoplot as newgp
        x          = [point[0]-width*1.2, point[0] + width*1.2]
        y          = [point[1]-width*1.2, point[1] + width*1.2]

        # Establish georeference
        gp         = newgp(x, y)
        img        = gp.img
        extent     = gp.extent
        
    # Prepare the figure
    # ----
    rect       = [0.05, 0.05, 0.9, 0.9]
    rect_polar = [0.20, 0.20, 0.6, 0.6]

    # We also need to rotate the subplot to align with true north!
    # Remember that utm north != true north. The windrose needs to be realligned,
    # and velocities from FVCOM needs to be adjusted. (But for now, we accept a small error)

    fig   = plt.figure(figsize=(8,8))

    # Georeferenced part
    ax    = fig.add_axes(rect)
    ax.imshow(img, extent = extent, interpolation = 'spline16')

    if title is not None:
        ax.set_title(title)
    
    # Windrose path
    wrose = WindroseAxes(fig, rect_polar)
    wrose.patch.set_alpha(0)

    # you can manually modify bins here: bins=np.arange(0,10,1)
    if maxvel is None:
        maxvel = np.max(speed)

    if steps is None:
        steps = 6
        
    bins  = np.linspace(0, maxvel, steps)
    wrose.bar(direction, speed, normed=True, bins=bins)
    ax2   = fig.add_axes(wrose)
    ax2.set_legend(loc = 'best', bbox_to_anchor=(0,0,0.15,0.15))

    # Make sure the windrose is centered over the right point
    ax.set_xlim([point[0]-width, point[0]+width])
    ax.set_ylim([point[1]-width, point[1]+width])
    ax.set_xticks([])
    ax.set_yticks([])

    return fig
