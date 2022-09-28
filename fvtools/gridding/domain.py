import numpy as np
import matplotlib.pyplot as plt
import sys

import fvtools.grid.fvgrid as fvgrd
import fvtools.gridding.coast as coast
import scipy.interpolate as scint
import scipy.ndimage as ndimage
import pandas as pd
import math
import fvtools.gridding.coast as coast
import warnings
import subprocess
import os
from matplotlib.path import Path
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.path import Path
from scipy.spatial import cKDTree as KDTree

def read_mesh(meshfile):
    with open(meshfile) as f:
        nodenum = int(f.readline())
        points  = np.loadtxt(f, delimiter = ' ', max_rows = nodenum)
        trinum  = int(f.readline())
        tri     = np.loadtxt(f, delimiter = ' ', max_rows = trinum, dtype=int)
    return points, tri 

def plot_mesh(points, triangles):
    # remove triangles outside bounds
    # not sure how they can appear
    _triangles = []
    for triple in triangles:
        keep_triangle = True
        for t in triple:
            if not -1 < t < len(points):
                keep_triangle = False
        if keep_triangle:
                _triangles.append(triple)

    x, y = zip(*points)

    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(x, y, _triangles, 'g-', markersize=0.2, linewidth=0.2)

def gfunc(rmax, dfact, rfact, dev1, dev2, Ld, rcoast):
    x     = np.arange(0,Ld,10)
    r2    = rmax / rfact
    x2    = dfact * r2
    a1    = x2  / dev1
    a2    = (Ld - x2) / dev2
    xm    = 2*a1
    xm2   = xm +2 * a2
    gf    = r2 * (2 - (1 - np.tanh((x - xm) / a1))) / 2 + (rmax - r2) * (2 - (1 - np.tanh((x - xm2) / a2))) / 2
    gf    = gf - gf[0]
    res   = gf+rcoast 
    return res, x

def distfunc(rfact    = 35.0, 
             dfact    = 35.0, 
             Ld       = 50.0e3,  
             dev1     = 25.0, 
             dev2     = 25.0, 
             maxres   = None, 
             strres   = None, 
             obcnodes = None,
             resfield = None,
             polyparam='PolyParameters.txt',
             boundaryfile='output/boundary.txt', 
             islandfile='output/islands.txt'):
    '''
    Testing distance resolution function, plus estimate of necessary number of nodes
    All values except for rmax, Ld and rcoast can be adjusted within the widget.

    res     = distfunc(rfact=3.0, dfact=12.0, Ld=4.0e5,  dev1=6.0, dev2=4.0,
                       rcoast=100.0)
    rfact  - factor determining the near coastal length scales
    dfact  - factor determining the middle resolution from rcoast and rmax
    Ld     - typical length from coast to obc
    dev1   - factor determining near coastal gradient, higher number=steeper curve
    dev2   - (what is this then?)
    strres - Resolution of the structured grid used to estimate the necessary number of nodes
    '''
    # Finding the stuff neaded to estimate the number of nodes
    if strres is None:
        par    = pd.read_csv(polyparam, sep=';')
        strres = (par['max_res'].max()+par['min_res'].min())/2

    if maxres is None:
        par    = pd.read_csv(polyparam, sep=';')
        maxres = par['max_res'].max()

    grid = make_structgrid(strres, obcnodes, boundaryfile='output/boundary.txt', islandfile='output/islands.txt')

    # Check if there is a resolution field in the input folder
    grid['resolution_field'] = None
    if resfield is None:
        resfield = look_for_resfield()
        if resfield is not None:
            grid = add_resfield_data(grid, resfield)

    nodenum, dum  = get_numberofnodes(dfact/2, rfact/2, dev1/2, dev2/2, 
                                      Ld/2, grid, maxres, strres)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.40)
    rmax    = maxres
    res, x  = gfunc(rmax, dfact/2, rfact/2, dev1/2, dev2/2, Ld/2, grid['coast_res'].min())

    # The custom_function for standard inputs
    l,     = plt.plot(x, res, lw=2)
    ax     = plt.gca()
    ax.set_ylabel('grid resolution [m]')
    ax.set_xlabel('meters away from nearest coast')
    ax.margins(x=0)

    # Initializing the sliders
    axcolor  = 'lightgoldenrodyellow'
    axrfact  = plt.axes([0.17, 0.25, 0.65, 0.03], facecolor = axcolor)
    axdfact  = plt.axes([0.17, 0.20, 0.65, 0.03], facecolor = axcolor)
    axLd     = plt.axes([0.17, 0.15, 0.65, 0.03], facecolor = axcolor)  
    axdev1   = plt.axes([0.17, 0.10, 0.65, 0.03], facecolor = axcolor)
    axdev2   = plt.axes([0.17, 0.05, 0.65, 0.03], facecolor = axcolor)

    srfact   = Slider(axrfact, 'rfact', 0.1, rfact, valinit = rfact/2, valstep = rfact/1000.0)
    sdfact   = Slider(axdfact, 'dfact', 0.1, dfact, valinit = dfact/2, valstep = dfact/1000.0)
    sLd      = Slider(axLd,   'Ld', 0.1, Ld, valinit = Ld/2, valstep = Ld/1000.0)
    sdev1    = Slider(axdev1, 'dev1', 0.1, dev1, valinit = dev1/2, valstep = dev1/1000.0)
    sdev2    = Slider(axdev2, 'dev2', 0.1, dev2, valinit = dev2/2, valstep = dev2/1000.0)

    axnodenr = plt.axes([0.02,0.9,0.2,0.03])
    giver    = Button(axnodenr,'Nodes needed:')
    
    ax.set_title(f'You need ca. {int(nodenum)} nodes.')    

    # Creating the functions for updating the figure
    def update(val):
        rfact   = srfact.val
        dfact   = sdfact.val
        dev1    = sdev1.val
        dev2    = sdev2.val
        Ld      = sLd.val
        
        # The function
        res, x = gfunc(rmax,dfact,rfact,dev1,dev2,Ld,grid['coast_res'].min())
        
        # updating the figure
        l.set_data(x,res)
        #ax.set_ylim([0,res.max()+10])
        ax.set_xlim([0,Ld])
        ax.set_title('Press the button to get an estimate')
        fig.canvas.draw_idle()

    def nodenr(event):
        nodenum,theres = get_numberofnodes(sdfact.val, srfact.val, sdev1.val, 
                                           sdev2.val, sLd.val, grid, maxres, strres)
        ax.set_title(f'You need ca. {int(nodenum)} nodes.')

        plt.figure()
        plt.hist(theres,bins=150)
        plt.title('Histogram of the resolution used to get the estimate')
        plt.show(block=False)
    
    # Redrawing the figure when the slider is used
    srfact.on_changed(update)
    sdfact.on_changed(update)
    sdev1.on_changed(update)
    sdev2.on_changed(update)
    sLd.on_changed(update)

    giver.on_clicked(nodenr)
    plt.show(block=True)

def make_structgrid(strres, obcnodes, boundaryfile='output/boundary.txt', islandfile='output/islands.txt'):
    '''
    Creates structured grids with resolution res covering the part of the domain 
    which is not covered by land.

    res          = resolution of the structured grid
    boundaryfile = file containing the boundary polygon
    islandfile   = file containing island polygons
    '''
    # Initiate storage:
    grid   = {}

    # 1. Read the island and boundary
    try:
        pi, xi, yi, di  = coast.read_islands(islands_file   = islandfile)
    except:
        xi = np.empty(0)
        yi = np.empty(0)
        pi = np.empty(0)
        di = np.empty(0)
        print('- no islands')
    xb, yb, db, mob, obc = coast.read_boundary(boundary_file = boundaryfile)

    grid['coast_x']   = np.append(xi, xb)
    grid['coast_y']   = np.append(yi, yb)
    grid['coast_res'] = np.append(di, db)

    # 2. Create structured mesh covering everything
    tmp    = np.array(np.meshgrid(np.arange(xb.min(),xb.max()+strres, strres),
                                  np.arange(yb.min(),yb.max()+strres, strres)))
    xarr   = np.ravel(tmp[0,:,:])
    yarr   = np.ravel(tmp[1,:,:])

    # 3. Remove points falling outside the domain or within islands
    #    i.  Outside the outer boundary
    p      = Path(np.array([xb,yb]).T)
    flags  = p.contains_points(np.array([xarr,yarr]).T)
    xarr   = xarr[flags]
    yarr   = yarr[flags]
    
    #    ii. Points inside islands
    pi = np.array(pi)
    for i in np.unique(pi):
        p     = Path(np.array([xi[pi==i],yi[pi==i]]).T)
        flags = p.contains_points(np.array([xarr,yarr]).T)
        xarr  = xarr[flags==False]; yarr = yarr[flags==False]

    grid['x'] = xarr
    grid['y'] = yarr

    return grid

def structured_subdomains(xarr,yarr,dst,rcoast,areas):
    '''
    Divides grid into subdomains. Probably not needed anymore.
    '''
    area = []; indices = []
    for (index,tmp) in enumerate(areas):
        area.append(tmp)
        indices.append(index)
    
    grids     = dict(type='Subdomain grids',grids=[])
    xcov      = []; ycov  = []; dcov= []
    remaining = np.ones(xarr.shape,dtype=bool)
    for i in indices:
        p     = Path(np.array(area[i].exterior.coords.xy).transpose())
        flags = p.contains_points(zip(xarr,yarr))
        tmpd  = dict(type='Subdomain '+str(i), x = xarr[flags], y = yarr[flags], 
                     ds = dst[flags], rc = rcoast[flags])
        grids['grids'].append(tmpd)
        remaining[flags] = False

    if len(xarr[remaining] > 0):
        tmpd = dict(type='Remaining',x=xarr[remaining], y = yarr[remaining], 
                    ds = dst[remaining], rc = rcoast[remaining])
        grids['grids'].append(tmpd)
    
    return grids

def read_domainres(grids=None, filename='PolyParameters.txt'):
    par    = pd.read_csv(filename,sep=';')
    maxres = par.max_res
    minres = par.min_res
    if grids is None:
        n_it = 1
    else:
        n_it   = len(grids['grids'])
    minr   = []
    maxr   = []
    if (n_it) > len(minres):
        # The outermost grid may not be completely covered by the polygons.
        smin    = pd.Series([max(minres)],index=[n_it-1])
        smax    = pd.Series([max(maxres)],index=[n_it-1])
        minrres = minres.append(smin)
        maxrres = maxres.append(smax)

    return minres, maxres
    
def get_numberofnodes(dfact, rfact, dev1, dev2, Ld, grid, rmax, resolution):
    '''
    Just taking the distance from nearest coast into account.
    The result will be more realistic when including stuff like topores
    and pointres in the calculation.
    '''
    # 1. The surface area of each gridpoint is res**2
    area       = float(resolution**2) # To avoid integer errors
    nodes      = 0.0
    theres     = []
    x          = []
    y          = []

    for n in range(len(grid['x'])):
        res  = distfunc_onepoint(grid, 
                                 grid['x'][n], 
                                 grid['y'][n], 
                                 dfact = dfact, 
                                 rfact = rfact, 
                                 dev1 = dev1, 
                                 dev2 = dev2, 
                                 Ld = Ld, 
                                 rmax = rmax)
    
        if grid['resolution_field'] is not None:
            res = min(res, grid['resolution_field'][n])

        Atri   = (np.sqrt(3.0)/4.0)*res**2
        nodes += area/Atri
        theres.append(res)

    # Develop: Adjustable colorbar
    # --------------------------------------------------------------------
    plt.figure()
    plt.scatter(grid['x'],grid['y'],1,theres)
    plt.axis('equal')
    plt.title('Resolution in the domain.')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='resolution [m]')
    plt.show(block=False)

    print(' ')
    print(f'Given: {Ld=} {rfact=} {dfact=} {dfact=} {dev1=} {dev2=}')
    print(f'--> We need ~{int(nodes/1.8)} nodes.')
    return int(nodes/1.8), theres

def distfunc_onepoint(grid, xp, yp, rfact=3.0, dfact=12.0, Ld=4.0e5,  dev1=4.0, dev2=4.0, 
                      rmax=2400.0):
    # 1. Find the distance from point to coast
    x = np.sqrt((grid['coast_x']-xp)**2+(grid['coast_y']-yp)**2)

    # 2. Solve gfunc
    r2    = rmax / rfact
    x2    = dfact * r2
    a1    = x2  / dev1

    if (Ld-x2) <= 0: # Warn the user if gfunc decreases from the coast (since that violates smeshing assumptions)
        warning.warn('Increase Ld, dfact or rfact', RuntimeWarning)

    a2     = (Ld - x2) / dev2
    xm     = 2 * a1
    xm2    = xm +2 * a2
    gfunc  = r2 * (2 - (1 - np.tanh((x - xm) / a1))) / 2 + (rmax - r2) * (2 - (1 - np.tanh((x - xm2) / a2))) / 2
    gfunc0 = r2 * (2 - (1 - np.tanh((0.0 - xm) / a1))) / 2 + (rmax - r2) * (2 - (1 - np.tanh((0.0 - xm2) / a2))) / 2
    gfunc  = gfunc - gfunc0

    # determine the resolution based on f as described on the smeshing git
    return np.min(gfunc+grid['coast_res'])

def old_distfunc(rfact  = 3.0, 
                 dfact  = 12.0, 
                 Ld     = 4.0e5,  
                 dev1   = 4.0, 
                 dev2   = 4.0, 
                 rmax   = 2400.0, 
                 rcoast = 100):
    '''
    Testing distance resolution function

    res   = distfunc(rfact=3.0, dfact=12.0, Ld=4.0e5,  dev1=6.0, dev2=4.0,
                     rcoast=100.0, rmax=2400.0)
    rfact - factor determining the near coastal length scales
    dfact - factor determining the middle resolution from rcoast and rmax
    Ld    - typical length from coast to obc
    dev1  - factor determining near coastal gradient, higher number=steeper curve
    dev2  - factor determining the far field gradient
    rmax  - maximum grid resolution to converge against
    '''

    x     = np.arange(0,Ld,10)
    r2    = rmax / rfact
    x2    = dfact * r2
    a1    = x2  / dev1
    a2    = (Ld - x2) / dev2
    xm    = 2 * a1
    xm2   = xm +2 * a2
    gfunc = r2 * (2 - (1 - np.tanh((x - xm) / a1))) / 2 + (rmax - r2) * (2 - (1 - np.tanh((x - xm2) / a2)))
    #gfunc = gfunc - gfunc[0]
    #res   = gfunc + rcoast
    res = gfunc

    plt.plot(x, res)
    plt.show()

    return rfact, dfact, Ld,  dev1, dev2, rmax, rcoast, res

def smoothres(h, gridspacing, drelmax, ncount):
    ''' '''
    #dmax = 2 * (drelmax - 1) / (drelmax + 1)
    dx = gridspacing / 2
    hnew = np.copy(h)
    count = 1
    hs = h.shape
    dmax = drelmax / 2
    while count < ncount:
        DpX = (h[2:hs[0],1:hs[1]-1] - h[1:hs[0]-1,1:hs[1]-1]) / gridspacing
        DmX = (h[1:hs[0]-1,1:hs[1]-1] - h[0:hs[0]-2,1:hs[1]-1]) / gridspacing
        DpY = (h[1:hs[0]-1,2:hs[1]] - h[1:hs[0]-1,1:hs[1]-1]) / gridspacing
        DmY = (h[1:hs[0]-1,1:hs[1]-1] - h[1:hs[0]-1,0:hs[1]-2]) / gridspacing
        DpX[DpX>0] = 0
        DmX[DmX<0] = 0
        DpY[DpY>0] = 0
        DmY[DmY<0] = 0
        Dp = np.sqrt(np.square(DmX) + np.square(DpX) + np.square(DmY) + np.square(DpY))
        itest = len(Dp[Dp>dmax+dmax/100])
        print(str(count) + ': ' + str(itest))
        if itest == 0:
            break
        hnewtmp = hnew[1:hs[0]-1,1:hs[1]-1]
        htmp = h[1:hs[0]-1,1:hs[1]-1]
        hnewtmp[Dp > dmax] = htmp[Dp > dmax] +dx * (dmax - Dp[Dp > dmax])
        hnew[0,:] = hnew[1,:]
        hnew[:,0] = hnew[:,1]
        hnew[hs[0]-1,:] = hnew[hs[0]-2,:]
        hnew[:,hs[1]-1] = hnew[:,hs[1]-2]

        count = count + 1
        h = np.copy(hnew)

    return hnew


def old_inside_polygon(x, y, xp, yp, noisy = False):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies xp, yp.

    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    inside = np.empty([len(x),])
    for k in range(len(x)):
        if noisy:
            if np.mod(k, 10000) == 0:
                print(str(k) + ' of ' + str(len(x)))
        n = len(xp)
        inside[k] = False
        p1x = xp[0]
        p1y = yp[0]
        for i in range(1, n + 1):
            p2x = xp[i % n]
            p2y = yp[i % n]
            if y[k] > min(p1y, p2y):
                if y[k] <= max(p1y, p2y):
                    if x[k] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y[k] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x[k] <= xinters:
                            inside[k] = not inside[k]
            p1x, p1y = p2x, p2y
    inside = inside == 1
    return inside

def inside_polygon(x,y,xp,yp):
    '''
    Returns "True" if a point is in the polygon
    '''
    polypoints = zip(xp,yp)
    p     = Path(polypoints)
    flags = p.contains_points(zip(x,y))
    return flags

def prepare_resolution_data(xg, yg, res, boundaryfile, islandfile, pout = 10, nout = True):
    ''' '''
    # Get the OBC-indices from appdata, or ask the user to supply them
    #try:
    #    obcind = np.load('appdata/obcind.npy')
    #except:
    #    raise ValueError('Your obc-inds must be stored in the appdata folder as obcind.npy!')

    #As most of the points are equal to the maximum resolution, we first remove most of those.
    x = xg.ravel()
    y = yg.ravel()
    resolution = res.ravel()
    x1 = x[resolution == np.max(res)]
    y1 = y[resolution == np.max(res)]
    r1 = resolution[resolution == np.max(res)]
    l = len(x1)
    x1 = x1[::pout]
    y1 = y1[::pout]
    r1 = r1[::pout]
    x2 = x[resolution < np.max(res)]
    y2 = y[resolution < np.max(res)]
    r2 = resolution[resolution < np.max(res)]
    x = np.append(x1, x2)
    y = np.append(y1, y2)
    resolution = np.append(r1, r2)

    #Then we remove points outside domain
    print('Finding points inside boundary')
    xb, yb, distance, obc = coast.read_boundary(boundaryfile)
    inside = inside_polygon(x, y, xb, yb)
    x = x[inside]
    y = y[inside]
    resolution = resolution[inside]
    try:
        polynum, xi, yi, distance = coast.read_islands(islandfile)
        polynum = np.asarray(polynum)
        for n in np.arange(np.min(polynum), np.max(polynum)):
            print('Finding points outside island number ' + str(n) + ' of ' + str(np.max(polynum)))
            xp = xi[polynum==n]
            yp = yi[polynum==n]
            inside = inside_polygon(x, y, xp, yp)
            x = x[~inside]
            y = y[~inside]
            resolution = resolution[~inside]
    except:
        print('- Did not find any islands')
    ## Avoid this, better to use a topo field (more control)
    #Nudge towards max res near the obc
    #mr = np.max(res)
    #rad = np.empty(len(x),)
    #x_obc = xb[obcind]
    #y_obc = yb[obcind]
    #for n in range(len(x)):
    #    rad[n] = np.min(np.sqrt(np.square(x_obc - x[n]) + np.square(y_obc - y[n])))

   # weight = 1 - rad / obcrad
   # weight[rad > obcrad] = 0.0
   # resolution = resolution * (1 - weight) + weight * mr
    return x, y, resolution

def write_resolution(file, x, y, res):
    ''' '''
    with open(file, 'w') as f:
        n = len(x)
        for i in range(n):
            f.write('%.4f %0.4f %.4f\n' % (x[i],y[i],res[i]))

def pointres(resg, pointresfile, maxres, lscale, boundaryfile = '../coastal_resolution/output/boundary.txt'):
    ''' '''
    xb, yb, distance, obc = coast.read_boundary(boundary_file = boundaryfile)

    # Creating regular grid
    # ----
    xtmp     = np.arange(np.min(xb), np.max(xb), resg)
    ytmp     = np.arange(np.min(yb), np.max(yb), resg)
    xg, yg   = np.meshgrid(xtmp, ytmp)

    # Reading points and resolution
    # ----
    points   = np.loadtxt(pointresfile)
    s        = xg.shape
    res      = maxres * np.ones(s)

    # loop to set resolution in all points
    # ----
    for n in range(points.shape[0]):
        x0   = points[n][0]
        y0   = points[n][1]
        res0 = points[n][2]
        dist = np.sqrt(np.square(xg - x0) + np.square(yg - y0))
        res2 = maxres * (1- np.exp(-np.square(dist) / np.square(lscale))) + res0
        res  = np.minimum(res, res2)
    return xg, yg, res

def topores(resg = None, 
            sigma = None, 
            rx1max = None, 
            min_depth = None,
            max_res = None, 
            min_res = None, 
            drelmax = None, 
            ncount = None,
            boundaryfile = None,
            topofile = '/tos-project1/NS9067K/apn_backup/Topo/NordNorgeTopo.txt'):

    '''
    Set the resolution resolution in the domain to minimize sigma coordinate errors
    - resg:         Resolution of the structured grid used
    - sigma:        Number of sigma layer distribution
    - rx1max:       Depth difference parameter
    - min_depth:    Minimum depth in the run
    - max_res:      Maximum resolution
    - min_res:      Minimum resolution
    - drelmax:      Mesh smoothness parameter (0.1?)
    - ncount:       Number of iterations for smoothing?
    - boundaryfile: Place where the smoothed coastline is
    '''
    xb, yb, distance, obc = coast.read_boundary(boundary_file = boundaryfile)
        
    print('- Loading topofile and cropping topo ....')
    if topofile.split('.')[-1] == 'npy':
        topo = np.load(topofile)
    else:
        try:
            topo = np.loadtxt(topofile)
        except:
            topo = np.loadtxt(topofile, skiprows = 1, delimiter = ',', usecols = [0,1,2])

        new_topofile = topofile.split('.txt')[0]+'.npy'
        np.save(new_topofile, topo)
        print('For future reference:\n- ' + topofile + '\nwas stored as:\n- '+new_topofile)

    # Create regular grid
    xtmp   = np.arange(np.min(xb), np.max(xb), resg)
    ytmp   = np.arange(np.min(yb), np.max(yb), resg)
    xg, yg = np.meshgrid(xtmp, ytmp)

    print('- Interpolating topography to mesh')
    gtopo = coast.croptopo(topo, (np.min(xb) - 10000, np.min(yb) - 10000, np.max(xb) +  10000, np.max(yb) + 10000))
    h = scint.griddata((gtopo[:,0], gtopo[:,1]), gtopo[:,2], (xg, yg))
    h[np.isnan(h)] = 0.0
    h[h < min_depth] = min_depth

    # Filter the raw topography
    print('- Filter the raw topography')
    h = ndimage.filters.gaussian_filter(h, 0.5)

    print('- Computing depth gradients and finding required resolution')
    dhx, dhy = np.gradient(h, resg)
    hg  = np.sqrt(np.square(dhx) + np.square(dhy))
    hg[hg < 1.0e-20] = 1.0e-20
    sps = sigma[1:len(sigma)] + sigma[0:len(sigma)-1]
    sms = sigma[1:len(sigma)] - sigma[0:len(sigma)-1]
    R   = max(np.abs(sps/sms))
    res = 2 * h * rx1max / ((R - rx1max) * hg)
    res[res > max_res] = max_res
    res[res < min_res] = min_res

    print('- Smooth the topo.resolution field')
    ress = smoothres(res, resg, drelmax, ncount)

    return xg, yg, res, ress

def crop_resfield(xg, yg, res, maxres, drelmax, resg, ncount, topocrop):
    '''
    Smooth transition of topofile based on a boundary polygon(s)
    - Developed for use in huge domains where we only want to pay special interest to small parts
    '''
    x,y,npol = coast.read_map(topocrop)

    # Reshape to get vector
    xg_r  = np.ravel(xg)
    yg_r  = np.ravel(yg)
    res_r = np.ravel(res)

    # Identify points outside of the field where we want topores to apply
    inds  = np.zeros(len(xg_r), dtype=bool)
    for i in range(len(np.unique(npol))):
        thispol = np.where(npol == i+1)
        ind     = inside_polygon(xg_r, yg_r, x[thispol], y[thispol])
        inds[ind] = True

    # Set the max resolution to the points outside
    res_r[~inds] = maxres

    # Reshape bach to matrix
    xg_c   = xg.reshape(*xg.shape)
    yg_c   = yg.reshape(*yg.shape)
    res_c  = res_r.reshape(*res.shape)

    # Smooth the domain
    ress_c = smoothres(res_c, resg, drelmax, ncount)

    # return the cropped resolution field
    return xg_c, yg_c, ress_c

def look_for_resfield():
    '''
    Look for files in input, return potential  
    '''
    input_files  = os.listdir('./output/')
    normal_files = ['boundary.txt','islands.txt','.gitkeep']
    revised_if   = [file for file in input_files if file not in normal_files]
    if any(revised_if):
        print('-----------------------------------------------------------------------------------')
        if len(revised_if) > 1:
            print('There are '+str(len(revised_if)) +' files that potentially contain resolution fields.')
            print('Should either of these be used in the number-of-nodes estimate, and if so: which?')
            print('Yes: path to file. No: Any other keypress')
            for file in revised_if:
                print('- input/'+file)
            resfield = subprocess.check_output('read -e -p "Which file: " var ; echo $var', shell=True).rstrip()
                
            if resfield.split('/')[-1] not in revised_if:
                print(' ')
                print('Ok, a resolution field will not be used in the estimate.')
                resfield = None
            else:
                print(' ')
                print(resfield+' will be used when estimating the target resolution.')

        else:
            print(revised_if[0]+' is potentially holding a resolution field.\n '+\
                  'Should it be included in the number-of-nodes estimate?')
            print('input/'+revised_if[0])
            answer = raw_input('Yes: y/ No [n]')
            if answer == 'y':
                resfield = 'input/'+revised_if[0]
            else:
                resfield = None
        print('-----------------------------------------------------------------------------------')

    else:
        resfield = None

    return resfield

def add_resfield_data(grid, resfield):
    '''
    Adds resfield data to the grid dictionary.
    '''
    # Load topores
    print('- load the resolution field')
    data = np.loadtxt(resfield)

    # Find the nearest griddata point
    x_df = grid['x'];  y_df = grid['y']
    x_re = data[:,0];  y_re = data[:,1]

    # Store as point-arrays
    p_df = np.array([x_df, y_df]).transpose()
    p_re = np.array([x_re, y_re]).transpose()

    print('- Connect resfield to the distfunc grid')
    tree   = KDTree(p_re)
    p,inds = tree.query(p_df)

    print('- Dump the resolution field data to the grid dict')
    nearest_resfield_point   = inds.astype(int)
    grid['resolution_field'] = data[nearest_resfield_point,2]

    return grid

def write_2dm(datafile, cangle = 5, new2dm = None):
    '''
    Reads a SMESHING output file and writes the output as a 2dm file
    - optional:
        cangle --> minimum acceptable angle in triangle corner
        new2dm --> name of new 2dm file (standard: new2dm = datafile - .txt)
    '''
    if new2dm is None:
        new2dm = datafile.split('.')[0]

    points, trangs = read_mesh(datafile)

    print('Raw mesh:')
    theta  = trangles(points,trangs)
    plt.show(block=False)

    while True:
        # Keep triangles with angles greater than cangle, delete the rest
        gtc    = np.where(theta.min(axis=1)>cangle)
        plt.figure()
        plt.triplot(points[:,0],points[:,1],trangs,c='g',lw=0.2)
        plt.title('raw grid from SMESHING')
        plt.axis('equal')
            
        plt.figure()
        plt.triplot(points[:,0], points[:,1], trangs[gtc[0],:], c='g', lw=0.2)
        plt.title(f'modified grid after removing angles less than {cangle}')
        plt.axis('equal')
        plt.show(block = False)
        
        print('\n---------------------- ')
        print('Old number of triangles: '+str(len(trangs[:,0])))
        print('New number of triangles: '+str(len(gtc[0])))
        print(' ')

        if input('Good enough? y/[n] ').lower()=='y':
            break

        cangle = float(input('Enter the new critical angle:\n'))

    # Show the angles that are less than 35 degrees
    # -------------------------------------------------------------------
    print('\nSlightly refined mesh:')
    angles    = check_angles(trangs[gtc[0],:], points, cang=35)
    
    # Look for triangles with 2 edges toward land
    # -------------------------------------------------------------------
    neighbors = look_for_neighbors(points, trangs[gtc[0],:])

    # Write the mesh to a .2dm file
    # ----
    trangs  = trangs[gtc[0],:]+1 # to get format that SMS is happy with
    newfile = new2dm+'.2dm'
    fid     = open(newfile,'w')

    # write triangulation
    # ----
    for i in range(len(trangs[:,0])):
        fid.write('E3T '+str(i+1)+' '+str(trangs[i,0])+' '+str(trangs[i,1])+\
                  ' '+str(trangs[i,2])+' 1\n')

    # write node positions
    # ----
    for i in range(len(points[:,0])):
        fid.write('ND '+str(i+1)+' '+str(points[i,0])+' '+str(points[i,1])+' 0.00000001\n')

    fid.close()

def trangles(p,t):
    '''
    Reads points and triangulations from SMESHING and removes triangles 
    with too small angles
       C
      / \
     /___\
    A     B
    Solves the equation cos(theta) = a*b/|a||b|
    '''
    xn    = p[:,0]; yn = p[:,1]
    x     = np.array([xn[t[:,0]],xn[t[:,1]],xn[t[:,2]]])
    y     = np.array([yn[t[:,0]],yn[t[:,1]],yn[t[:,2]]])
    
    # cos(theta) = a * b / |a||b|
    # Store each side as vectors
    AB    = np.array([x[0,:]-x[1,:], y[0,:]-y[1,:]])
    BC    = np.array([x[1,:]-x[2,:], y[1,:]-y[2,:]])
    CA    = np.array([x[2,:]-x[0,:], y[2,:]-y[0,:]])
    
    # length of each triangle side
    lAB   = np.sqrt(AB[0,:]**2+AB[1,:]**2)
    lBC   = np.sqrt(BC[0,:]**2+BC[1,:]**2)
    lCA   = np.sqrt(CA[0,:]**2+CA[1,:]**2)
    a     = [lAB.min(), lBC.min(), lCA.min()]
    b     = [lAB.max(), lBC.max(), lCA.max()]
    
    print(f'Minimum triangle wall length (resolution): {min(a)} m')
    print(f'Maximum triangle wall length (resolution): {max(b)} m')
    
    # dot products
    ABAC  = -(AB[0,:]*CA[0,:]+AB[1,:]*CA[1,:])
    BABC  = -(AB[0,:]*BC[0,:]+AB[1,:]*BC[1,:])
    CABC  = -(CA[0,:]*BC[0,:]+CA[1,:]*BC[1,:])
    
    # Get the angle (in degrees)
    theta = np.array([np.arccos(ABAC/(lAB*lCA)), \
                      np.arccos(BABC/(lAB*lBC)), \
                      np.arccos(CABC/(lCA*lBC))])

    # This will be removed soon...
    theta[np.isnan(theta)] = 0
    theta = theta*360.0/(2*np.pi) # radians to degrees
    
    # Find number of corners < 35*
    th_raveled = np.ravel(theta)
    gt35       = np.where(th_raveled<=35.0)
    print(f'There are {len(th_raveled[gt35])} corners less than 35 degrees in this mesh\n')

    fig, ax = plt.subplots(2,1)
    ax[0].hist(lAB.ravel(),bins=80)
    ax[0].set_title('Histogram of mesh resolution')
    ax[0].set_xlabel('resolution')
    ax[0].set_ylabel('# triangle corners')

    ax[1].hist(theta.ravel(),bins=80)
    ax[1].set_title('Histogram of triangle corner angles')
    ax[1].set_xlabel('angle')
    ax[1].set_ylabel('# triangle corners')
    
    return theta.T

# ----------------------------------------------------------------------------------------------
#   Basic QC stuff before you pass this to BuildCase
# ----------------------------------------------------------------------------------------------

def check_2dm(my2dm):
    '''
    Check if your cleaned 2dm file is any good
    - Are the angles good?
    - Are there any bad land-triangles?
    '''
    print('Loading the grid\n')
    try:
        triangle, nodes, X, Y, Z, types = fvgrd.read_sms_mesh(my2dm)
    except ValueError:
        raise ValueError('Make sure to save the file with a nodestring')

    points    = np.array([X,Y]).transpose()

    print('Calculating angles and resolutions\n')
    #trangs    = trangles(points, triangle)
    
    # Check if the angles are less than 35 degrees
    # -------------------------------------------------------------------
    try:
        angles    = check_angles(triangle, points, cang=35)
    except IndexError:
        raise IndexError('Remember to renumber the nodes before storing the .2dm file!')

    # Look for triangles with 2 edges toward land
    # -------------------------------------------------------------------
    neighbors = look_for_neighbors(points, triangle)

    if angles and neighbors:
        print('-------\nThe mesh is good to go! :)')

    elif angles and not neighbors:
        print('\nFix the triangle(s) with only one neighbor\n')
        print('--> Open the 2dm in SMS and fix it.')

    elif neighbors and not angles:
        print('\nFix the angles!')
        print('--> Open the 2dm in SMS and fix it.')

    elif not neighbors and not angles:
        print('\nFix both the angles and the neighbors!')
        print('--> Open the 2dm in SMS and fix it.')

def look_for_neighbors(points, tris):
    '''
    See if the triangles have three neighbors
    '''
    from matplotlib.tri import Triangulation
    tri      = Triangulation(points[:,0],points[:,1],triangles=tris)
    near_tri = tri.neighbors
    nnbor    = []

    print('Counting number of neighbors for each triangle...')
    for i in range(len(near_tri[:,0])):
        nnbor.append(np.where(near_tri[i,:]==-1)[0].size)
    nnbor = np.array(nnbor)

    # If there are bad triangles present
    # ------
    bad_tris = np.where(nnbor == 2)[0]
    if len(bad_tris)==1:
        print('--> Found a triangle with just one neighbor\n')
        good = False

    elif len(bad_tris)>1:
        print('--> Found ' + str(len(bad_tris)) + ' triangles with just one neighbor\n')
        good = False

    elif len(bad_tris)<1:
        print("--> Didn't find bad triangles\n")
        good = True

    if good==False:
        plt.figure()
        plt.triplot(points[:,0], points[:,1], tris, c='g', lw=0.2)
        bad_tri_pts = points[tris[bad_tris,:],:]
        for i in range(len(bad_tri_pts[:,0,0])):
            # First round: (initializing the vector)
            if i == 0:
                pts_this_tri = bad_tri_pts[i,:,:]
                xbad = pts_this_tri[:,0]
                ybad = pts_this_tri[:,1]

                # Close the loop
                xbad = np.append(xbad,pts_this_tri[0,0])
                ybad = np.append(ybad,pts_this_tri[0,1])  

            # All other rounds: 
            else:
                pts_this_tri = bad_tri_pts[i,:,:]
                xbad = np.append(xbad,pts_this_tri[:,0])
                ybad = np.append(ybad,pts_this_tri[:,1])
                
                # Close the loop
                xbad = np.append(xbad,pts_this_tri[0,0])
                ybad = np.append(ybad,pts_this_tri[0,1])                

            # Make sure the triangles are plotted one by one
            xbad = np.append(xbad, np.nan)
            ybad = np.append(ybad, np.nan)
            
        plt.plot(xbad, ybad, 'or')

        plt.title('Triangle(s) with just one neighbour (shown in red)')
        plt.axis('equal')
        plt.show(block = False)

    return good

def check_angles(triangles, points, cang=35):
    '''
    Checks if the triangles are < cang, and shows which triangles needs to be fixed.
    '''
    # If there are no bad angles, then we don't need to report anything
    angles = True

    # Find the angles
    trangs = trangles(points, triangles)

    # Shows the bad triangles
    if np.any(np.where(trangs<cang)):
        angles = False
        print('\nThere are sharp triangles in the mesh, these are marked with red\n')

        plt.figure()
        plt.triplot(points[:,0], points[:,1], triangles, c='g', lw = 0.2)
        
        # Create new tri object
        bad,vind    = np.where(trangs<35)
        bad_tri_pts = points[triangles[bad,:],:]

        # Visualize
        for i in range(len(bad_tri_pts[:,0,0])):
            # First round: (initializing the vector)
            if i == 0:
                pts_this_tri = bad_tri_pts[i,:,:]
                xbad = pts_this_tri[:,0]
                ybad = pts_this_tri[:,1]

                # Close the loop
                xbad = np.append(xbad,pts_this_tri[0,0])
                ybad = np.append(ybad,pts_this_tri[0,1])  

            # All other rounds: 
            else:
                pts_this_tri = bad_tri_pts[i,:,:]
                xbad = np.append(xbad,pts_this_tri[:,0])
                ybad = np.append(ybad,pts_this_tri[:,1])
                
                # Close the loop
                xbad = np.append(xbad,pts_this_tri[0,0])
                ybad = np.append(ybad,pts_this_tri[0,1])                

            # Make sure the triangles are plotted one by one
            xbad = np.append(xbad, np.nan)
            ybad = np.append(ybad, np.nan)
        
        plt.plot(xbad, ybad, 'or')

        plt.title('Triangles with too sharp angles (shown in red)')
        plt.axis('equal')
        plt.show(block = False)
    
    return angles
