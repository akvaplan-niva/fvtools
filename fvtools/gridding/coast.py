import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mPolygon
from matplotlib.collections import PatchCollection 
import pandas as pd
import os
import sys
import scipy.interpolate as scint

from shapely.geometry import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
from shapely.ops import polygonize
from shapely.geometry import LineString


import fvtools.gridding.process_coastline as pc

from matplotlib.widgets import Slider, TextBox

class CoastLine():
    '''Coastline object containing polygons outlining islands and domain boundary'''

    def __init__(self, pathToBoundary='boundary.txt', pathToIslands='islands.txt'):
        '''Read boundary and island files'''
        self.boundary_file = pathToBoundary
        self.islands_file = pathToIslands

        xb, yb, db, mb, ob = read_boundary(pathToBoundary)
        p = [1]
        pi, xi, yi, di = read_islands(pathToIslands)

        x = np.append(xb, xi)
        y = np.append(yb, yi)
        d = np.append(db, di)
        p = p.append(pi)



def read_coast(pathToBoundary='boundary.txt', pathToIslands='islands.txt', obcind =[]):
    xb, yb, db, mob, obcb = read_boundary(boundary_file=pathToBoundary, obcind=obcind)
    p = [1] * len(xb)
    pi, xi, yi, di = read_islands(islands_file=pathToIslands)
    obci = np.zeros(len(xi))

    x = np.append(xb, xi)
    y = np.append(yb, yi)
    d = np.append(db, di)
    obc = np.append(obcb, obci)
    p = p + pi
    
    coast = pd.DataFrame({'polygon_number': p,
                         'x': x,
                         'y': y,
                         'distance': d,
                         'obc': obc}, 
                         columns = ['polygon_number', 'x', 'y', 'distance', 'obc'])

    return coast


def read_boundary(boundary_file='boundary.txt', obcind = []):
    '''Read file with grid boundary points
       input: boundary file (default = 'boundary.txt')
              obcind - obc indices (default = [])
       returns: x, y, distance, obc
    '''
    fid = open(boundary_file, 'r')
    x = np.empty(0)
    y = np.empty(0)
    distance = np.empty(0)
    mobility = np.empty(0)
    for line in fid:
        line = line.split()
        if len(line) == 1:
            numberOfPoints = int(line[0])
        elif len(line) == 2:
            x = np.append(x, float(line[0]))
            y = np.append(y, float(line[1]))
        elif len(line) == 3:
            x = np.append(x, float(line[0]))
            y = np.append(y, float(line[1]))
            distance = np.append(distance, float(line[2]))
        elif len(line) == 4:
            x = np.append(x, float(line[0]))
            y = np.append(y, float(line[1]))
            distance = np.append(distance, float(line[2]))
            mobility = np.append(mobility, float(line[3]))
    obc = np.zeros(len(x))
    obc[obcind] = 1
    fid.close()
    return x, y, distance, mobility, obc


def plot_coast(xb, yb, xi, yi, polynum):
    ''' '''
    plt.plot(xb, yb, 'k')
    p = np.array(polynum)
    for n in range(int(np.min(p)), int(np.max(p))+1):
        plt.plot(xi[p==n], yi[p==n], 'k')
        

def read_islands(islands_file='islands.txt'):
    '''Read file with island polygons
       input: islands file (default = 'islands.txt')
       returns: polygon_number, x, y, distance
    '''
    fid               = open(islands_file, 'r')    
    x                 = np.empty(0)
    y                 = np.empty(0)
    distance          = np.empty(0)
    number_of_islands = 0
    island_number     = 1
    polygon_number    = []
    for line in fid:
        line = line.split()
        if len(line) == 1:
            number_of_islands += 1
            island_number     += 1 

        else:
            #polygon_number = np.append(polygon_number, island_number)
            polygon_number.append(island_number)
            x = np.append(x, float(line[0]))
            y = np.append(y, float(line[1]))
            distance = np.append(distance, float(line[2]))
    

    fid.close()
    return polygon_number, x, y, distance


def separate_polygons(x, y, npol, bpath = os.getcwd()+'/boundary.txt', 
                      ipath = os.getcwd()+'/islands.txt'):
    # Including a quick hack when only one polygon
    u, indices = np.unique(npol, return_inverse = True)
    bind0      = int(u[np.argmax(np.bincount(indices))])

    fig, ax    = plt.subplots()
    plt.subplots_adjust(bottom=0.15)
    plot_coast([],[],x,y,npol)
    l,         = plt.plot(x[npol==bind0],y[npol==bind0],'.r')
    plt.xticks(([]))
    plt.yticks(([]))
    plt.axis('equal')
    plt.title('Close the figure when the boundary is highlighted')
    axcolor    = 'lightgoldenrodyellow'
    axbind     = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor = axcolor)
    sbind      = Slider(axbind, 'index',  int(npol.min()), int(npol.max()), valinit = bind0, valstep = 1)

    def update(val):
        bind = sbind.val
        plot_coast([],[],x,y,npol)
        l.set_data(x[npol==bind],y[npol==bind])
        fig.canvas.draw_idle()
        
    sbind.on_changed(update)
    plt.show(block=True)

    print('Storing the boundary and the islands in separate files;')
    print(bpath)
    print(ipath)

    # Write the boundary
    write_bound_isl(x[npol==sbind.val], y[npol==sbind.val], npol[npol==sbind.val]-(sbind.val-1), path=bpath)
    
    # Write the islands
    if npol.max()>1:
        a = np.not_equal(npol,sbind.val)
        npoln = npol[a]
        npoln[npoln>sbind.val] = npoln[npoln>sbind.val]-1
        write_bound_isl(x[a], y[a], npoln, path=ipath)
    else:
        write_bound_isl(0,0,0,path=ipath)
        
def find_boundary(boundaryfile = f'{os.getcwd()}boundary.txt'):
    xb, yb, db, mb, ob = read_boundary(boundaryfile)
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    plt.plot(xb, yb, '.k')
    l, = plt.plot(xb, yb, '.r')
    
    plt.xticks(([]))
    plt.yticks(([]))
    plt.axis('equal')
    plt.title('Close the figure when the open boundary is shaded in red')

    axcolor    = 'lightgoldenrodyellow'
    axbig      = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor = axcolor)
    axsmall    = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor = axcolor)
    axtext     = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor = axcolor)

    sbig       = Slider(axbig, 'to',  0, int(len(xb)), valinit = int(len(xb)), valstep = 1)
    ssmall     = Slider(axsmall, 'from',  0, int(len(xb)), valinit = 0, valstep = 1)
    text_box   = TextBox(axtext, 'from, to', initial = 'type 0,50 for inds 0 to 50')

    def update(val):
        big   = int(sbig.val)
        small = int(ssmall.val)
        if big>small:
            l.set_data(xb[small:big], yb[small:big])
        else:
            l.set_data(xb[big:small], yb[big:small])
        fig.canvas.draw_idle()

    def submit(text):
        inp   = text.split(',')
        big   = int(inp[1])
        small = int(inp[0])
        sbig.val = big
        ssmall.val = small
        if big>small:
            l.set_data(xb[small:big], yb[small:big])
        else:
            l.set_data(xb[big:small], yb[big:small])
        fig.canvas.draw_idle()
        
    sbig.on_changed(update)
    ssmall.on_changed(update)
    text_box.on_submit(submit)
    plt.show(block=True)

    print(f'Found that the indices {int(ssmall.val)} to {int(sbig.val)} covers the boundary')
    return int(ssmall.val), int(sbig.val)

def read_smscoast(coast_file):
    '''Read sms textfile with coast polygons
       input: sms file
       returns: polygon_number, x, y
    '''
    fid = open(coast_file, 'r')
    line = fid.readline() 
    line = fid.readline()
    x = np.empty(0)
    y = np.empty(0)
    polygon_number = np.empty(0)
    number_of_islands = 0
    island_number = 1
    for line in fid:
        line = line.split()
        if float(line[1]) == 1.0 or float(line[1]) == 0.0:
            number_of_islands += 1
            island_number += 1 
        else:
            #polygon_number = np.append(polygon_number, island_number)
            polygon_number = np.append(polygon_number, island_number)
            x = np.append(x, float(line[0]))
            y = np.append(y, float(line[1]))
    print(number_of_islands)

    fid.close()
    return polygon_number, x, y

def read_map(mapfile='Polylines.map'):
    '''
    Read arcs, consisting of nodes and vertices, from an sms mapfile
    '''

    fid = open(mapfile, 'r')
    x = np.empty(0)
    y = np.empty(0)
    polygon_number = np.empty(0, int)
    nx = np.empty(0)
    ny = np.empty(0)
    nid = np.empty(0, int)
    aid = 1

    while True:
        line = fid.readline()
        if line[0:2] == 'XY':
            nx = np.append(nx, float(line.split()[1]))
            ny = np.append(ny, float(line.split()[2]))
            line = fid.readline()
            nid = np.append(nid, int(line.split()[1]))
        elif line[0:3] == 'ARC' and len(line)<10:
            line = fid.readline()
            #aid = int(line.split()[1])
            line = fid.readline()
            line = fid.readline()
            a1 = int(line.split()[1])
            a2 = int(line.split()[2])
            line = fid.readline()
            # Add a mechanism for crealing lines without needing the verticies in between
            nverts = int(line.split()[1])
            ax = nx[nid == a1]
            ay = ny[nid == a1]
            k = 0
            while True:
                line = fid.readline()
                ax = np.append(ax, float(line.split()[0]))
                ay = np.append(ay, float(line.split()[1]))
                k = k+1
                if k == nverts:
                    break
            if (nx[nid==a1]!=nx[nid==a2])&(ny[nid==a1]!=ny[nid==a2]):
                ax = np.append(ax, nx[nid == a2])
                ay = np.append(ay, ny[nid == a2])
                lstr = 2
            else:
                lstr = 1
            x = np.append(x, ax)
            y = np.append(y, ay)
            pn = aid * np.ones(nverts + lstr)
            aid = aid + 1
            polygon_number = np.append(polygon_number, pn)
            line = fid.readline()
        elif line[0:4] == 'LEND':
            break
    fid.close()

    return x, y, polygon_number

def prepare_gridding(boundary, islands, polygons, poly_parameters, obcind=[], write=False):
    '''Combine distances and other parameters for each point into single file'''
    # Save the obc-indices for later
    np.save('obcind.npy',obcind)

    # Read coastline with distances
    coastline = read_coast(pathToBoundary=boundary, pathToIslands=islands, obcind=obcind)

    # Read polygons
    polyg = pc.read_poly_data(polygons)

    # Read parameters for each polygon
    par = pd.read_csv(poly_parameters, sep=';')


    outfile  = pd.DataFrame({'polygon_number': np.zeros(len(coastline)),
                         'x': coastline.x.values,
                         'y': coastline.y.values,
                         'distance': coastline.distance.values,
                         'obc': coastline.obc.values,
                         'min_res': np.zeros(len(coastline)),
                         'max_res': np.zeros(len(coastline)),
                         'points_across': np.zeros(len(coastline)),
                         'resolution': np.zeros(len(coastline)),
                         'distres': np.zeros(len(coastline)),
                         'topores': np.zeros(len(coastline)),
                         'boundary': np.zeros(len(coastline)),
                         'island_number': np.zeros(len(coastline)),
                         'h': np.zeros(len(coastline)),
                         'hgrad': np.zeros(len(coastline)),
                         'rx1': np.zeros(len(coastline)),
                         'mobility': np.ones(len(coastline))},
                         columns = ['polygon_number', 'x', 'y', 
                                    'distance', 'obc', 'min_res', 'max_res', 
                                    'points_across', 'resolution', 'distres', 
                                    'topores', 'boundary',
                                    'island_number', 'h', 'hgrad', 'rx1', 'mobility'])


    # Loop through coastline point and assign parameters to each point
    for index, row in coastline.iterrows():
        if np.mod(index, 500) == 0:
            print(str(index) + ' of ' + str(len(coastline)))

        p = Point((row.x, row.y))
        outfile.island_number[index] = row.polygon_number 
        if row.polygon_number == 1:
            outfile.boundary[index] = 1
        for n, pol in enumerate(polyg):
            if pol.contains(p):
                outfile.polygon_number[index] = par.poly_number[n]
                outfile.min_res[index] = par.min_res[n]
                outfile.max_res[index] = par.max_res[n]
                outfile.points_across[index] = par.points_across[n]
                break
    outfile.polygon_number[outfile.obc == 1.] = 0.0
    outfile.min_res[outfile.obc == 1.] = 1.0
    outfile.max_res[outfile.obc == 1.] = 100000.
    outfile.points_across[outfile.obc == 1.] = 1.0

    if write is not False:
        outfile.to_csv(write, sep=';', index=False)

    return outfile

def croptopo(topo, bounds):
    '''crops topo according to the limits in bounds'''

    ind = topo[:,0]<bounds[2]
    topo = topo[ind,:]
    ind = topo[:,0]>bounds[0]
    topo = topo[ind,:]
    ind = topo[:,1]<bounds[3]
    topo = topo[ind,:]
    ind = topo[:,1]>bounds[1]
    topo = topo[ind,:]

    return topo

def topography(kyst, par, min_depth, topofile = '/tos-project1/NS9067K/apn_backup/Topo/NordNorgeTopo.txt'):
    if os.path.isfile('NearCoastTopo.npy'):

        topo = np.load('NearCoastTopo.npy')

    else:
        # Load the topography
        print('Loading topofile and cropping topo ....')
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

        # crop it
        topo = croptopo(topo, (min(kyst.x) - 2000, min(kyst.y) - 2000, max(kyst.x) +  2000, max(kyst.y) + 2000))
        print('....done')
        

        x_close = np.empty(0)
        y_close = np.empty(0)
        z_close = np.empty(0)

        min_dist = 2000
        k=0
        dy0 = 1000.0
        y0 = np.arange(np.floor(min(kyst.y)/100000.)*100000., np.ceil(max(kyst.y)/100000.)*100000., 2*dy0)
        kyst_obc = kyst[kyst.obc == 0]

        for n in np.nditer(y0):
            print('Creating near-coast topo:  ' + str(n) + ' of ' + str(y0[len(y0)-1]))
            kyst_in = kyst_obc[np.abs(kyst_obc.y - n -dy0) <= dy0]
            if len(kyst_in) > 0:
                h = croptopo(topo, (min(kyst_in.x) - min_dist, n - min_dist, max(kyst_in.x) +  min_dist, n + 2*dy0 + min_dist))
                x = np.array(kyst_in.x)
                y = np.array(kyst_in.y)
                xtopo = h[:,0].reshape(len(h),1)
                ytopo = h[:,1].reshape(len(h),1)
                dist = np.sqrt(np.square(x - xtopo) + np.square(y - ytopo))
                ind = np.where(dist <= min_dist)
                ind = np.unique(ind[0])
                x_close = np.append(x_close, h[ind,0])
                y_close = np.append(y_close, h[ind,1])
                z_close = np.append(z_close, h[ind,2])
        topo = np.array([x_close, y_close, z_close]).T
        np.save('NearCoastTopo', topo)

    for i in np.arange(len(par)):
        kyst_in = kyst[(kyst.polygon_number == par.poly_number[i]) & (kyst.obc == 0)]
        if len(kyst_in.x) == 0:
            continue
        h = croptopo(topo, (min(kyst_in.x) - 2000, min(kyst_in.y) - 2000, max(kyst_in.x) +  2000, max(kyst_in.y) + 2000))
        if len(h) > 250000:
            dn = int(np.ceil(len(h)/250000.))
            dy = (h[len(h)-1,1] - h[0,1])/dn
            y0 = h[0,1]
            for il in np.arange(dn):
                print('Polygon: ' + str(par.poly_number[i]) + ' - subset: ' + str(il) + ' of ' + str(dn-1))
                C = kyst_in[(kyst_in.y >= y0) & (kyst_in.y <= y0 + dy)]
                y0 = y0 + dy
                if len(C) > 0:
                    h = croptopo(topo, (min(C.x) - 2000, min(C.y) - 2000, max(C.x) +  2000, max(C.y) + 2000))
                    print(h.shape)
                    mr = par.min_res[i]
                    depth = scint.griddata((h[:,0], h[:,1]), h[:,2], (C.x, C.y))
                    depth[depth < min_depth] = min_depth
                    x = np.array(C.x)
                    y = np.array(C.y)
                    x = x.reshape(len(x),1)
                    y = y.reshape(len(y),1)
                    xtopo = h[:,0]
                    ytopo = h[:,1]
                    dist = np.sqrt(np.square(x - xtopo) + np.square(y - ytopo))
                    ind = np.where(dist <= 2000)
                    dist = dist[ind]
                    dep = h[ind[1],2]
                    n = 0
                    for q,row in C.iterrows():
                        ik = np.where(ind[0] == n)
                        tmpgrad = np.abs((dep[ik] - depth[n]) / dist[ik])
                        w = np.exp(- np.square(dist[ik] - mr)/np.square(row.distance/4.))
                        if np.sum(w) == 0.0:
                            kyst.hgrad[q] = 0.
                        else:
                            kyst.hgrad[q] = np.average(tmpgrad, weights = w)
                        kyst.h[q] = depth[n]
                n = n + 1
        else: 
            print('Polygon: ' + str(par.poly_number[i]))        
            mr = par.min_res[i]
            depth = scint.griddata((h[:,0], h[:,1]), h[:,2], (kyst_in.x, kyst_in.y))
            depth[depth < min_depth] = min_depth
            x = np.array(kyst_in.x)
            y = np.array(kyst_in.y)
            x = x.reshape(len(x),1)
            y = y.reshape(len(y),1)
            xtopo = h[:,0]
            ytopo = h[:,1]
            dist = np.sqrt(np.square(x - xtopo) + np.square(y - ytopo))
            ind = np.where(dist <= 2000)
            dist = dist[ind]
            dep = h[ind[1],2]
            n = 0
            for q,row in kyst_in.iterrows():
                ik = np.where(ind[0] == n)
                tmpgrad = np.abs((dep[ik] - depth[n]) / dist[ik])
                w = np.exp(- np.square(dist[ik] - mr)/np.square(row.distance/4.))
                if np.sum(w) == 0.0:
                    kyst.hgrad[q] = 0.
                else:
                    kyst.hgrad[q] = np.average(tmpgrad, weights = w)
                kyst.h[q] = depth[n]
                n = n + 1

    return kyst

def sigma_tanh(nlev, dl, du):
    '''Generate a tanh sigma coordinate distribution'''

    dist = np.zeros(nlev)

    for k in np.arange(nlev-1):
        x1 = dl + du
        x1 = x1 * (nlev - 2 - k) / (nlev - 1)
        x1 = x1 - dl
        x1 = np.tanh(x1)
        x2 = np.tanh(dl)
        x3 = x2 + np.tanh(du)
        dist[k+1] = (x1 + x2) / x3 - 1.0

    return dist

def resolution(kyst, obcres, force_points_across = 0, f2f = False, topores = False, sigma = 0, rx1max = 0, min_depth = 0):
    '''Calculate coastal resolution'''

    kyst.points_across[kyst.obc == 1] = 1.
    kyst.distres = kyst.distance / kyst.points_across
    kyst.distres[kyst.obc == 1] = obcres
    if f2f:
        kyst.mobility[kyst.obc == 1] = 0.0
        resobc = np.zeros(len(kyst.x))
        for n in np.arange(len(kyst.x)):
            if n == 0:
                resobc[n] = np.sqrt(np.square(kyst.x[n+1] - kyst.x[n]) + np.square(kyst.y[n+1] - kyst.y[n]))
            elif n == len(kyst.x)-1:
                resobc[n] = np.sqrt(np.square(kyst.x[n] - kyst.x[n-1]) + np.square(kyst.y[n] - kyst.y[n-1]))
            else:
                ds1 = np.sqrt(np.square(kyst.x[n+1] - kyst.x[n]) + np.square(kyst.y[n+1] - kyst.y[n]))
                ds2 = np.sqrt(np.square(kyst.x[n] - kyst.x[n-1]) + np.square(kyst.y[n] - kyst.y[n-1]))
                resobc[n] = np.min([ds1, ds2])
    if topores:
        sps = sigma[1:len(sigma)] + sigma[0:len(sigma)-1]
        sms = sigma[1:len(sigma)] - sigma[0:len(sigma)-1]
        R = max(np.abs(sps/sms))
        kyst.hgrad[kyst.hgrad < 1.0e-20] = 1.0e-20
        kyst.topores = 2 * kyst.h * rx1max / ((R - rx1max) * kyst.hgrad)
        kyst.topores[kyst.obc == 1] = obcres
    
        ind = kyst.distres > kyst.topores
        kyst.resolution = kyst.distres
        kyst.resolution[ind] = kyst.topores[ind]
        indlarger = kyst.resolution > kyst.max_res
        kyst.resolution[indlarger] = kyst.max_res[indlarger]
        indsmaller = kyst.resolution < kyst.min_res
        kyst.resolution[indsmaller] = kyst.min_res[indsmaller]
        kyst.resolution[kyst.obc == 1] = obcres
        kyst.rx1 = R * kyst.hgrad * kyst.resolution / (2 * kyst.h + kyst.hgrad * kyst.resolution)
    else:
        kyst.resolution = kyst.distres
        indlarger = kyst.resolution > kyst.max_res
        kyst.resolution[indlarger] = kyst.max_res[indlarger]
        indsmaller = kyst.resolution < kyst.min_res
        kyst.resolution[indsmaller] = kyst.min_res[indsmaller]
        kyst.resolution[kyst.obc == 1] = obcres
    if f2f:
        kyst.resolution[kyst.mobility == 0.0] = resobc[kyst.mobility == 0.0]
    if force_points_across > 0:
        kyst.resolutions[kyst.distance / kyst.resolution > force_points_across] = kyst.distance / force_points_across

    return kyst
    
def write_to_file(kyst, path=os.getcwd()):
    '''Write to file boundary.txt and islands.txt'''
    
    # Write boundary
    boundary = kyst[kyst.boundary == 1]
    fid = open(os.path.join(path,'boundary.txt'), 'w')
    fid.write(str(len(boundary)) + '\n')
    for index, row in boundary.iterrows():
        line = str(row.x) + ' ' + str(row.y) + ' ' + str(row.resolution) + ' ' + str(row.mobility) + '\n'
        fid.write(line)
    fid.close()
    
    # Write islands
    islands    = kyst[kyst.boundary == 0]
    island_num = np.unique(islands.island_number) 
   
    fid = open(os.path.join(path,'islands.txt'), 'w')
    if len(island_num)>0:
        for i in np.nditer(island_num):
            island_i = islands[islands.island_number == i]
            fid.write(str(len(island_i)) + '\n')
            
            for index, row in island_i.iterrows():
                line = str(row.x) + ' ' + str(row.y) + ' ' + str(row.resolution) + '\n'
                fid.write(line)
    else:
        fid.write('0\n')

    fid.close()

def write_polygon(x, y, polygon_number, filename='Polygon.txt', path=os.getcwd()):
    '''Write polygon to file'''

    #Header
    fid = open(os.path.join(path,filename), 'w')
    line = 'COAST' + '\n'
    fid.write(line)
    nump = int(np.max(polygon_number)) #number of polygons
    line = str(nump) + '\n'
    fid.write(line)

    #Write polygons
    p = 1
    while True:
        xp = x[polygon_number==p]
        yp = y[polygon_number==p]
        plen = len(xp)
        if plen > 0:
            line = str(plen) + ' ' + '1' + '\n'
            fid.write(line)
            for n in np.arange(plen):
                line = str(xp[n]) + ' ' + str(yp[n]) + '\n'
                fid.write(line)
        if p == nump:
            break
        p = p + 1
    fid.close()

def write_bound_isl(x, y, polygon_number, path=os.getcwd()):
    '''Write polygon to file'''

    #Header
    fid = open(path, 'w')
    nump = int(np.max(polygon_number)) #number of polygons
    line = str(nump) + '\n'

    #Write polygons
    p = 1
    while True:
        if nump>0:
            xp   = x[polygon_number==p]
            yp   = y[polygon_number==p]
            plen = len(xp)
        else:
            plen = 0
            p    = 0

        if plen > 0:
            line = str(plen) + ' ' +'\n'
            fid.write(line)
            for n in np.arange(plen):
                line = str(xp[n]) + ' ' + str(yp[n]) + '\n'
                fid.write(line)
        if p == nump:
            break
        p = p + 1
    fid.close()
    
def old_grabPolylines(mapfile = 'PolyLines.map', npol = 9999):
    '''Creates closed polygons from sms-map-file containing polygon lines''' 
    #Read and plot arcs from mapfile
    xpl, ypl, pnl = read_map(mapfile)
    not_used      = np.unique(pnl)
    mp            = int(np.max(pnl))
    xpoly         = np.empty(0)
    ypoly         = np.empty(0)
    polynum       = np.empty(0, int)
    pcount        = 1
    dummy         = True
    for n in range(1, mp+1):
        plt.plot(xpl[pnl == n], ypl[pnl == n], 'k')

    while True:
#        for n in range(1, mp+1):
#            plt.plot(xpl[pnl == n], ypl[pnl == n], 'k')
        ax = plt.axes()
        if dummy:
            ax.set_aspect('equal')
            dummy = False
        p0 = plt.ginput(n = -1, timeout = 0)

        mkpoly = False
        polytest = False

        #Select lines with ginput and plot selected lines in red
        ind = np.empty(0, int)
        for n in range(len(p0)):
            ds2 = np.square((xpl - p0[n][0])) + np.square((ypl - p0[n][1]))
            id = np.where(ds2==np.min(ds2))
            if len(id[0]) > 1:
                id = int(id[0][0])
            else:
                id = int(id[0])
            ind = np.append(ind, int(pnl[id]))
        #for n in range(len(ind)):
        #    plt.plot(xpl[pnl == ind[n]], ypl[pnl == ind[n]], 'r')
        used = ind

        #Convert the selected lines into a polygon
        try:
            xx = xpl[pnl == ind[0]]
            yy = ypl[pnl == ind[0]]
            x1 = np.empty(0)
            y1 = np.empty(0)
            x2 = np.empty(0)
            y2 = np.empty(0)
            ind = ind[1:,]
            for m in ind:
                x0 = xpl[pnl == m]
                y0 = ypl[pnl == m]
                x1 = np.append(x1, x0[0])
                x2 = np.append(x2, x0[len(x0)-1])
                y1 = np.append(y1, y0[0])
                y2 = np.append(y2, y0[len(x0)-1])
            while True:
                i1 = ind[np.where((x1==xx[len(xx)-1]) & (y1==yy[len(yy)-1]))]
                i2 = ind[np.where((x2==xx[len(xx)-1]) & (y2==yy[len(yy)-1]))]
        
                if len(i1) == 1:
                    xx = np.append(xx, xpl[pnl==i1])
                    yy = np.append(yy, ypl[pnl==i1])
                    x1 = x1[ind!=i1]
                    y1 = y1[ind!=i1]
                    x2 = x2[ind!=i1]
                    y2 = y2[ind!=i1]
                    ind = ind[ind!=i1]
                else:
                    xx = np.append(xx, np.flipud(xpl[pnl==i2]))
                    yy = np.append(yy, np.flipud(ypl[pnl==i2]))
                    x2 = x2[ind!=i2]
                    y2 = y2[ind!=i2]
                    x1 = x1[ind!=i2]
                    y1 = y1[ind!=i2]
                    ind = ind[ind!=i2]
                if len(ind) == 0:
                   break
        except:
            print('ERROR - probably a missing line - try again')
            for n in range(len(used)):
                plt.plot(xpl[pnl == used[n]], ypl[pnl == used[n]], 'r')
        else:
            mkpoly = True
            if (xx[0]==xx[len(xx)-1]) & (yy[0]==yy[len(yy)-1]):
                polytest = True
            if polytest & mkpoly:
                for k in range(len(used)):
                    not_used = not_used[not_used != used[k]]
                poly = np.empty([len(xx), 2])
                poly[:,0] = xx
                poly[:,1] = yy
                polygons = []
                polygons.append(mPolygon(poly))
                p = PatchCollection(polygons, alpha=0.4)
                ax.add_collection(p)
                xpoly = np.append(xpoly, xx)
                ypoly = np.append(ypoly, yy)
                polynum = np.append(polynum, pcount * np.ones(xx.shape))
                plt.text(np.mean(xx), np.mean(yy), str(pcount-1))
                pcount = pcount + 1
                for n in range(1, mp+1):
                    plt.plot(xpl[pnl == n], ypl[pnl == n], 'k')
                if npol == 9999:
                    terminate = (len(not_used) == 0)
                else:
                    terminate = np.max(polynum) == npol
        if terminate:
                break

    return xpoly, ypoly, polynum
                
def find_polygons(mapfile='Polylines.map'):
    xpl, ypl, pnl = read_map(mapfile)

    # Storing the lines as shapely objects
    for i in range(pnl.astype('int').max()):
        line      = np.empty([len(pnl[pnl==i+1]),2])
        line[:,0] = xpl[pnl==i+1]
        line[:,1] = ypl[pnl==i+1]
        if i == 0:
            lines = LineString(line)
        elif i == 1:
            lines = lines, LineString(line)
            lines = list(lines)
        else:
            lines.append(LineString(line))
            
    # Identifying the individual polygons, storing some important stuff in dicts
    areas      = polygonize(lines)
    return areas

def grabPolylines(mapfile = 'Polylines.map', return_polygons = False):
    '''
    Identifies the sub-polygons in the domain
    '''
    areas      = find_polygons(mapfile=mapfile)
    xpl, ypl, pnl = read_map(mapfile)
    output     = dict(type='FeatureCollection', features = [])
    centroids  = []
    xpoly      = np.empty(0)
    ypoly      = np.empty(0)
    polynum    = np.empty(0, int)
    
    # Get centroids
    # ----
    for (index,area) in enumerate(areas):
        feature   = dict(type='Feature', properties = dict(index=index))
        feature['geometry'] = area.__geo_interface__
        output['features'].append(feature)
        centroids.append(area.centroid)

    # get polygons
    # ----
    polygons = []
    for n in range(output['features'][-1]['properties']['index']+1):
        coordinates = np.array(output['features'][n]['geometry']['coordinates'])
        poly        = np.empty([len(coordinates[0,:,0]),2])
        poly[:,0]   = coordinates[0,:,0]
        poly[:,1]   = coordinates[0,:,1]
        polygons.append(mPolygon(poly))
        xpoly   = np.append(xpoly, coordinates[0,:,0])
        ypoly   = np.append(ypoly, coordinates[0,:,1])
        polynum = np.append(polynum, output['features'][n]['properties']['index']*np.ones(len(poly[:,0]))+1)

    plotPolylines(xpoly, ypoly, polynum, centroids, polygons)

    if return_polygons:
        return xpoly, ypoly, polynum, centroids, polygons
    else:
        return xpoly, ypoly, polynum

def plotPolylines(xpoly, ypoly, polynum, centroids, polygons):
    '''
    Plots polygons and labels them accordingly
    '''
    fig, ax = plt.subplots()
    plt.axis('equal')
    for n in range(polynum.astype('int').max()):
        ax.plot(xpoly[polynum == n+1], ypoly[polynum == n+1], 'k')

    p = PatchCollection(polygons, alpha=0.4)
    ax.add_collection(p)
    for n in range(len(polygons)):
        plt.text(centroids[n].x, centroids[n].y, f'{n}')

    plt.show(block=False)