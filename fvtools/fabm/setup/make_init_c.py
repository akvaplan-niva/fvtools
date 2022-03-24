import sys
import os
user = os.getlogin()
appendpath = '/cluster/work/' + user + '/fvtools/grid/'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from netCDF4 import Dataset
from fvcom_grd import FVCOM_grid

def make_init_c():
    M = FVCOM_grid('M.npy')
    plt.triplot(M.x, M.y, M.tri,color='grey')
    plt.axis('equal')
    plt.title('Zoom in to your region of interest, choose one point and hit enter')
    out     = plt.ginput(n=9999, timeout = -1)
    xp, yp  = zip(*out)

    # Anta at mæra er 50m i diameter
    # -------------------------------------------------------------
    R       = 25
    theta   = np.linspace(0,2*np.pi,100)
    x       = R*np.cos(theta)+xp[0]
    y       = R*np.sin(theta)+yp[0]

    # Finner puktene innenfor stien
    # -------------------------------------------------------------
    sti     = np.column_stack((x,y))
    queries = np.column_stack((M.x, M.y))
    p       = path.Path(sti)
    inp     = p.contains_points(queries)

    # Finner mappen vi jobber i
    str1    = os.getcwd()
    str2    = str1.split('/')
    n       = len(str2)
    folder  = str2[n-1]

    # Antar at restartfilen er nr 1
    restartfile = folder + '_restart_0001.nc'

    # Lager variabelen
    data    = Dataset(restartfile)
    var1    = data['salinity']
    tracer_c = np.zeros(var1.shape)
    tracer_c[0,0:10,inp] = 1

    ## Redigerer restartfilen
    # -----------------------------------------------
    infile  = Dataset(restartfile,'r+')


    # En switch som gjør at koden forstår om det eksisterer en restartfil med
    # variablen eller ei
    try:
        infile['tracer_c'][:] = tracer_c
    except IndexError:
        # Leser dimensjonene i restartfilen
        print('tracer_c does not exist in the file, we must therefore create one')
        #node    = infile.dimensions['node']
        #time    = infile.dimensions['time']
        #siglay  = infile.dimensions['siglay']
        tracer  = infile.createVariable('tracer_c', 'f4', ('time', 'siglay', 'node'))
        tracer.units = 'quantity m-3'

        # Dytter tracer_c inn i netCDF-filen
        tracer[:,:,:] = tracer_c
       
    infile.close()

    print('Fin')
#    plt.figure()
#    plt.axis('equal')
#    plt.triplot(M.x[:,0],M.y[:,0],M.tri,lw=0.2,color='grey')
    plt.scatter(M.x[inp],M.y[inp],c='r')
    plt.show()
