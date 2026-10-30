import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from fvcom_grd import FVCOM_grid

from netCDF4 import Dataset
from datetime import date,datetime,timedelta

# ----------------------------------------------------------
# Constants
# ---------
v        = 0.5
dt       = 10 # sekunder
T        = 12 # timer
checkit  = False # See the ship moving


# Definer seileruten
# ------------------------------
M = FVCOM_grid('M.mat')
goodtrack = False
while not goodtrack:
    plt.triplot(M.x[:,0],M.y[:,0],M.tri)
    plt.axis('equal')
    plt.title('Zoom in to your area of interest, and click a track\n Press enter when you are done.')
    pts    = plt.ginput(90)
    x, y   = zip(*pts)
    plt.close()
    
    x = np.array(x)
    y = np.array(y)
    
    y = np.append(y,y[0])
    x = np.append(x,x[0])

    fig = plt.figure()
    plt.axis('equal')
    plt.triplot(M.x[:,0],M.y[:,0],M.tri)
    plt.plot(x, y, color = 'red')
    plt.show()
    goodtrack = input('Are you happy with your track? y (yes) or n (no)\n')
    plt.close()

    if goodtrack=='n':
        goodtrack = False


# Seilasen
# ------------------------------------------------------------------------------------
# 1. Noen tomme variabler:
length        = []
antall        = []
x_insitu      = []
y_insitu      = []
nodeind       = []
Release       = []

# Bruker stiens lengde
length = []
for i in range(len(x)-1):
    length.append(np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2))
length        = np.array(length)
len_end       = np.sqrt((x[-1]-x[0])**2+(y[-1]-y[0])**2)
length        = np.append(length,len_end)

# Total seilt distanse etter en runde
tot_len = np.sum(length)

# Finn nodene som følger stien
n_it            = int(T*60*60/dt) # Antall tilfeller i perioden
len_tot         = 0 # Hvor langt skipet har seilt etter n-tidssteg

ind             = np.arange(0,len(M.x),1)
nodeind         = []
for i in range(n_it): # Finner nærmeste node hvert tidssteg.
    T_actual    = i*dt
    dst_dt      = v*dt
    len_tot     = len_tot+dst_dt
    if len_tot>tot_len:
        len_tot = len_tot-tot_len
    
    # Identifiserer hvor skipet er
    index       = 0
    len_max     = length[0]

    while len_tot > len_max:
        index       = index+1
        if index>(len(x)-1):
            index   = 0
        len_max     = len_max+length[index]
    
    len_in_line     = length[index]-(len_max-len_tot)
    frac_in_line    = len_in_line/length[index]

    if index<len(x)-1:
        x_insitu.append(x[index]+frac_in_line*(x[index+1]-x[index]))
        y_insitu.append(y[index]+frac_in_line*(y[index+1]-y[index]))
    else: # I det skipet begynner på en ny runde
        x_insitu.append(x[index]+frac_in_line*(x[0]-x[index]))
        y_insitu.append(y[index]+frac_in_line*(y[0]-y[index]))
        
    # Finner nærmeste node
    dst             = np.sqrt((M.x-x_insitu[i])**2+(M.y-y_insitu[i])**2)
    tmp             = np.argmin(dst[:,0]) # Dette steget er vanvittig tregt!!!
    nodeind.append(tmp) # Dette er nærmeste node ved tidssteg i
    print('Shiptrack is ' + str(np.floor(i/(n_it)*100)) + '% completed')

print('Shiptrack 100% completed, moving on')

# netCDF fila
# Lager flux-fila
flux = np.zeros((n_it,len(M.x)))
for i in range(n_it):
    flux[i,nodeind[i]] = 1

# Lager tidsrekka
timeFn = []
for i in range(n_it):
    timeFn.append(i*dt/(24*60*60))

timeFn = np.array(timeFn)

FVCOM_reldate = datetime(1858,11,17,0,0,0)
MYDATE        = datetime(2014,8,19,0,0,0) # Hardcoda, kan hentes fra restartfil istedet.
timeFn       += (MYDATE-FVCOM_reldate).days

print('Writing netCDF file')
fout            = Dataset(os.path.join('/cluster/home/hes001/python3/', 'flux.nc'),'w',format='NETCDF3_64BIT_OFFSET')
time            = fout.createDimension('time',None)
node            = fout.createDimension('node',M.x.shape[0])
inj             = fout.createVariable('injection_flux_int','f8',('time','node'))
inj.units       = 'kg s-1 m-2'

times           = fout.createVariable('time','f8',('time',))
times.long_name = 'time'
times.units     = 'days since '+str(FVCOM_reldate)
times.format    = 'modified julian day (MJD)'
times.time_zone = 'UTC'

# Write to netcdf created
times[:]        = timeFn.transpose()[:]
inj[:,:]        = flux[:,:]

fout.close()
print('Fin')

#fig,ax = plt.subplot()
#for i in range(len(nodeind)):
#    ax.triplot(M.x[:,0],M.y[:,0],M.tri)
#    ax.scatter(M.x[nodeind[i],0],M.y[nodeind[i],0],color='red')
#    ax.xlim((308307.96796875005,318438.86071875005))
#    ax.ylim((7193547.707142858, 7206081.867857143))
#    plt.show()
#    time.sleep(0.1)
#    ax.close
