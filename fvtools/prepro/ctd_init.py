#       Here, we make use of CTDs to initialize any given domain.
# --------------------------------------------------------------------------
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from fvcom_grd import FVCOM_grid
from matplotlib.path import Path
from netCDF4 import Dataset
from scipy.io import loadmat

# Use a CTD profile to initialize parts of a FVCOM domain
# --------------------------------------------------------------------------
def main(ctd,restartfile):
    '''
    reads the CTD profile, uses it to change the initial state of the
    chosen domain.
    '''
    profile = CTD(ctd)             # CTD profile from field
    rest    = Restart(restartfile) # Load restartfile, find indices to change
    update_field(profile,rest)     # Interpolate CTD to model

class CTD():
    'CTD profile from AdFontes (.mat)'
    def __init__(self, pathToFile):
        self.filepath = pathToFile
        self.add_parameters(['T','S','P'])
        self.standard_format()

    def add_parameters(self, names):
        ctdfile = loadmat(self.filepath)
        for name in names:
            setattr(self, name, ctdfile['data'][0,0][name])

    def standard_format(self):
        '''
        Arange the data such that z=0 is at the front
        end of the vector, and z=max at the rear end.
        Remove duplicates and return
        '''
        P,ind  = np.unique(self.P[:,0],return_index=True)
        self.P = P
        S      = self.S[ind]
        T      = self.T[ind]
        self.S = S
        self.T = T

class Restart():
    'Restartfile from FVCOM. Has to already be interpolated from mothergrid'
    def __init__(self, pathToFile):
        self.filepath = pathToFile
        self.read_ncdata()
        self.change_these_nodes()

    def read_ncdata(self):
        nc         = Dataset(self.filepath)
        self.T     = nc['temp'][:]
        self.S     = nc['salinity'][:]
        self.x     = nc['x'][:]
        self.y     = nc['y'][:]
        self.nv    = nc['nv'][:].transpose()-1
        self.dpt   = -nc['h'][:]*nc['siglay'][:]
        nc.close()

    def change_these_nodes(self):
        '''
        The user draws a polygon surrounding the region we are re-initializing
        '''
        # Create the bounding region
        plt.triplot(self.x,self.y,self.nv,lw=0.2,c='g')
        plt.axis('equal')
        plt.title('Click and create a polygon. Hit Enter when the region is bounded')
        pts     = plt.ginput(n=-1,timeout=-1)
        x,y     = zip(*pts)
        x       = np.array(x); y = np.array(y)
        x       = np.append(x,x[0]); y = np.append(y,y[0])
        sti     = np.column_stack((x,y))
        p       = Path(sti)

        # Check which FVCOM nodes are within that region
        queries       = np.column_stack((self.x,self.y))
        self.ind_bool = p.contains_points(queries)

        # Show which indices were chosen
        plt.scatter(self.x[self.ind_bool],self.y[self.ind_bool],s=10,c='r')
        plt.show()
        self.ind      = [indekser for indekser, xtmp in enumerate(self.ind_bool) if xtmp]

def update_field(ctd,grid):
    '''
    Update the initial field and write to the restartfile
    '''

    nc = Dataset(grid.filepath,'r+')

    for i in grid.ind:
        T,S                   = vinterp(ctd,grid.T[0,:,i],grid.S[0,:,i],grid.dpt[:,i])
        nc['temp'][0,:,i]     = T
        nc['salinity'][0,:,i] = S

    nc.close()

def vinterp(ctd,T,S,dpt):
    '''
    For any given node, interpolate and return new initial profiles.
    Ideally, this script is used once every timestep.
    -- not tested (yet).
    '''
    Tnew = []; Snew = []
    for i in range(len(dpt)):
        for j in range(len(ctd.P)-1):
            # 1 Find the closest index
            if dpt[i]<ctd.P[0]:
                int1 = 0; int2 = 0
                break

            elif dpt[i]==ctd.P[j]:
                int1 = j; int2 = j
                break

            elif ctd.P[j]>dpt[-1]:
                int1 = len(ctd.P)-1; int2 = len(ctd.P)-1
                break

            elif ctd.P[j+1]>dpt[i] and ctd.P[j]<dpt[i]:
                int1 = j; int2 = j+1
                break

        interp_coef_1 = 1.0/(0.1+abs(dpt[i]-ctd.P[int1]))
        interp_coef_2 = 1.0/(0.1+abs(dpt[i]-ctd.P[int2]))
        sum_coef      = interp_coef_1+interp_coef_2

        Tnew.append(ctd.T[int1]*interp_coef_1/sum_coef + ctd.T[int2]*interp_coef_2/sum_coef)
        Snew.append(ctd.S[int1]*interp_coef_1/sum_coef + ctd.S[int2]*interp_coef_2/sum_coef)
    return Tnew, Snew
