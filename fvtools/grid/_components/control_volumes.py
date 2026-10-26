"""Plot FVCOM nodal control volumes and build their polygon boundaries.

This contains the original Numba-accelerated module-level helpers as well
as the ControlVolumePlotter mixin. No calculation logic is changed.
"""

import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as mPolygon
from numba import njit


class ControlVolumePlotter:
    @property
    def cv(self):
        '''
        A list of patch collections for the node based control volumes
        '''
        if not hasattr(self, '_cv'):
            self._cv = self._get_cvs()
        return self._cv

    def plot_cvs(self, 
                 field,
                 cmap = 'jet', 
                 cmax = None, 
                 cmin = None, 
                 Norm = None, 
                 edgecolor = 'face', 
                 inds = None,
                 **kwargs):
        '''
        Plots nodal fields on control volume patches

        Optional:
        ----
        - edgecolor: None by default. Set a color, e.g. "k" to draw interfaces between CVs
        - cmap:      colormap
        - cmax:      cap colors to cmax
        - cmin:      floor colors to cmin
        - Norm:      to create non-linear colorbars (ref matplotlib documentation)
        --> for cmax and cmin, the routine exclusively plots CVs with values within in that range

        keyword arguments are sent to matplotlib.collections.PatchCollection
        '''
        if inds == None:
            inds = np.arange(0, len(field), dtype = int)

        # mask cvs with values below/above threshold
        if cmax is not None and cmin is None:
            inds = np.where(field<=cmax)[0]
        elif cmin is not None and cmax is None: 
            inds = np.where(field>=cmin)[0]
        elif cmax is not None and cmin is not None:
            inds = np.intersect1d(np.where(field<=cmax)[0], np.where(field>=cmin)[0])

        if len(inds) == len(self.cv): 
            collection = PatchCollection(self.cv, cmap=cmap, edgecolor=edgecolor, norm=Norm, **kwargs)
        else: 
            collection = PatchCollection(self.cv[inds], cmap=cmap, edgecolor=edgecolor, norm=Norm, **kwargs)
        collection.set_array(field[inds])
        ax   = plt.gca()
        _ret = ax.add_collection(collection)
        ax.autoscale_view(True)
        ax.set_aspect('equal')
        if _ret._A is not None: 
            plt.gca()._sci(_ret)
        return _ret

    def _get_cvs(self, nodes = None):
        '''
        - computes matplotlib-patches for each control voulme, adds to self as self.cvs
        - Returns numpy file (cvs.npy) containing all of the control volume patches
        '''
        if nodes is None: 
            nodes = np.arange(0, self.node_number)
        xcv_full, ycv_full = self._get_control_volumes(self.nbsn, self.nbve, self.x, self.y, self.xc, self.yc, nodes)
        bar    = pb.ProgressBar(widgets = ['Making control volume patches: ', pb.Percentage(), ' ', pb.BouncingBar(), pb.AdaptiveETA()], maxval=len(nodes))
        bar.start()
        cv = []
        for i, (xcv, ycv) in enumerate(zip(xcv_full, ycv_full)):
            bar.update(i)
            cv.append(mPolygon(np.array([xcv, ycv]).transpose(), closed = True))
        bar.finish()
        return cv
        #np.save('cvs.npy', self.cv) # for instant access to cvs at a later date

    @staticmethod
    def _get_control_volumes(nbsn, nbve, x, y, xc, yc, nodes):
        '''returns list of control volumes'''
        xcv_full, ycv_full = [], []
        for i, node in enumerate(nodes):
            indexes   = np.where(nbsn[node,:]>=0)[0]
            nbsnhere  = nbsn[node, indexes]
            elemshere = nbve[node, indexes]

            xcv, ycv = _find_cv_walls(node, nbsnhere, elemshere, x, y, xc, yc)
            xcv_full.append(xcv)
            ycv_full.append(ycv)
        return xcv_full, ycv_full

@njit
def _find_cv_walls(node, nbsnhere, elemshere, x, y, xc, yc):
    '''find points that should connect to make a cv wall'''
    xmid  = (x[node] * np.ones((len(nbsnhere),)) + x[nbsnhere])/2
    ymid  = (y[node] * np.ones((len(nbsnhere),)) + y[nbsnhere])/2
    xcell, ycell = xc[elemshere], yc[elemshere]
    return _connect_cv_walls(xmid, ymid, xcell, ycell, x, y, nbsnhere, elemshere, node)

@njit
def _connect_cv_walls(xmid, ymid, xcell, ycell, x, y, nbsnhere, elemshere, node):
    xcv = []; ycv = []
    for i in range(len(xmid)):
        xcv.append(xmid[i])
        ycv.append(ymid[i])

        # Find direction
        boundary = False
        if i < len(xmid)-1:
            xcvmid = (xmid[i]+xmid[i+1])/2
            ycvmid = (ymid[i]+ymid[i+1])/2
            dst    = np.sqrt((xcell-xcvmid)**2+(ycell-ycvmid)**2)
            ind    = np.argwhere(dst==dst.min())[0]

            # Be aware of land
            if nbsnhere[i+1] == node:
                boundary = True
                xcv.append(x[nbsnhere[i+1]])
                ycv.append(y[nbsnhere[i+1]])
            elif nbsnhere[i] == node:
                boundary = True
        if not boundary:
            xcv.append(xcell[ind][0])
            ycv.append(ycell[ind][0])

        uxcv    = np.array(xcv)
        unique  = np.unique(uxcv)
        nunique = len(unique)
        ncv     = len(xcv)
        nunique_lt_ncv = nunique < ncv
        if boundary and nunique_lt_ncv:
            break
    return xcv, ycv
