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

    def plot_cvs(self, field, cmap='jet', cmax=None, cmin=None, Norm=None,
                 edgecolor='face', inds=None, **kwargs):
        """Plot a nodal field on control-volume patches.

        ``inds`` selects node indices (or a boolean mask). If ``cmin`` and/or
        ``cmax`` are given, thresholding is applied *within* that selection.
        Patches and plotted values use exactly the same node order.

        Additional keyword arguments are forwarded to ``PatchCollection``.
        """
        field = np.asanyarray(field)
        if field.ndim != 1 or len(field) != self.node_number:
            raise ValueError('field must be one-dimensional with one value per node')

        # Preserve the old fast path for repeated full-domain frames.
        full_domain = inds is None and cmin is None and cmax is None
        if inds is None:
            inds = np.arange(self.node_number, dtype=np.intp)
        else:
            inds = np.asarray(inds)
            if inds.ndim != 1:
                raise ValueError('inds must be a one-dimensional index array or mask')
            if np.issubdtype(inds.dtype, np.bool_):
                if len(inds) != self.node_number:
                    raise ValueError('boolean inds must have one entry per node')
                inds = np.flatnonzero(inds)
            elif not np.issubdtype(inds.dtype, np.integer):
                raise TypeError('inds must contain integer node indices')
            if np.any(inds < 0) or np.any(inds >= self.node_number):
                raise IndexError('inds contains node indices outside the grid')

        if cmax is not None:
            inds = inds[field[inds] <= cmax]
        if cmin is not None:
            inds = inds[field[inds] >= cmin]

        # self.cv is a Python list. NumPy advanced indexing does not work on
        # it, and comparing lengths is insufficient if inds permutes nodes.
        patches = self.cv if full_domain else [self.cv[int(i)] for i in inds]
        collection = PatchCollection(
            patches, cmap=cmap, edgecolor=edgecolor, norm=Norm, **kwargs
        )
        collection.set_array(field[inds])
        ax = plt.gca()
        artist = ax.add_collection(collection)
        ax.autoscale_view(True)
        ax.set_aspect('equal')
        if artist._A is not None:
            ax._sci(artist)
        return artist

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

        # The original condition was: boundary and
        # len(np.unique(np.array(xcv))) < len(xcv).
        # Converting a growing Numba list to an ndarray here can fail during
        # LLVM lowering on some Numba/llvmlite versions. Compare directly.
        if boundary:
            duplicate_x = False
            for j in range(len(xcv)):
                for k in range(j):
                    left = xcv[j]
                    right = xcv[k]
                    if left == right or (np.isnan(left) and np.isnan(right)):
                        duplicate_x = True
                        break
                if duplicate_x:
                    break
            if duplicate_x:
                break
    return xcv, ycv
