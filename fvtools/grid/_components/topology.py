"""FVCOM mesh connectivity and TGE-derived properties."""

import fvtools.grid.tge as tge
import numpy as np


class PropertiesFromTGE:
    '''
    These grid connectivity indentifiers are handy when dealing with the mesh using the same methods as FVCOM does.
    '''
    @property
    def T(self, file = None):
        '''
        T computes the same grid connectivity metrics as tge.F
        Does also include routines to compute gradients, vorticity and okubo weiss parameter.
        '''
        if not hasattr(self, '_T'):
            print('- Computing triangle, grid, elements (TGE) - this may take some minutes for large grids')
            self._T = tge.main(self, verbose = False)
            self._nbsn = self._T.NBSN
            self._ntsn = self._T.NTSN
            self._nbve = self._T.NBVE
            self._ntve = self._T.NTVE
            self._nbe = self._T.NBE
            self._ISBCE = self._T.ISBCE
            self._ISONB = self._T.ISONB
            self._nbse, self._nese = tge.get_NBSE_NESE(self.nbve, self.ntve, self.tri, len(self.xc))
        return self._T

    @property
    def ISBCE(self):
        '''
        "is boundary control element"
        = 0 in the interior computational domain
        = 1 on the solid boundary
        = 2 on the open boundary
        = 3 element with 2 solid boundary edges
        '''
        if not hasattr(self, '_ISBCE'):
            # Calculate it yourself otherwise
            _T = tge.calculate_nbe_and_boundaries(self, verbose = False)
            self._ISBCE = _T.ISBCE
        return self._ISBCE

    @ISBCE.setter
    def ISBCE(self, var):
        self._ISBCE = var

    @property
    def ISONB(self):
        '''
        "is on boundary"
        = 0 in the interior computational domain
        = 1 on the solid boundary
        = 2 on the open boundary
        '''
        if not hasattr(self, '_ISONB'):
            # Calculate it yourself otherwise
            _T = tge.calculate_nbe_and_boundaries(self, verbose = False)
            self._ISONB = _T.ISONB
        return self._ISONB

    @ISONB.setter
    def ISONB(self, var):
        self._ISONB = var

    @property
    def nbe(self):
        '''indices of elements sharing a wall with *this* element'''
        if not hasattr(self, '_nbe'):
            # Calculate it yourself otherwise
            _T = tge.calculate_nbe_and_boundaries(self, verbose = False)
            self._nbe = _T.NBE
        return self._nbe

    @nbe.setter
    def nbe(self, var):
        self._nbe = var

    @property
    def nbsn(self):
        '''indices of nodes surrounding each node'''
        if not hasattr(self, '_nbsn'): 
            self._nbsn = self.T.NBSN
        return self._nbsn

    @nbsn.setter
    def nbsn(self, val):
        self._nbsn = val

    @property
    def ntsn(self):
        '''number of nodes surrounding each node'''
        if not hasattr(self, '_ntsn'): 
            self._ntsn = self.T.NTSN
        return self._ntsn

    @ntsn.setter
    def ntsn(self, val):
        self._ntsn = val

    @property
    def nbve(self):
        '''indices of elements surrounding each node'''
        if not hasattr(self, '_nbve'): 
            self._nbve = self.T.NBVE
        return self._nbve

    @nbve.setter
    def nbve(self, val):
        self._nbve = val

    @property
    def nese(self):
        '''number of elements surrounding each element'''
        if not hasattr(self, '_nese'): 
            self._nbse, self._nese = tge.get_NBSE_NESE(self.nbve, self.ntve, self.tri, len(self.xc))
        return self._nese

    @property
    def nbse(self):
        '''indices of elements surrounding each element'''
        if not hasattr(self, '_nbse'): 
            self._nbse, self._nese = tge.get_NBSE_NESE(self.nbve, self.ntve, self.tri, len(self.xc))
        return self._nbse

    @property
    def ntve(self):
        '''the number of elements connected to this node'''
        if not hasattr(self, '_ntve'): 
            self._ntve = self.T.NTVE
        return self._ntve

    @ntve.setter
    def ntve(self, var):
        self._ntve = var

    @property
    def art1(self):
        '''the node centered control area'''
        if not hasattr(self, '_art1'):
            self.T.get_art1()
            self._art1 = self.T.art1
        return self._art1

    @art1.setter
    def art1(self, val):
        self._art1 = val
