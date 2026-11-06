"""Cell-centred triangulation and cache interface."""

import matplotlib


class CellTriangulation:
    @property
    def ctri(self):
        '''Triangulation using xc, yc as nodes. Land is masked.'''
        if not hasattr(self, '_ctri'): 
            self._ctri = self.cell_tri()
        return self._ctri

    @ctri.setter
    def ctri(self, var):
        self._ctri = var
        # A replacement of the cell connectivity invalidates the cached
        # Matplotlib triangulation (the nodal triangulation is unchanged).
        self.__dict__.pop('mtri_cells', None)

    def cell_tri(self):
        '''Get triangulation connecting cell centres.'''
        import fvtools.util.tools as tools
        if self.verbose: print('- Triangulating using Delaunays method')
        ctri  = matplotlib.tri.Triangulation(self.xc, self.yc)
        masked_ctris = tools.mask_land_tris(self.x, self.y, self.tri, ctri.triangles)
        return masked_ctris
