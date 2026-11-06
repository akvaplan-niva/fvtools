"""Legacy FVCOM_grid aliases and the NetCDF compatibility mixin.

Class implementations extracted verbatim from fvcom_grd.py.
"""

import numpy as np
from fvtools.grid.fvcom.netcdf_io import FVCOMNetCDFIO


class LegacyPropertyAliases:
    '''
    Aliases for properties with mutliple names, used when writing forcing/initial files (turns out these names are actually used by FVCOM, won't get around them)
    '''
    @property
    def siglev_center(self):
        return self.siglev_c

    @property
    def siglay_center(self):
        return self.siglay_c


class LegacyFunctions:
    def cell2node(self, fieldin):
        """Average cell values at connected mesh nodes (equal cell weights).

        Vectorized via bincount. Nodes without any attached cell return NaN.
        """
        values = np.asarray(fieldin, dtype=float)
        if values.ndim != 1 or values.size != self.cell_number:
            raise ValueError('fieldin must have one value per cell')
        nodes = self.tri.ravel()
        total = np.bincount(nodes, weights=np.repeat(values, 3),
                            minlength=self.node_number)
        count = np.bincount(nodes, minlength=self.node_number)
        return np.divide(total, count, out=np.full(self.node_number, np.nan),
                         where=count > 0)


class OutputLoader(FVCOMNetCDFIO):
    """FVCOM_grid NetCDF compatibility layer. See netcdf_io.FVCOMNetCDFIO."""
