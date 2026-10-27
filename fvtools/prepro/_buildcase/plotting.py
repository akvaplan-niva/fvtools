"""Original BuildCase bathymetry diagnostics; preserves Jupyter widget plots."""
import numpy as np
import matplotlib.pyplot as plt


class BathymetryPlotter:
    def show_depth(self):
        """
        Plot the raw, smoothed and difference between the two of them

        Parameters:
        ----
        A dict (M) with h_raw, h and grid info in it.

        Out:
        ----
        Returns a plot of the raw bathymetry
        """
        increment = int(max(self.h_raw)/100)*10
        if increment == 0:
            return

        levels  = np.arange(0, max(self.h_raw), increment)
        
        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15,8))
        cdat = ax[0].tricontourf(self.x, self.y, self.tri, self.h_raw, levels = levels, cmap = 'terrain')
        ax[0].tricontour(self.x, self.y, self.tri, self.h_raw, levels = levels, colors = 'k')
        ax[0].axis('equal')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title('raw bathymetry')
        plt.colorbar(cdat, ax = ax[0])

        cdat = ax[1].tricontourf(self.x, self.y, self.tri, self.h, levels = levels, cmap = 'terrain')
        ax[1].tricontour(self.x, self.y, self.tri, self.h, levels = levels, colors = 'k')
        ax[1].axis('equal')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title('smoothed bathymetry')
        plt.colorbar(cdat, ax = ax[1])

        cdat = ax[2].tricontourf(self.x, self.y, self.tri, self.h_raw-self.h, 100, cmap = 'terrain')
        ax[2].axis('equal')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_title('raw-smoothed bathymetry')
        plt.colorbar(cdat, ax = ax[2])
