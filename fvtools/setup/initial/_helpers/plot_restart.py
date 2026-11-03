import matplotlib.pyplot as plt
import netCDF4
import cmocean as cmo
import numpy as np

def plot_interps(mother_file, child_file, M_mother, M_child, ind):
    '''
    Visualize results from the interpolation in an attempt at building confidence in the quality of the interpolation
    '''
    def make_comparison_figure(M_mother, mother_field, M_child, child_field, levels, cmap, title):
        fig, ax = plt.subplots(1,2,figsize = (20,10))
        tp = ax[0].tricontourf(M_child.x, M_child.y, M_child.tri, child_field, levels = levels, extend = 'both')
        ax[0].set_title(f'child {title}')
        ax[0].set_aspect('equal')

        tp = ax[1].tricontourf(M_mother.x, M_mother.y, M_mother.tri, mother_field, levels = levels, extend = 'both')
        ax[1].set_title(f'mother {title}')
        ax[1].set_xlim(M_child.x.min(), M_child.x.max())
        ax[1].set_ylim(M_child.y.min(), M_child.y.max())
        ax[1].set_aspect('equal')

        fig.subplots_adjust(right = 0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.675])
        cb = fig.colorbar(tp, cax = cbar_ax)
        cb.set_label(f'mother {title}')

    with netCDF4.Dataset(mother_file, 'r') as mother:
        with netCDF4.Dataset(child_file, 'r') as child:
            levels  = np.linspace(child.variables['zeta'][0, :].min(), child.variables['zeta'][0, :].max(), 30)
            make_comparison_figure(M_mother, mother['zeta'][ind, M_mother.cropped_nodes], 
                                   M_child, child['zeta'][0,:], 
                                   levels, cmo.cm.amp, 'sea surface elevation [m]')

            levels  = np.linspace(child.variables['temp'][0, -1, :].min(), child.variables['temp'][0, 0, :].max(), 30)
            make_comparison_figure(M_mother, mother['temp'][ind, 10, M_mother.cropped_nodes], 
                                   M_child, child['temp'][0, 10, :], 
                                   levels, cmo.cm.thermal, 'sigma = 10 temperature [C]')

            levels  = np.linspace(29, 35, 30)
            make_comparison_figure(M_mother, mother['salinity'][ind, 0, M_mother.cropped_nodes], 
                                   M_child, child['salinity'][0, 0, :], 
                                   levels, cmo.cm.haline, 'surface salinity [psu]]')
    plt.show(block = False)
