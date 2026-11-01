"""Original boundary-triangle validation extracted from BuildCase.__init__."""
import numpy as np
import matplotlib.pyplot as plt


def validate_boundary_triangles(self):
    """Apply the original boundary-triangle check and return whether it changed the grid."""
    print('- Check boundary triangles')
    cells = self._return_valid_triangles(self.x, self.y, self.tri)
    if len(cells) != len(self.xc):
        print(f'  - There are {len(self.xc)-len(cells)} illegal nodes in this mesh, removing those')
        _, ax = plt.subplots(1, 1)
        self.plot_grid(ax = ax, label='input grid', show = False)

        # Find the ones we remove
        new_points = np.array([
            (np.mean(self.x[self.tri[cells]], axis = 1)),
            (np.mean(self.y[self.tri[cells]], axis = 1))
        ]).T
        old_points = np.array([self.xc, self.yc]).T

        # Find removed points
        set_old = set(map(tuple, old_points))
        set_new = set(map(tuple, new_points))
        points_removed = np.array(list(set_old - set_new))
        

        # Subgrid (returns a new FVCOM_grid instance, we update necessary fields)
        # - a BuildCase class is not an FVCOM_grid, so the subgridding enherits sone stuff that we don't want
        # - to get the nodestrings, we must delete the _full_model_boundary
        new = self.subgrid(cells = cells)
        self.x, self.y, self.lon, self.lat, self.tri, self.obc_nodes = new.x, new.y, new.lon, new.lat, new.tri, new.obc_nodes
        for attr in ['full_model_boundary', 'nodestrings', 'coastline_nodes']:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except AttributeError:
                    print(f"Attribute {attr} could not be deleted.")

        # Plot the corrected grid
        self.plot_grid(ax = ax, c='r-', label='fixed grid', show = False)

        # Plot removed
        ax.scatter(points_removed[:,0], points_removed[:,1], label = 'removed cells')

        # Finish
        self.plot_obc(ax = ax)
        plt.legend()
        updated_grid = True
    else:
        updated_grid = False
        print('  - Did not find triangles with more than one side facing the boundary')
    return updated_grid
