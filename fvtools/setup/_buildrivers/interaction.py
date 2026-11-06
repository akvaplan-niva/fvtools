"""Interactive Jupyter/ipympl river outlet editing and domain selection."""
import asyncio
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import shapely as shp
from IPython.display import display
from shapely.plotting import plot_polygon


class DraggablePoints:
    """Drag outlets in a widget figure and display vectors from original outlets."""

    def __init__(self, ax, x_data, y_data, x_original, y_original):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.x_data = np.asarray(x_data, dtype=float).copy()
        self.y_data = np.asarray(y_data, dtype=float).copy()
        self.original_x = np.asarray(x_original, dtype=float).copy()
        self.original_y = np.asarray(y_original, dtype=float).copy()

        n = len(self.x_data)
        if any(len(arr) != n for arr in
               (self.y_data, self.original_x, self.original_y)):
            raise ValueError('Original and current river coordinate arrays must align')

        self.selected_index = None
        self.was_modified = False

        # Use ONE line artist for all rivers rather than thousands of artists.
        # Each vector is [original, current, NaN] in plotting coordinates.
        self._line_x = np.full(3 * n, np.nan)
        self._line_y = np.full(3 * n, np.nan)
        self._line_x[0::3] = self.original_x
        self._line_x[1::3] = self.x_data
        self._line_y[0::3] = self.original_y
        self._line_y[1::3] = self.y_data
        self.displacement_lines, = ax.plot(
            self._line_x, self._line_y, '-', color='darkorange',
            linewidth=1.2, alpha=0.8, zorder=7,
            label='Original → edited position'
        )

        self.original_points, = ax.plot(
            self.original_x, self.original_y, '.', color='darkorange',
            markersize=3, alpha=0.7, zorder=7
        )
        self.points, = ax.plot(
            self.x_data, self.y_data, 'ko', markersize=10, zorder=8,
            label='Draggable outlets'
        )

        self.cid_press = self.canvas.mpl_connect(
            'button_press_event', self.on_press
        )
        self.cid_motion = self.canvas.mpl_connect(
            'motion_notify_event', self.on_motion
        )
        self.cid_release = self.canvas.mpl_connect(
            'button_release_event', self.on_release
        )

        self.done_button = widgets.Button(
            description='Done', button_style='success', icon='check'
        )
        self.done_button.on_click(self.on_done)
        self._future = None
        display(self.done_button)

    def on_press(self, event):
        if event.inaxes != self.ax or not len(self.x_data):
            return
        pixels = self.ax.transData.transform(
            np.column_stack((self.x_data, self.y_data))
        )
        distances = np.linalg.norm(pixels - [event.x, event.y], axis=1)
        closest = np.argmin(distances)
        if distances[closest] < 15:
            self.selected_index = closest

    def on_motion(self, event):
        if (self.selected_index is None or event.inaxes != self.ax
                or event.xdata is None or event.ydata is None):
            return
        i = self.selected_index
        self.was_modified = True
        self.x_data[i] = event.xdata
        self.y_data[i] = event.ydata
        self.points.set_data(self.x_data, self.y_data)
        self._line_x[3 * i + 1] = event.xdata
        self._line_y[3 * i + 1] = event.ydata
        self.displacement_lines.set_data(self._line_x, self._line_y)
        self.canvas.draw_idle()

    def on_release(self, event):
        self.selected_index = None

    def on_done(self, event):
        self.done_button.disabled = True
        self.done_button.description = 'Done ✓'
        if self._future is not None and not self._future.done():
            self._future.set_result(self.get_data())

    async def wait(self):
        """Suspend this coroutine until Done, without blocking Jupyter events."""
        loop = asyncio.get_running_loop()
        self._future = loop.create_future()
        return await self._future

    def get_data(self):
        return self.x_data.copy(), self.y_data.copy()

    def disconnect(self):
        self.canvas.mpl_disconnect(self.cid_press)
        self.canvas.mpl_disconnect(self.cid_motion)
        self.canvas.mpl_disconnect(self.cid_release)

class CropRivers:
    '''
    Use FVCOM_rivers to crop the national river database to the FVCOM experiment
    - Includes functions to move rivers in the domain to match with the mesh
    '''
    @property
    def model_domain(self):
        '''
        The outermost solid boundary of the model domain
        '''
        return shp.geometry.Polygon(self.M.model_boundary)

    @property
    def buffered_domain(self):
        '''
        A concave hull of the solid boundary, used to filter out rivers in nearby fjords draining to areas outside of the model domain
        '''
        return shp.concave_hull(shp.geometry.Polygon(self.model_domain), ratio = self.info['buffer domain ratio']).buffer(self.info['land buffer'])

    async def crop_rivers_far_from_land_and_close_to_OBC(self, Rivers):
        """Interactively edit outlets, retain original coordinates and crop rivers."""
        print('- Adjust river positions and remove rivers outside the model domain')
        modified = False
        iteration = 0

        while True:
            iteration += 1
            print(f' - iteration {iteration}')
            # Create the figure BEFORE georeferencing. M.georeference()
            # configures the current figure (Cartopy/map background), so a
            # subsequent plt.subplots() would discard that georeferenced Axes.
            fig = plt.figure(figsize=(10, 10))
            self.M.re_project(self.info['grid_projection'])
            self.M.georeference()
            ax = plt.gca()

            # All manually drawn positions use the grid projection of this map.
            x, y = self.M.Proj(Rivers.lon, Rivers.lat)
            in_model = np.asarray(self.buffered_domain.contains(
                [shp.geometry.Point(a, b) for a, b in zip(x, y)]
            ), dtype=bool)

            ax.plot(self.x_land, self.y_land, 'b.', label='land nodes', zorder=1)
            plot_polygon(self.buffered_domain, ax=ax, facecolor='lightblue',
                         edgecolor='blue', alpha=0.5)
            ax.set_title('Drag black river points, then press Done. Orange lines show original outlets.')

            # Use projected x/y directly, as for the draggable dots and lines.
            # Do not pass Rivers.crs here: those coordinates are in the
            # database projection, which may differ from the grid projection.
            ax.scatter(x[~in_model], y[~in_model],
                       s=Rivers.rivers.areal.to_numpy()[~in_model], color='r',
                       label='Will not be used')
            ax.scatter(x[in_model], y[in_model],
                       s=Rivers.rivers.areal.to_numpy()[in_model], color='g',
                       label='Will be used')

            original_lon, original_lat = Rivers.Proj(
                Rivers.rivers['x_original'].to_numpy(),
                Rivers.rivers['y_original'].to_numpy(), inverse=True)
            xo, yo = self.M.Proj(original_lon, original_lat)
            editor = DraggablePoints(ax, x, y, xo, yo)
            # Scale scatter markers in the legend independently of runoff area.
            # This leaves the actual map symbols unchanged.
            legend = ax.legend(loc='upper right', fontsize=9)
            for handle in legend.legend_handles:
                if hasattr(handle, 'set_sizes'):
                    handle.set_sizes([24.0])
            plt.show()
            try:
                edited_x, edited_y = await editor.wait()
            finally:
                editor.disconnect()
                plt.close(fig)

            if not editor.was_modified:
                break

            modified = True
            lon, lat = self.M.Proj(edited_x, edited_y, inverse=True)
            new_x, new_y = Rivers.Proj(lon, lat)
            Rivers.x = np.asarray(new_x)
            Rivers.y = np.asarray(new_y)
            print('  - River positions updated; recalculating the domain mask')

        # The final mask is calculated from the last confirmed coordinates.
        self.M.re_project(self.info['grid_projection'])
        x, y = self.M.Proj(Rivers.lon, Rivers.lat)
        in_model = np.asarray(self.buffered_domain.contains(
            [shp.geometry.Point(a, b) for a, b in zip(x, y)]
        ), dtype=bool)
        Rivers.crop(np.flatnonzero(in_model))

        if modified:
            # Save both x/y_outlet and unchanged x/y_original in the case CSV.
            output_dir = os.path.dirname(self.M.filepath)
            new_name = os.path.join(output_dir, f'river_positions_{self.M.casename}.csv')
            Rivers.rivers.to_csv(new_name)
            print(f'- Saved edited river positions and original coordinates to {new_name}')

        self.M.re_project(Rivers.reference)
        print(f'- Remove rivers within {self.info["obc buffer"]} m of the open boundary')
        distances, _ = self.obc_tree.query(np.column_stack((Rivers.x, Rivers.y)))
        Rivers.crop(np.flatnonzero(distances > self.info['obc buffer']))
        return Rivers
