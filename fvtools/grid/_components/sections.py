"""Transect sampling and velocity interpolation mixin for FVCOM_grid.

Imported/re-exported by fvtools.grid.fvcom_grd for backward compatibility.
The implementation is moved verbatim; numerical behavior is unchanged.
"""

import asyncio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from pyproj import Proj


class SectionMaker:
    '''
    Rotuines to define a section within the grid, interpolate data to the section and return the transect
    - Will actually not be 100% correct, since we "crop" the sea surface elevation somewhat to make the bottom appear to be static. That is not the case, the z-level of
      the bottom sigma level is not static (since min(sigma)>-1).
    - Still room to make this more consise...
    '''
    @property
    def dst_sec(self):
        '''Distance along a transect measured in kilometers from first point'''
        dx = np.diff(self.x_sec, prepend = self.x_sec[0])/1000 # since we count starting with the first segment position
        dy = np.diff(self.y_sec, prepend = self.y_sec[0])/1000 # dive by 1000 to get distance in kilometers
        return np.cumsum(np.sqrt(dx**2 + dy**2))

    @property
    def dpt_sec(self):
        """
        Calculate physical vertical coordinates along the transect.
        Supports node- and cell-based data, sigma layers andsigma levels, including horizontally varying sigma distributions.

        Returns
        -------
        np.ndarray
            Vertical coordinates in meters, with shape (n_section_points, n_vertical_levels).
            Coordinates are relative to mean sea level, positive upwards.
        """
        # Select horizontal grid and vertical coordinates
        if self._transect_grid == "":
            sigma = getattr(self, self._transect_sigma)
            h = self.h
            zeta = self.zeta

        elif self._transect_grid == "c":
            sigma = getattr(self, f"{self._transect_sigma}_c")
            h = self.hc
            zeta = np.mean(self.zeta[self.tri], axis=1)

        else:
            raise ValueError("Invalid transect grid type")

        # Physical vertical coordinates at every grid point
        z_grid = (zeta[:, None] + (h + zeta)[:, None] * sigma)

        # Visualization adjustment:
        # - Keep the deepest sigma-layer midpoint stationary when the sea surface elevation changes.
        z_grid -= zeta[:, None] * (1.0 + sigma[:, [-1]])

        # - Stick the upper level to the sea surface, not to the uppermost siglay point
        z_grid[:, 0] = zeta

        # Interpolate each vertical level to the transect
        return self._interpolate_3D(z_grid)


    @property
    def _min_sigma(self):
        '''bottom sigma level value. included since I want to keep bottom depth constant in section movies (which is a lie...)'''
        _min_sigma = -1
        if self._static_depth: _min_sigma = self.siglay[-1, -1]
        return _min_sigma

    def prepare_section(self, section_file=None, res=None, store_transect_img=False, section_points=None):
        """
        Prepare coordinates along a transect for section analysis.

        The transect can be defined using a section file, interactively,
        or by providing section points directly.
        """
        if section_points is not None:
            x_section, y_section = section_points

        elif section_file is not None:
            x_section, y_section = self._load_section_file(section_file)

        else:
            x_section, y_section = self._graphical_section()

        if res is None:
            res = np.sum(
                np.hypot(np.diff(x_section), np.diff(y_section))
            ) / 60

        self._assemble_section(x_section, y_section, res)

        if section_file is None:
            self.write_section()

        self.__dict__.pop('trs', None)
        self.__dict__.pop('_section_weights_cache', None)
        self.__dict__.pop('_section_vector_basis_cache', None)

        if store_transect_img:
            self._plot_section_line()

        # Do not close the interactive figure in JupyterLab
        if section_points is None:
            plt.close('all')

    # Rotuines that define where the section goes (ie. defining nodes on the section line)
    def _graphical_section(self):
        '''Define a transect by plotting on a map'''
        c = self.plot_contour(self.h, levels = np.linspace(self.h.min(), self.h.max(), 30))
        self.plot_grid(rasterized=True)
        plt.title('Click where you want the section to go (minimum two points). Hit "Enter" to continue.')
        places = plt.ginput(timeout = -1, n = -1)
        return np.array([p[0] for p in places]), np.array([p[1] for p in places])

    async def prepare_section_interactive(self, res=None, store_transect_img=False):
        """
        Interactively define a transect in JupyterLab.

        Left click: Add point
        Backspace/Delete: Remove last point
        Enter or Finish button: Complete the selection

        Returns
        -------
        x, y : np.ndarray
            Coordinates of the selected transect vertices.

        Example use in jupyterlab:
        %matplotlib widget
        points = await M.prepare_section_interactive()

        - will also write .x_sec, .y_sec to the grid object
        """
        points = []
        future = asyncio.get_running_loop().create_future()

        # Create the map without automatically displaying the figure
        with plt.ioff():
            plt.figure()
            self.georeference()
            self.plot_contour(self.h, levels = np.linspace(self.h.min(), self.h.max(), 30), show = False)
            self.plot_grid(rasterized=True, show = False)

            fig = plt.gcf()
            ax = plt.gca()

            line, = ax.plot([], [], "ro-", markersize=5, linewidth=1.5, zorder=100)

            ax.set_title("Create a transect")

        # Create controls
        finish_button = widgets.Button(description="Finish", button_style="success")
        undo_button = widgets.Button(description="Undo")
        status = widgets.Label(value="Select at least two points.")

        def redraw():
            """Update the transect line and status."""
            line.set_data([p[0] for p in points], [p[1] for p in points])
            status.value = f"{len(points)} points selected"
            fig.canvas.draw_idle()

        def on_click(event):
            """Register a new point from a left mouse click."""
            if future.done():
                return

            if event.inaxes != ax or event.button != 1:
                return

            # Ignore clicks while the toolbar is in pan/zoom mode
            toolbar = getattr(fig.canvas, "toolbar", None)
            if toolbar is not None and toolbar.mode:
                return

            if event.xdata is not None and event.ydata is not None:
                points.append((event.xdata, event.ydata))
                redraw()

        def on_undo(_=None):
            """Remove the most recently selected point."""
            if points and not future.done():
                points.pop()
                redraw()

        def on_finish(_=None):
            """Finish selecting the transect."""
            if future.done():
                return

            if len(points) < 2:
                status.value = "At least two points are required."
                return

            future.set_result(np.asarray(points, dtype=float))

        def on_key(event):
            """Handle keyboard shortcuts."""
            if event.key == "enter":
                on_finish()
            elif event.key in ("backspace", "delete"):
                on_undo()

        # Connect Matplotlib events
        click_id = fig.canvas.mpl_connect(
            "button_press_event", on_click
        )
        key_id = fig.canvas.mpl_connect(
            "key_press_event", on_key
        )

        finish_button.on_click(on_finish)
        undo_button.on_click(on_undo)

        # Display the interactive map and controls
        display(widgets.VBox([
            fig.canvas,
            widgets.HBox([finish_button, undo_button]),
            status
        ]))

        try:
            # Yield control to Jupyter while waiting for user input
            xy = await future

        finally:
            # Disconnect event handlers
            fig.canvas.mpl_disconnect(click_id)
            fig.canvas.mpl_disconnect(key_id)
            finish_button.disabled = True
            undo_button.disabled = True

        x_section = xy[:, 0]
        y_section = xy[:, 1]

        # Use the existing section preparation routine
        self.prepare_section(
            res=res,
            store_transect_img=store_transect_img,
            section_points=(x_section, y_section)
        )

        return x_section, y_section

    def _load_section_file(self, file):
        '''
        Load section coordinates from a file
        - accepts both ',' and ' ' as delimiters
        '''
        try:
            lon, lat = np.loadtxt(file, delimiter = ',', unpack = True)
        except:
            lon, lat = np.loadtxt(file, delimiter = ' ', unpack = True)
        return self.Proj(lon, lat)

    def _assemble_section(self, x_section, y_section, res):
        """Create equally spaced transect points along each segment."""

        points = np.column_stack((x_section, y_section))

        if len(points) < 2:
            raise ValueError("A section requires at least two points")

        if not np.isfinite(res) or res <= 0:
            raise ValueError("res must be positive")

        segments = []

        for start, stop in zip(points[:-1], points[1:]):
            distance = np.linalg.norm(stop - start)

            if distance == 0:
                continue

            n = max(1, int(np.ceil(distance / res)))
            t = np.linspace(0, 1, n + 1)

            # Avoid duplicate points at segment junctions.
            if segments:
                t = t[1:]

            segments.append(
                start + t[:, None] * (stop - start)
            )

        if not segments:
            raise ValueError("Section has zero length")

        xy = np.concatenate(segments, axis=0)
        self.x_sec, self.y_sec = xy.T

    def get_section_data(self, data,
                         section_file = None, res = None,
                         store_transect_img = False):
        '''
        Dump data from from the full solution to an interpolated transect  (ie. not following nodes/cells)
        - data:         Field for transect calculation
        - section_file: File with lon lat coordinate positions of segment fixed points. You can either use space or , as separator
        - res:          Resolution of the transect in m (defaults to 1/100 of span)
        - store_transect_img: Store a plot showing the transect overlayd on the grid

        Output (as dictionary):
          - x, y, dst, h positions, distance along and depth of transect
          - transect: field you gave the routine at x, y
        '''
        out = {}
        if data.shape[0] not in (self.node_number, self.cell_number):
            data = data.T

        sigma = self._check_if_siglay_or_siglev(data)
        grid = self._check_if_cell_or_node(data)
        self._initialize_section(section_file, res, store_transect_img, sigma, grid) # TBD: in here, find way to remove points outside of the domain?
        out['h'], out['x'], out['y'] = self.dpt_sec, self.x_sec, self.y_sec

        # Interpolate data to the section
        # ----
        if len(data.shape)>1:
            out['transect'] = self._interpolate_3D(data)
            out['dst'] = np.repeat(self.dst_sec[:, None], data.shape[1], axis = 1)
        else:
            out['transect'] = self._interpolate_2D(data)
            out['dst'] = self.dst_sec

        # Remove nans and crop dict, contourf won't accept the input otherwise -- do this earlier though, not every time...
        # ----
        if np.isnan(out['transect']).any():
            if len(out['transect'].shape) == 1:
                not_horizontal = ~np.isnan(out['transect'])
            else:
                not_horizontal = ~np.isnan(out['transect'][:, 0])
            for key in out.keys():
                if len(out[key].shape) == 1:
                    out[key] = out[key][not_horizontal]
                else:
                    out[key] = out[key][not_horizontal, :]
        return out

    def _section_velocity_basis(self, velocity_coordinates='east_north'):
        """Unit tangent and normal in the velocity component coordinate frame.

        x_sec/y_sec must be in a metric, projected CRS. For east_north,
        rotate geographic east/north into grid x/y using short geodesic steps;
        the resulting basis is cached across frames.
        """
        if velocity_coordinates not in ('east_north', 'projected'):
            raise ValueError("velocity_coordinates must be 'east_north' or 'projected'")

        xy = np.column_stack((self.x_sec, self.y_sec))
        cache = self.__dict__.get('_section_vector_basis_cache')
        if (cache is not None and cache['coordinates'] == velocity_coordinates
                and cache['reference'] == self.reference
                and np.array_equal(cache['xy'], xy)):
            return cache['tangent_e'], cache['normal_e'], cache['tangent_n'], cache['normal_n']

        if len(xy) < 2 or not np.isfinite(xy).all():
            raise ValueError('Section requires at least two finite projected points')
        from pyproj import CRS, Geod
        crs = CRS.from_user_input(self.reference)
        if not crs.is_projected or not all(axis.unit_conversion_factor and
                                            np.isclose(axis.unit_conversion_factor, 1.0)
                                            for axis in crs.axis_info[:2]):
            raise ValueError('Section coordinates must use a projected CRS with metre units')

        # A length-weighted central tangent is appropriate for an interpolated
        # polyline. At sharp corners its normal is an average of both segments.
        dx = np.gradient(xy[:, 0])
        dy = np.gradient(xy[:, 1])
        length = np.hypot(dx, dy)
        if np.any(length <= 0):
            raise ValueError('Section contains points with undefined tangent')
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx  # positive normal: left of start -> end

        if velocity_coordinates == 'east_north':
            lon, lat = self.Proj(xy[:, 0], xy[:, 1], inverse=True)
            geod = Geod(ellps='WGS84')
            lon_e, lat_e, _ = geod.fwd(lon, lat, np.full(len(xy), 90.0), np.full(len(xy), 10.0))
            lon_n, lat_n, _ = geod.fwd(lon, lat, np.zeros(len(xy)), np.full(len(xy), 10.0))
            xe, ye = self.Proj(lon_e, lat_e)
            xn, yn = self.Proj(lon_n, lat_n)
            e_dx, e_dy = xe - xy[:, 0], ye - xy[:, 1]
            n_dx, n_dy = xn - xy[:, 0], yn - xy[:, 1]
            e_norm = np.hypot(e_dx, e_dy)
            n_norm = np.hypot(n_dx, n_dy)
            if np.any(e_norm <= 0) or np.any(n_norm <= 0):
                raise ValueError('Could not derive east/north vectors from CRS')
            ex, ey = e_dx / e_norm, e_dy / e_norm
            gx, gy = n_dx / n_norm, n_dy / n_norm
            tangent_e, normal_e = tx * ex + ty * ey, nx * ex + ny * ey
            tangent_n, normal_n = tx * gx + ty * gy, nx * gx + ny * gy
        else:
            tangent_e, normal_e, tangent_n, normal_n = tx, nx, ty, ny

        self._section_vector_basis_cache = dict(
            xy=xy.copy(), reference=self.reference, coordinates=velocity_coordinates,
            tangent_e=tangent_e, normal_e=normal_e,
            tangent_n=tangent_n, normal_n=normal_n,
        )
        return tangent_e, normal_e, tangent_n, normal_n

    def get_section_velocity(self, u, v, velocity_coordinates='east_north'):
        """Interpolate FVCOM 3-D cell velocities onto the current section.

        Parameters
        ----------
        u, v : array-like
            Cell-centred velocities with shape (sigma_layers, n_cells) or
            (n_cells, sigma_layers). Normally read using load_netCDF.
        velocity_coordinates : {'east_north', 'projected'}
            'east_north' (default): u geographic east and v geographic north.
            'projected': u/v already aligned to the projected mesh x/y axes.
            Section x/y must be in metres in either case.

        Returns
        -------
        dict
            'normal', 'along', 'speed', 'u', 'v', 'h', 'dst', 'x', 'y'.
            Arrays of velocity and physical z coordinate are indexed by
            (section_point, sigma_layer). Positive normal is to the LEFT
            looking from the first section point toward the last.
            NaNs in either velocity component mask BOTH components.
        """
        def normalize(field, name):
            arr = np.ma.filled(np.ma.asarray(field, dtype=float), np.nan)
            layers = self.siglay.shape[1]
            if arr.ndim != 2:
                raise ValueError(f'{name} must be a 3-D FVCOM velocity field at one timestep')
            if arr.shape == (layers, self.cell_number):
                return arr.T
            if arr.shape == (self.cell_number, layers):
                return arr
            raise ValueError(f'{name} has shape {arr.shape}; expected (sigma, cell) or (cell, sigma)')

        u = normalize(u, 'u')
        v = normalize(v, 'v')
        if not hasattr(self, 'x_sec') or not hasattr(self, 'y_sec'):
            raise ValueError('Call prepare_section() before get_section_velocity()')

        self._initialize_section(None, None, False, 'siglay', 'c')
        u_sec = self._interpolate_3D(u)
        v_sec = self._interpolate_3D(v)

        # Physical z of each SIGLAY midpoint. Unlike the visualization-only
        # dpt_sec, this is NOT adjusted to pin the bottom at a fixed level.
        # Masked/dry zeta must not change section geometry from frame to frame.
        # Velocity is still masked there, but use zeta=0 to locate its grid layers.
        safe_zeta = np.where(np.isfinite(self.zeta), self.zeta, 0.0)
        zeta_cells = np.mean(safe_zeta[self.tri], axis=1)
        z_cells = (zeta_cells[:, None]
                   + (self.hc + zeta_cells)[:, None] * self.siglay_c)
        z_sec = self._interpolate_3D(z_cells)

        te, ne, tn, nn = self._section_velocity_basis(velocity_coordinates)
        normal = u_sec * ne[:, None] + v_sec * nn[:, None]
        along = u_sec * te[:, None] + v_sec * tn[:, None]
        speed = np.hypot(u_sec, v_sec)

        invalid = (~np.isfinite(u_sec) | ~np.isfinite(v_sec) | ~np.isfinite(z_sec))
        for arr in (normal, along, speed, u_sec, v_sec):
            arr[invalid] = np.nan

        # Exclude only locations lacking GEOMETRY (not temporarily dry
        # locations); prevents changing horizontal sampling across frames.
        valid_rows = np.all(np.isfinite(z_sec), axis=1) & np.isfinite(self.dst_sec)
        dst = np.broadcast_to(self.dst_sec[:, None], z_sec.shape)
        return dict(
            u=u_sec[valid_rows], v=v_sec[valid_rows],
            normal=normal[valid_rows], along=along[valid_rows], speed=speed[valid_rows],
            h=z_sec[valid_rows], dst=dst[valid_rows],
            x=np.asarray(self.x_sec)[valid_rows], y=np.asarray(self.y_sec)[valid_rows],
        )

    def _initialize_section(self, section_file, res, store_transect_img, sigma, grid):
        '''
        Prepare the section interpolators etc.
        '''
        # A cached triangulation for node data must not be used for cell data.
        if getattr(self, '_transect_grid', None) != grid:
            self.__dict__.pop('trs', None)
            self.__dict__.pop('_section_weights_cache', None)
        self._transect_sigma, self._transect_grid = sigma, grid
        if not (hasattr(self, 'x_sec') and hasattr(self, 'y_sec')):
            self.prepare_section(section_file=section_file, res=res,
                                 store_transect_img=store_transect_img)
        if not hasattr(self, 'trs'):
            self.trs = matplotlib.tri.Triangulation(
                getattr(self, f'x{self._transect_grid}'), 
                getattr(self, f'y{self._transect_grid}'), 
                triangles = getattr(self, f'{self._transect_grid}tri')
                )

    def _section_weights(self):
        """Return cached triangle vertices/weights for the section points.

        A triangulation lookup is needed only when the transect/grid changes,
        not for every vertical level and every movie frame.
        """
        xy = np.column_stack((self.x_sec, self.y_sec))
        cache = self.__dict__.get('_section_weights_cache')
        if (cache is not None and cache['triangulation'] is self.trs
                and np.array_equal(cache['xy'], xy)):
            return cache['vertices'], cache['weights'], cache['inside']

        triangle_ids = np.asarray(self.trs.get_trifinder()(
            xy[:, 0], xy[:, 1]), dtype=int)
        inside = triangle_ids >= 0
        vertices = np.zeros((len(xy), 3), dtype=int)
        weights = np.zeros((len(xy), 3), dtype=float)
        if inside.any():
            vertices[inside] = self.trs.triangles[triangle_ids[inside]]
            v = vertices[inside]
            x0, y0 = self.trs.x[v[:, 0]], self.trs.y[v[:, 0]]
            x1, y1 = self.trs.x[v[:, 1]], self.trs.y[v[:, 1]]
            x2, y2 = self.trs.x[v[:, 2]], self.trs.y[v[:, 2]]
            xx, yy = xy[inside].T
            denominator = ((y1 - y2) * (x0 - x2)
                           + (x2 - x1) * (y0 - y2))
            w0 = ((y1 - y2) * (xx - x2)
                  + (x2 - x1) * (yy - y2)) / denominator
            w1 = ((y2 - y0) * (xx - x2)
                  + (x0 - x2) * (yy - y2)) / denominator
            weights[inside] = np.column_stack((w0, w1, 1.0 - w0 - w1))

        self._section_weights_cache = {
            'triangulation': self.trs, 'xy': xy.copy(),
            'vertices': vertices, 'weights': weights, 'inside': inside
        }
        return vertices, weights, inside

    def _interpolate_2D(self, field):
        """Interpolate a 1-D nodal/cell field to the fixed section."""
        values = np.ma.filled(np.ma.asarray(field, dtype=float), np.nan)
        if values.ndim != 1 or len(values) != len(self.trs.x):
            raise ValueError('field must have one value per triangulation vertex')
        vertices, weights, inside = self._section_weights()
        out = np.full(len(self.x_sec), np.nan)
        if inside.any():
            out[inside] = np.sum(values[vertices[inside]] * weights[inside], axis=1)
        return out

    def _interpolate_3D(self, field):
        """Interpolate (grid_point, level) data using cached section weights."""
        values = np.ma.filled(np.ma.asarray(field, dtype=float), np.nan)
        if values.ndim != 2 or values.shape[0] != len(self.trs.x):
            raise ValueError('field must have shape (grid_point, level)')
        vertices, weights, inside = self._section_weights()
        out = np.full((len(self.x_sec), values.shape[1]), np.nan)
        if inside.any():
            out[inside] = np.einsum(
                'pv,pvk->pk', weights[inside], values[vertices[inside]],
                optimize=True)
        return out
