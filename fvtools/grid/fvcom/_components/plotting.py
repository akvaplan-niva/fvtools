"""FVCOM mesh, contour, georeferenced and section plotting mixin.

The implementation is extracted verbatim from fvcom_grd.py.
"""

from functools import cached_property

import cmocean as cmo
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class PlotFVCOM:
    @cached_property
    def mtri_nodes(self):
        """Cached Matplotlib triangulation for nodal fields.

        The geometry is shared by successive movie frames.  Only the
        triangle mask changes for wet/dry or below-bottom values.
        """
        return matplotlib.tri.Triangulation(self.x, self.y, self.tri)

    @cached_property
    def mtri_cells(self):
        """Cached cell-centre triangulation for FVCOM velocity fields."""
        return matplotlib.tri.Triangulation(self.xc, self.yc, self.ctri)

    @property
    def transform(self):
        if not hasattr(self, '_transform'):
            return None
        return self._transform

    @transform.setter
    def transform(self, var):
        self._transform = var

    def plot_grid(self, ax = None, c = 'g-', linewidth = 0.2, markersize = 0.2, show = True, *args, **kwargs):
        '''
        Plot mesh grid
        - arguments and keyword arguments are passed to pyplot.triplot
        '''
        if ax == None:
            ax = plt.gca()
        kwargs = self._transform_to_kwargs(**kwargs)
        ax.triplot(self.x, self.y, self.tri, c, *args, markersize=markersize, linewidth=linewidth, **kwargs)
        ax.set_aspect('equal')
        if show: plt.show(block = False)

    def plot_obc(self, ax = None, **kwargs):
        '''
        plot the obc nodes, show different nodestrings
        '''
        if ax == None:
            ax = plt.gca()
        kwargs = self._transform_to_kwargs(**kwargs)
        for i in range(len(self.x_obc)):
            ax.scatter(self.x_obc[i], self.y_obc[i], zorder = 10, label=f'boundary nr. {i+1}', **kwargs)
            ax.scatter(self.x_obc[i][[0,-1]], self.y_obc[i][[0,-1]], zorder = 11, c = 'r', **kwargs) # start and end points 
        #plt.draw()

    def plot_contour(self, field, show=True, ax=None, *args,
                     on_empty='raise', **kwargs):
        """Contour node- or cell-centred data on a cached triangulation.

        The mesh geometry is created once per grid.  Invalid values in a
        timestep only update its triangle mask, so masked z-level fields
        never reach an unmasked triangle in ``tricontourf``.  This method
        never copies or mutates the FVCOM connectivity arrays themselves.

        The cached triangulation's mask is the mask from its **most recent**
        contour call.  Callers should not rely on that mask remaining fixed.
        """
        if on_empty not in ('raise', 'skip'):
            raise ValueError("on_empty must be 'raise' or 'skip'")

        values = np.ma.asarray(np.squeeze(field), dtype=float)
        if values.ndim != 1:
            raise ValueError(f'Expected a one-dimensional contour field, got {values.shape}')
        if values.size == self.node_number:
            triangulation = self.mtri_nodes
        elif values.size == self.cell_number:
            triangulation = self.mtri_cells
        else:
            raise ValueError(
                f'Field length {values.size} does not match '
                f'{self.node_number} nodes or {self.cell_number} cells'
            )

        # Matplotlib rejects *masked* or non-finite values in the vertices
        # of a visible triangle.  Mask triangles, not just vertex data.
        data = np.asarray(np.ma.masked_invalid(values).filled(np.nan))
        triangles = triangulation.triangles
        triangle_mask = (~np.isfinite(data))[triangles].any(axis=1)
        caller_mask = kwargs.pop('mask', None)
        if caller_mask is not None:
            caller_mask = np.asarray(caller_mask, dtype=bool)
            if caller_mask.shape != (len(triangles),):
                raise ValueError('Triangle mask has wrong shape')
            triangle_mask |= caller_mask

        # set_mask() invalidates Matplotlib's cached C++ triangulation. Avoid
        # calling it when the mask has not changed (common for dry-area masks).
        previous = triangulation.mask
        replacement = triangle_mask if np.any(triangle_mask) else None
        changed = ((previous is None) != (replacement is None))
        if not changed and previous is not None:
            changed = not np.array_equal(previous, replacement)
        if changed:
            triangulation.set_mask(replacement)

        if replacement is not None and np.all(replacement):
            if on_empty == 'skip':
                return None
            raise ValueError('No valid triangles to contour')
        if triangles.size == 0:
            if on_empty == 'skip':
                return None
            raise ValueError('No triangles to contour')

        kwargs = self._transform_to_kwargs(ax=ax, **kwargs)
        return self._plot_contour(triangulation, data, show, ax=ax,
                                  *args, **kwargs)

    def _transform_to_kwargs(self, ax=None, **kwargs):
        """Pass the CRS transform to a Cartopy axes when applicable.

        Explicit axes avoid pyplot global-figure lookups during animation
        saving, where Matplotlib temporarily detaches the figure manager.
        """
        if self.transform is not None:
            import cartopy.mpl.geoaxes as geoaxes
            target = ax if ax is not None else plt.gca()
            if isinstance(target, geoaxes.GeoAxes):
                kwargs['transform'] = self.transform
        return kwargs

    def georeference(self, 
                    url='https://wms.geonorge.no/skwms1/wms.topograatone?service=wms&request=getcapabilities', 
                    layers=['topograatone'], wms=None, 
                    depth = True,
                    ax = None):
        '''
        Plot map data from WMS server as georeference. Must be done before plotting grid, contours etc.
        - requires cartopy
        ---
        url:    defaults to grey norgeskart
                - For international use, try e.g. url = 'https://ows.terrestris.de/osm/service?', layers = ['OSM-WMS-no-labels']
        layers: valid layer from the url, must be a tuple with string(s)
        wms:    overwrites other input if not None, current options: "raster" or "topo4"
        depth:  plot kartverket bathymetry
        ax:     cartopy axes (e.g. fig, ax = plt.subplots(2, 1, subplot_kw={'projection': ccrs.PlateCarree()}))

        Returns:
        ---
        ax, a GeoAxesSubplot

        - Also possible to plot fiskeridirektoratet akvakultur lokaliteter WMS on top:
          - ax.add_wms(wms='https://gis.fiskeridir.no/server/services/fiskeridirWMS_akva/MapServer/WMSServer?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities', 
                       layers=['akvakultur_lokaliteter'])
        '''
        if wms is not None:
            if wms =='raster':
                url = 'http://openwms.statkart.no/skwms1/wms.toporaster4?version=1.3.0&service=wms&request=getcapabilities'
                layers = ['toporaster']
            elif wms == 'topo4':
                url = 'https://openwms.statkart.no/skwms1/wms.topo4?service=wms&request=getcapabilities'
                layers = ['topo4_WMS']
            elif wms == 'sjokart':
                url = 'https://wms.geonorge.no/skwms1/wms.sjokartraster2?service=wms&version=1.3.0&request=getcapabilities'
                layers = ['Overseiling']
                depth = False
        if not ax:
            if int(self.reference.split(':')[-1]) == 4326:
                lonmin, lonmax = self.lon.min(), self.lon.max()
                latmin, latmax = self.lat.min(), self.lat.max()
                aspect_ratio = (float(latmax - latmin) / (float(lonmax - lonmin)))/np.cos(np.radians((latmin + latmax) / 2))

                # Draw, set projection
                crs = ccrs.Mercator()
                fig = plt.figure(figsize=(11. / aspect_ratio, 11.))
                ax = fig.add_subplot(111, projection=crs)
                self.draw_geo_grid([lonmin, lonmax, latmin, latmax])
                fig.canvas.draw()         
                fig.set_tight_layout(True)
                self.transform = ccrs.PlateCarree(globe=crs.globe)
            else:
                ax = plt.axes(projection = ccrs.epsg(int(self.reference.split(':')[-1])))
        
        ax.add_wms(wms=url, layers=layers)
        if wms != 'raster':
            if depth:
                ax.add_wms('https://wms.geonorge.no/skwms1/wms.dybdedata2?service=WMS&request=GetCapabilities', layers = ['Dybdedata2'])
        return ax

    def draw_geo_grid(self, extent):
        '''
        Automatically draw a geo-grid over the extent (lonmin, lonmax, latmin, latmax)
        '''
        ax = plt.gca()
        ax.set_extent(extent, crs=ccrs.PlateCarree(globe=ccrs.Mercator().globe))
        gl = ax.gridlines(ccrs.PlateCarree(globe=ccrs.Mercator().globe), draw_labels=True)         
        gl.top_labels = None

    @staticmethod
    def _plot_contour(triangulation, field, show=False, ax=None, *args, **kwargs):
        """Draw on the already constructed/masked triangulation."""
        if ax is None:
            ax = plt.gca()
        cont = ax.tricontourf(triangulation, field, *args, **kwargs)
        ax.set_aspect('equal')
        if show:
            plt.show(block=False)
        return cont

    def _plot_section_line(self):
        '''
        Show where the section goes relative to the grid
        '''
        self.plot_grid(rasterized=True, show = False)
        plt.plot(self.x_sec, self.y_sec, 'k', zorder = 10)
        plt.plot(self.x_sec[0], self.y_sec[0], 'g.', zorder = 11, label = 'start')
        plt.plot(self.x_sec[-1], self.y_sec[-1], 'r.', zorder = 11, label = 'stop')
        plt.xlim([self.x_sec.min()-10000, self.x_sec.max()+10000])
        plt.ylim([self.y_sec.min()-10000, self.y_sec.max()+10000])
        plt.legend()
        plt.title('Section')
        plt.savefig('Section_map.png')

    def plot_section_3D(self, x, y, z, data, vmin = None, cmap = cmo.cm.balance):
        '''
        Plot data at scattered x,y,z points
        - x and y are (N,) arrays
        - z and data are (N, D) arrays

        Optional
        
        '''
        import matplotlib.colors as colors

        vmin = vmin if vmin is not None else data.min()

        # Expand array
        x = np.tile(x[:, None], [1, data.shape[1]])
        y = np.tile(y[:, None], [1, data.shape[1]])

        # Prepare the figure
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        norm = colors.Normalize(vmin = vmin, vmax = data.max())

        # Plot the data
        surf = ax.plot_surface(x, y, z, facecolors = cmap(norm(data)), cmap = cmap)
        surf.set_clim(vmin = 0, vmax = data.max())
        fig.colorbar(surf, label = 'm/s')
        ax.set_zlabel('depth [m]')
        return fig, ax
