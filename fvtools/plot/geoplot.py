# -------------------------------------------------------------
# Create georeferenced images using contextily which extracts
# georeferenced images directly from Openstreetmap
# -------------------------------------------------------------

import contextily as ctx
import numpy as np
from pyproj import Proj, transform

class geoplot:
    '''
    Downloads georeferenced images from OpenStreetMap.
    Can read input as either UTM or latlon.

    Mandatory:
    x:          vector for max/min boundary eastering
    y:          vector for max/min boundary northering

    Optional:
    ll:         False by default, set true if latlon input
    projection: Of the data you want to plot over georeference
                (UTM33 - epsg:32633 - by default)

    source:     choose source of background map.
                - hot (default, contextily ref: OpenStreetMap.HOT)
                - mapnik (OpenStreetMap.Mapnik)
                - voyager (CartoDB.Voyager)

    '''
    def __init__(self, x, y, 
                 ll=False, 
                 projection = 'epsg:32633',
                 source = 'hot', 
                 verbose = False):
        # Projections
        self.WGS84   = Proj('EPSG:4326')
        self.WebMerc = Proj('EPSG:3857')
        self.UTM     = Proj(projection)
        
        # initialize offset
        self.offx    = 0
        self.offy    = 0

        # Make sure you don't get a list
        if isinstance(x,list):
            x = np.array(x)
        if isinstance(y,list):
            y = np.array(y)

        # Project to web mercurator
        if ll == False:
            self.lon, self.lat = self.UTM(x, y, inverse=True)
        else:
            self.lat = y
            self.lon = x

        # Choose source, download images
        if source == 'mapnik':
            src = ctx.providers.OpenStreetMap.Mapnik

        elif source == 'voyager':
            src = ctx.providers.CartoDB.Voyager

        elif source == 'hot':
            src = ctx.providers.OpenStreetMap.HOT

        xe,ns  = self.project_to_map(self.lon,self.lat)

        if verbose:
            print('Downloading images...')
        
        try:
            img, extent = ctx.bounds2img(s=min(ns)[0], n=max(ns)[0],
                                         e=max(xe)[0], w=min(xe)[0], source=src)
        except:
            img, extent = ctx.bounds2img(s=min(ns), n=max(ns),
                                         e=max(xe), w=min(xe), source=src)

        if verbose:
            print('Warp images to map reference')

        self.img, self.extent = ctx.warp_tiles(img, extent, projection) 
        self.extent = np.array(self.extent)

    def project_to_map(self,lon,lat):
        xe,ns      = transform(self.WGS84, self.WebMerc, lon, lat, always_xy = True)
        return xe,ns

    def project_to_utm(self, lon, lat):
        '''
        from wgs84 to utm
        '''
        x, y = self.UTM(lon, lat, inverse=False)
        return x,y