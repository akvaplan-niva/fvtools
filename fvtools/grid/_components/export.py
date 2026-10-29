"""FVCOM mesh export mixin; unchanged implementation."""

import os
import numpy as np
import geopandas as gpd
import shapely as shp
from pyproj import Proj, Transformer

class ExportGrid:
    '''
    Routines to export data in FVCOM_grid to other formats
    '''
    def get_xy(self, latlon):
        '''
        will return the positions we need to write to a file depending on wether you write for a spherical or carthesian FVCOM run 
        '''
        if latlon:
            return self.lon, self.lat
        else:
            return self.x, self.y

    def write_bath(self, filename = None, latlon = False):
        '''
        - Generates an ascii FVCOM 5.x format bathymetry file
        '''
        if filename is None: 
            filename = f'input/{self.casename}_dep.dat'

        x_grid, y_grid = self.get_xy(latlon)
        with open(filename, 'w') as f:
            f.write(f'Node Number = {self.node_number}\n')
            for x, y, h in zip(x_grid, y_grid, self.h):
                line = '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(h)+'\n'
                f.write(line)
        print(f'  - Wrote : {filename}')

    def write_grd(self, filename = None, latlon = False, store_z = False):
        '''
        - Generates an ascii FVCOM 5.x format grid file
        '''
        if filename is None: 
            filename = f'input/{self.casename}_grd.dat'

        x_grid, y_grid = self.get_xy(latlon)  
        if store_z:
            h = self.h
        else:
            h = np.zeros(x_grid.shape)
            
        with open(filename, 'w') as f:
            f.write(f'Node Number = {self.node_number}\n')
            f.write(f'Cell Number = {self.cell_number}\n')
            for i, (t1, t2, t3) in enumerate(self.tri):
                f.write(f'{i+1} {t1+1} {t2+1} {t3+1} {i}\n')

            for i, (x, y, z) in enumerate(zip(x_grid, y_grid, h)):
                line = str(i+1) +' '+ '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(z)+'\n'
                f.write(line)
        print(f'  - Wrote : {filename}')

    def write_sponge(self, filename = None):
        '''
        - Generates an ascii FVCOM 5.x formatted sponge file
        '''
        if filename is None: 
            filename = f'input/{self.casename}_spg.dat'

        with open(filename, 'w') as f:
            f.write(f'Sponge Node Number = {len(self.sponge_nodes)}\n')
            if any(self.sponge_nodes):
                for node in self.sponge_nodes:
                    line = str(node+1) + ' ' + '{0:.6f}'.format(self.sponge_radius) + ' ' + '{0:.6f}'.format(self.sponge_factor)+'\n'
                    f.write(line)
        print(f'  - Wrote : {filename}')

    def write_obc(self, filename = None):
        """Write FVCOM OBC nodes once each, even for circular SMS nodestrings."""
        if filename is None:
            filename = f'input/{self.casename}_obc.dat'

        # SMS represents closed nodestrings by repeating their first node.
        # FVCOM's OBC-node count must not include that closing duplicate.
        sequences = []
        for nodestring in self.nodestrings:
            nodes = np.asarray(nodestring, dtype=np.int64)
            if nodes.size > 1 and nodes[0] == nodes[-1]:
                nodes = nodes[:-1]
            sequences.append(nodes)

        with open(filename, 'w') as f:
            f.write(f'OBC Node Number = {sum(map(len, sequences))}\n')
            i = 0
            for nodes in sequences:
                for node in nodes:
                    i += 1
                    f.write(f'{i} {node+1} 1\n')
        print(f'  - Wrote : {filename}')

    def write_cor(self, filename = None, latlon = False):
        '''
        Generates an ascii FVCOM 5.x formatted coriolis file
        - filename: name of the output file
        - latlon:   True if the model is configured in spherical mode with WGS84 coordinates
        '''
        if filename is None: 
            filename = f'input/{self.casename}_cor.dat'

        if latlon:
            x, y = self.lon, self.lat
        else:
            x, y = self.x, self.y

        with open(filename, 'w') as f:
            f.write(f'Node Number = {self.node_number}\n')
            for x, y, lat in zip(x, y, self.lat):
                line = '{0:.6f}'.format(x) + ' ' + '{0:.6f}'.format(y) + ' ' + '{0:.6f}'.format(lat)+'\n'
                f.write(line)
        print(f'  - Wrote : {filename}')

    def write_coast(self, outfolder = None, buffer_width = 50000, verbose = False):
        '''
        Finds the coastline in your domain, writes to casename_coast.shp file
        ---
        - outfolder: Folder to store coast.shp
                     - default: current directory 
        - buffer:    Buffer side of polygon, basically the width of the "islands" we create for the exterior polygon
                     - must always be positive, buffer measured in meters
        '''
        # Move most of these functions to the CoastLine class
        if outfolder is None: 
            outfolder = os.getcwd()

        # Draw polygons based on the coastline
        islands = self.get_land_polygons()
        areas = [p.area for p in islands] # Find areas of the polygons to identify exterior polygon (max(areas) = exterior polygon)
        exterior = islands.pop(np.squeeze(np.where(np.array(areas) == max(areas)))) # Split islands and exterior polygon (e.g. remove exterior from islands)
        exterior_buffer = exterior.buffer(buffer_width, single_sided=True) # Exterior in one polygon

        # Cut the exterior polygon so that we can remove polygons connected to the OBC
        def construct_lines(line, polygon):
            '''Add segments of the inner and outer polygon to a list, prepare to polygonize'''
            x, y = polygon.exterior.xy
            for i in range(len(polygon.exterior.xy[0])-1):
                line.append(shp.geometry.LineString(np.array([[x[i], x[i+1]], [y[i], y[i+1]]]).T)) 

        lines = []
        construct_lines(lines, exterior_buffer)
        construct_lines(lines, exterior)

        # Add lines from OBC (in exterior) to the exterior_buffer polygon
        obc_points = []
        for nodestring in self.nodestrings:
            # Closed loops have no land-connected endpoints to extend into
            # the exterior buffer. Treat only open chains here.
            if len(nodestring) > 1 and nodestring[0] == nodestring[-1]:
                continue
            points = [nodestring[0], nodestring[-1]]
            ind = []
            for point in points:
                dst = np.sqrt((exterior_buffer.exterior.xy[0]-self.x[point])**2 + (exterior_buffer.exterior.xy[1]-self.y[point])**2)
                ind.append(np.where(dst==dst.min())[0][0])

            for point_ind, buffer_ind in zip(points, ind):
                lines.append(shp.geometry.LineString([[self.x[point_ind],   self.y[point_ind]],  
                                                      [np.array(exterior_buffer.exterior.xy[0])[buffer_ind], np.array(exterior_buffer.exterior.xy[1])[buffer_ind]], 
                                                      ]))

            # Make points on the obc, these will be used to mask parts of the obc polygons connecting to ocean
            obc_points.append(shp.geometry.Point(self.x[nodestring[int(len(nodestring)/2)]], self.y[nodestring[int(len(nodestring)/2)]])) # A point on the line
       
        exterior_polys = list(shp.ops.polygonize(lines)) # Connect lines to polygons
        cropped_polys = [pol for pol in exterior_polys if not exterior.contains(pol.representative_point())] # Remove interior polygon
        cropped_polys = [pol for pol in cropped_polys if not any([pol.distance(p) == 0 for p in obc_points])] # Remove polygons at obc
        islands.extend(cropped_polys) # Add obc polygons to the list with islands (land is now also an island)

        # Re-project to latlon since this is what OpenDrift expects
        project = Transformer.from_proj(self.Proj, Proj(init='epsg:4326')) # We always want to project to latlon, hence hardcoded
        islands_latlon = [shp.ops.transform(project.transform, pol) for pol in islands]

        # Put into geopandas object, save to outfolder
        print(f'- Writing {outfolder}/{self.casename}_coast.shp and associated files')
        s = gpd.GeoDataFrame({'FID': range(len(islands_latlon)), 'geometry': islands_latlon})
        s.to_file(f'{outfolder}/{self.casename}_coast.shp')
        if verbose:
            all_islands = gpd.GeoDataFrame({'FID': range(len(islands)), 'geometry': islands})
            all_islands.plot()
            self.plot_grid()
            
    def write_2dm(self, name = None):
        '''
        Writes a 2dm file for this grid
        name: to be saved
        '''
        import fvtools.grid.fvgrid as fvgrid
        if name is None: name = self.casename
        if str(name).lower().endswith('.2dm'):
            name = str(name)[:-4]
        if any(self.obc_nodes):
            fvgrid.write_2dm(self.x, self.y, self.tri, nodestrings = self.nodestrings, name = name, casename = name)
        else:
            fvgrid.write_2dm(self.x, self.y, self.tri, name = name, casename = name)

    def write_section(self):
        '''
        write x_sec, y_sec to a casename_section.txt file
        '''
        lonlat = np.vstack((self.Proj(self.x_sec, self.y_sec, inverse=True))).T
        np.savetxt(f'{self.casename}_section.txt', lonlat, delimiter = ' ')
        print(f'- saved section positions to {self.casename}_section.txt')

    def to_npy(self, filename = 'M.npy'):
        '''
        Write no M.npy file
        '''
        M = {}
        self.nv = self.tri
        for var in ['x', 'y', 'lon', 'lat', 'h', 'h_raw', 'nv', 'siglay', 'siglev', 'obc_nodes', 'nodestrings', 'reference', 'info', 'ts']:
            try:
                M[var] = getattr(self, var)
            except:
                pass
        np.save(filename, M)
        print(f'- Wrote {filename}')
