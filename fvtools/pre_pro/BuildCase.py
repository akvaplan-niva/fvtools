"""
--------------------------------------------------------------------------------------------------------
                            In development - Status: Testing (beta)
--------------------------------------------------------------------------------------------------------
BuildCase - Creates all the input files needed to initialize an FVCOM experiment
          - Works together with get_ngrd.py, obcgridding.py (obcgridding should be run before BC,
            get_ngrd should be run after)
          - new in v4: no longer writes read_obc_nodes (expect FVCOM_grid to do that if needbe)
--------------------------------------------------------------------------------------------------------
"""
global version_number
version_number = 4.0

import os
import numpy as np
import fvtools.grid.fvgrid as fvg
import matplotlib.pyplot as plt

from pyproj import Proj, transform
from scipy.interpolate import LinearNDInterpolator
from matplotlib.tri import Triangulation
from datetime import datetime
from numba import jit, njit
from fvtools.grid.fvcom_grd import GridLoader, InputCoordinates, Coordinates, OBC, PlotFVCOM, CropGrid, ExportGrid, CellTriangulation, PropertiesFromTGE, GridProximity, CoastLine, AnglesAndPhysics
from fvtools.grid.tge import get_NBE
from functools import cached_property

def main(dmfile = None,
         depth_file = None,
         casename  = os.getcwd().split('/')[-1],
         dm_projection = 'epsg:32633', 
         depth_projection = 'epsg:32633', 
         target_projection = 'epsg:32633',
         sigma_file = None,
         rx0max = 0.2,
         SmoothFactor = 0.2,
         min_depth = 5.0,
         sponge_radius = 0,
         sponge_factor = 0,
         latlon = False):
    '''
    Quality controls the mesh, interpolates the bathymetry to the mesh and smooths it to a desired rx0 value

    Provide:
    ----
    - Path to a 2dm file
    - Path to a file with depth data for the entire domain
    - projections
        - for the source grid file (2dm)
        - for the source depth file
        - for the target reference system
            - latlon: True if you will run this model in spherical mode
    - Sigma file (will automatically look for one in your input directory)

    Other settings (default):
    ----
    rx0max:        (0.2)    smoothing target for bathymetry
    min_depth:     (5.0 m)  minimum depth in domain (to prevent drying)
    sponge_radius: (0 m)    distance from OBC we add diffusion to damp waves
    sponge_factor: (0)      max diffusion in sponge zone (will not use sponge nodes if set equal to zero)
    SmoothFactor:  (0.2)    factor to weigh neighboring nodes during laplacian smoothing
    latlon:        (False)  write output as latlon
    '''
    if sigma_file is None: sigma_file = f'./input/{casename}_sigma.dat'
    bc = BuildCase(dmfile=dmfile, depth_file=depth_file, casename=casename,
                   dm_projection=dm_projection, depth_projection=depth_projection, target_projection=target_projection, 
                   sigma_file=sigma_file,
                   rx0max=rx0max, SmoothFactor=SmoothFactor, min_depth=min_depth, 
                   sponge_radius=sponge_radius, sponge_factor=sponge_factor, latlon=latlon)
    return bc

class DepthHandler:
    '''
    Loads data, interpolates it to the grid and smooths it to some rx0 value
    '''
    @property
    def xy_grid_at_depth_points(self):
        '''Projects mesh to depth-file coordinates'''
        if self.target_projection != self.depth_projection:
            return np.array([*self.Proj_depth(self.lon, self.lat)]).T
        else:
            return np.array([self.x, self.y]).T

    @property
    def x_d(self):
        '''mesh x-position in depth-file coordinates'''
        return self.xy_grid_at_depth_points[:,0]
    
    @property
    def y_d(self):
        '''mesh y-position in depth-file coordinates'''
        return self.xy_grid_at_depth_points[:,1]

    @property
    def h_raw(self):
        '''bathymetry directly interpolated from the source'''
        return self._h_raw
    
    @h_raw.setter
    def h_raw(self, var):
        self._h_raw = var    

    def make_depth(self):
        """Interpolate data from a "depth.txt" file to a FVCOM mesh, adds 'h_raw' to the mesh"""
        if type(self.depth_file) == int or type(self.depth_file) == float:
            h_raw = dptfile*np.ones((self.x.shape)) # in case you need flat bathymetry
        else:
            if self.depth_file.split('.')[-1] == 'npy':
                depth = self.load_numpybath(self.depth_file)
            else:
                depth = self.load_textbath(self.depth_file)

            # Crop the data so that we only have data covering the FVCOM domain
            depth = self.crop(depth)
            h_raw = self.interpolate_to_mesh(depth)

        if np.argwhere(np.isnan(h_raw)).size > 0:
            print(f'Nan found in h_raw. Nans are set to {self.min_depth} m. '+\
                  'Consider using a better bathymetry that has good coverage of the entire domain.'+\
                  'Number of nans found: {np.argwhere(np.isnan(h_raw)).size}')
            h_raw[np.argwhere(np.isnan(h_raw))] = self.min_depth
        self.h_raw = h_raw

    @staticmethod
    def load_numpybath(dptfile):
        """Load the bathymetry from a big numpy file"""
        depth_data = np.load(dptfile, allow_pickle = True)
        return depth_data

    @staticmethod
    def load_textbath(dptfile):
        """
        Load the depth file
        - .txt files are "raw", ie. has not been processed by any BuildCase runs before. (slow to load)
        - .npy files are already in a BuildCase friendly format, and don't take too long to load
        """
        # Load raw data
        print(f'-> Load: {dptfile}')
        try:
            depth_data = np.loadtxt(dptfile, delimiter=',')
        except:
          try:
              depth_data = np.loadtxt(dptfile)
          except:
            try:
                depth_data = np.loadtxt(dptfile, delimiter=' ')
            except:
                depth_data = np.loadtxt(dptfile, skiprows = 1, delimiter = ',', usecols = [0,1,2])
        numpy_bath_name = f'{dptfile.split(".txt")[0]}.npy'
        print(f'- Storing the full bathymetry as: {numpy_bath_name}')
        np.save(numpy_bath_name, depth_data)
        return depth_data

    def read_sigma(self):
        '''Generate a tanh sigma coordinate distribution

        Parameters:
        ----
        sigmafile:   A casename_sigma.dat file

        Returns:
        ----
        lev, lay:    Sigma coordinate lev and lay (valid for the entire domain)
        '''
        # Read the input parameters from the casename_sigma.dat file
        data = np.loadtxt(self.sigmafile, delimiter = '=', dtype = str)
        if data[1,1] == ' TANH ':
            nlev = int(data[0,1])
            du   = float(data[2,1])
            dl   = float(data[3,1])
            lev  = np.zeros((nlev))
            for k in np.arange(nlev-1):
                x1 = np.tanh((dl + du) * (nlev - 2 - k) / (nlev - 1) - dl)
                x2 = np.tanh(dl)
                x3 = x2 + np.tanh(du)
                lev[k+1] = (x1 + x2) / x3 - 1.0

        elif data[1,1] == ' UNIFORM' or data[1,1] == 'UNIFORM':
            nlev = int(data[0,1])
            lev = np.zeros(nlev)
            for k in np.arange(1,nlev+1):
                lev[k-1] = -((k - 1)/(nlev - 1))

        elif data[1,1] == ' GEOMETRIC':
            nlev = int(data[0,1])
            lev = np.zeros(nlev)
            p_sigma = np.double(data[2,1])
            for k in range(1, int(np.floor((nlev+1)/2)+1)):
                lev[k-1]=-((k-1)/((nlev+1)/2 - 1))**p_sigma/2
            for k in range(int(np.floor((nlev+1)/2))+1,nlev+1):
                lev[k-1]=((nlev-k)/((nlev+1)/2-1))**p_sigma/2-1

        else:
            raise ValueError(f'BuildCase supports tanh-, geometric- and uniform-coordinates at the moment. {data[1,1]} is invalid.')

        # Siglay
        lay = [(lev[k]+lev[k+1])/2 for k in range(len(lev)-1)]

        # Store
        self.siglev = np.tile(lev, (len(self.x), 1))
        self.siglay = np.tile(np.array(lay), (len(self.x), 1))

    def crop(self, depth_data):
        """
        Crop the depth file to your domain
        """
        if max(depth_data[:,0]) < max(self.x_d) or min(depth_data[:,0]) > min(self.x_d) or max(depth_data[:,1]) < max(self.y_d) or min(depth_data[:,1]) > min(self.y_d):
            plt.plot(depth_data[:,0], depth_data[:,1], 'r.', label = 'depth data')
            plt.triplot(self.x_d, self.y_d, self.tri, label = 'FVCOM ', zorder = 10)
            plt.legend()
            plt.show(block=False)
            raise ValueError('The bathymetry file does not cover the model domain!')

        print('  - Crop the bathymetry data')
        ind1 = np.logical_and(depth_data[:,0] >= min(self.x_d)-5000.0, depth_data[:,0] <= max(self.x_d)+5000.0)
        ind2 = np.logical_and(depth_data[:,1] >= min(self.y_d)-5000.0, depth_data[:,1] <= max(self.y_d)+5000.0)
        ind  = np.logical_and(ind1, ind2)

        depth = {}
        depth['x'] = depth_data[ind,0]
        depth['y'] = depth_data[ind,1]
        depth['h'] = depth_data[ind,2]
        return depth

    def interpolate_to_mesh(self, depth):
        """
        Interpolate data from the unstructured data array (data) to the unstructured mesh (M)
        """
        print('  - Build interpolator')
        point = np.array([depth['x'], depth['y']]).T
        interpolant = LinearNDInterpolator(point, depth['h'])

        print('  - Interpolate topography to nodes')
        h_raw = interpolant(self.x_d, self.y_d)

        i = np.where(h_raw[:] < self.min_depth)[0]
        if i.size:
            print(f'  - The depth at {len(i)} points was set equal to {self.min_depth} m')
        h_raw[i] = self.min_depth
        return h_raw

    def smooth_topo(self):
        """
        Smooth the topography where the smoothness is bad
        """
        # Initial pass to do the initial smoothing (reduce dataset noise)
        print('  - Reduce noise')
        print('    - Laplacian filter')
        self.h = self.laplacian_filter(self.x, self.y, self.h, self.nbsn, self.ntsn, self.SmoothFactor)

        print('  - Adjust slope')
        print('    - Mellor, Ezer and Oey scheme')
        i = 0
        while True:
            i += 1
            self.h, rx0_max, corrected = self.mellor_ezer_oey(self.h, self.ntsn, self.nbsn, rx0max = self.rx0max - 0.02)
            print(f'    {i}: Max rx0: {rx0_max:.2f} - Number of adjustments: {corrected}')
            if abs(rx0_max - self.rx0max) < self.rx0max*0.01 or rx0_max == 0:
                print('  - Bathymetry smoothed.')
                break

    @staticmethod
    @njit
    def laplacian_filter(x, y, h, nbsn, ntsn, SmoothFactor):
        """
        A simple lapacian filter to smooth the topography
        """
        h_smooth = np.copy(h)
        for node in range(len(x)):
            nodes  = nbsn[node,:ntsn[node]]
            smooth = np.mean(h[nodes])
            h_smooth[node] = (1-SmoothFactor)*h_smooth[node] + SmoothFactor*smooth
        return h_smooth

    @staticmethod
    @jit(nopython = True)
    def mellor_ezer_oey(raw_bath, ntsn, nbsn, rx0max = 0.2, nodes = None):
        """
        Parameters:
        -----------
        bathymetry, number of surrounding nodes, index of surrounding nodes, smoothing goal and
        what nodes to inspect

        Returns:
        -----------
        Smoothed bathymetry

        ===================================================================================
        Mellor, Ezer and Oey:
        The Pressure Gradient Conundrum of Sigma Coordinate Ocean Models
        J. Atmos. Oceanic Technol. (1994)
        https://doi.org/10.1175/1520-0426(1994)011%3C1126:TPGCOS%3E2.0.CO;2
        ===================================================================================
        """

        # initialize
        if nodes is None:
            nodes = np.arange(len(raw_bath))

        # Iterate over the mesh
        rmax, bath, corrected = 0, np.copy(raw_bath), 0
        for i in nodes:
            diff = bath[i] - bath[nbsn[i,:ntsn[i]]]
            nbsn_ind  = np.where(diff == np.max(diff))[0][0]
            ind  = nbsn[i,nbsn_ind]
            difference = diff[nbsn_ind]
            if difference < 0:
                continue

            ph = bath[i] + bath[ind]

            r  = diff[nbsn_ind]/ph
            if r > rx0max:
                corrected += 1
                rarray     = np.array((r,rmax))
                rmax       = np.max(rarray)
                delta      = 0.5*(bath[i]-bath[ind]-rx0max*(bath[i]+bath[ind]))
                bath[i]   -= delta
                bath[ind] += delta

        return bath, rmax, corrected

    def setup_metrics(self):
        """Setup metrics for secondary connectivity (nodes surrounding nodes)"""
        tri   = Triangulation(self.x, self.y, self.tri)
        ntsn  = np.zeros((len(self.x))).astype(int)
        nbsn  = -1*np.ones((len(self.x),12)).astype(int)
        edges = tri.edges
        self.nbsn, self.ntsn = self.calculate_nbsn_ntsn(edges, nbsn, ntsn)
        self.edges = edges

    @staticmethod
    @njit
    def calculate_nbsn_ntsn(edges, nbsn, ntsn):
        for i, edge in enumerate(edges):
            lmin = np.min(np.abs(nbsn[edge[0],:]-edge[1]))
            if lmin != 0:
                nbsn[edge[0], ntsn[edge[0]]] = edge[1]
                ntsn[edge[0]] += 1
            lmin = np.min(np.abs(nbsn[edge[1],:]-edge[0]))
            if lmin != 0:
                nbsn[edge[1], ntsn[edge[1]]] = edge[0]
                ntsn[edge[1]] += 1
        return nbsn, ntsn

class BuildCase(GridLoader, InputCoordinates, Coordinates, OBC, PlotFVCOM, CropGrid, GridProximity, CoastLine,
                ExportGrid, DepthHandler, CellTriangulation, PropertiesFromTGE, AnglesAndPhysics):
    '''
    Class to eat up the functionality in the rest of the script :)
    '''
    def __init__(self,
                 dmfile = None,
                 depth_file = None,
                 casename  = os.getcwd().split('/')[-1],
                 dm_projection = 'epsg:32633', 
                 depth_projection = 'epsg:32633', 
                 target_projection = 'epsg:32633',
                 sigma_file = f'./input/{os.getcwd().split("/")[-1]}_sigma.dat',
                 rx0max = 0.2,
                 SmoothFactor = 0.2,
                 min_depth = 5.0,
                 sponge_radius = 0,
                 sponge_factor = 0,
                 latlon = False):
        '''
        Input:
        ----
        - Path to a 2dm file
        - Path to a file with depth data for the entire domain
        - projections
            - for the source grid file (2dm)
            - for the source depth file
            - for the target reference system
                - latlon: True if you will run this model in spherical mode
        - Sigma file (will automatically look for one in your input directory)

        Other settings (default):
        ----
        rx0max:        (0.2)    smoothing target for bathymetry
        min_depth:     (5.0 m)  minimum depth in domain (to prevent drying)
        sponge_radius: (8000 m) distance from OBC we add diffusion to damp waves
        sponge_factor: (0.001)  max diffusion in sponge zone
        SmoothFactor:  (0.2)    factor to weigh neighboring nodes during laplacian smoothing
        latlon:        (False)  write output as latlon

        Output:
        ----
        BuildCase object, i.e.
        let,
           import fvtools.pre_pro.BuildCase as bc
           case = bc.main( ... )
        -> case is the BuildCase object

        if the mesh is invalid, BuildCase will pause to let you assess if the fixed mesh is ok for your purpose. If so call;
           case.main()

        and it will run and generate all necessary input files.
        '''
        # Some names are expected by GridLoader and Coordinates - such as filepath and reference
        self.latlon = latlon
        self.filepath, self.sigmafile, self.depth_file, self.reference = dmfile, sigma_file, depth_file, dm_projection
        self.verbose = False # since FVCOM_grid expects this one
        if latlon:
            print(f'\nBuilding case: {casename} from: {dmfile} with bathymetry from: {depth_file.split("/")[-1]} in spherical coordinates')
        else:
            print(f'\nBuilding case: {casename} from: {dmfile} with bathymetry from: {depth_file.split("/")[-1]} in carthesian coordinates')

        # read file, update input fields
        self._add_grid_parameters_2dm()
        self.casename = casename
        self.Proj = self._get_proj()
        self._project_xy()
        self.dm_projection, self.target_projection, self.depth_projection = dm_projection, target_projection, depth_projection
        self.min_depth, self.rx0max, self.SmoothFactor = min_depth, rx0max, SmoothFactor
        self.sponge_factor, self.sponge_radius = sponge_factor, sponge_radius
        print(self)

        print('- Check boundary triangles')
        cells = self._return_valid_triangles(self.x, self.y, self.tri)
        if len(cells) != len(self.xc):
            print(f'  - There are {len(self.xc)-len(cells)} illegal nodes in this mesh, removing those')
            plt.figure()
            self.plot_grid(label='input grid')

            # subgrid (returns a FVCOM_grid instance, we update necessary fields)
            # - a BuildCase class is not a FVCOM_grid, so the subgridding is a bit of a hack - to get the nodestrings, we must delete the _full_model_boundary
            new = self.subgrid(cells = cells)
            self.x, self.y, self.lon, self.lat, self.tri, self.obc_nodes = new.x, new.y, new.lon, new.lat, new.tri, new.obc_nodes
            for attr in ['full_model_boundary', 'nodestrings', 'coastline_nodes']:
                delattr(self, attr)

            self.plot_grid(c='r-', label='fixed grid')
            self.plot_obc()
            plt.legend()
            updated_grid = True
        else:
            updated_grid = False
            print('  - Did not find triangles with more than one side facing the boundary')

        # Re-project so that xy has same coordinates as it will in operational mode
        self.re_project(self.target_projection)
        if not updated_grid:
            self.main()
        else:
            print('- The grid was changed, you may want to inspect it before running BuildCase.main()')

    def __str__(self):
        return f'''
---
    Grid:
        number of nodes:     {len(self.x)}
        number of triangles: {len(self.xc)}
        minimum angle:       {self.grid_angles.min():.2f}
        minimum resolution:  {self.grid_res.min():.2f} m
        #triangles <35 degrees: {len(np.where(self.grid_angles.min(axis=1)<35)[0])}

    Settings:
        sponge radius: {self.sponge_radius}
        sponge factor: {self.sponge_factor}
        minimum depth: {self.min_depth}
        rx0 target:    {self.rx0max}
        laplacian smoothing factor: {self.SmoothFactor}

    Sources:
        2dm file:      {self.filepath}
         - projection: {self.dm_projection}
        depth file:    {self.depth_file}
         - projection: {self.depth_projection}

    Target projection: {self.target_projection}
----
'''

    def main(self):
        '''
        Interpolate bathymetry to mesh, smooth it, write .dat input files
        '''
        print('- Read sigma')
        self.read_sigma()
        print('- Compute nbsn, ntsn')
        self.setup_metrics()
        print('- Interpolate depth to mesh')
        self.make_depth()
        self.h = np.copy(self.h_raw)
        print('- Smooth topo')
        self.smooth_topo()
        self.show_depth()
        print('- Compute CFL criteria')
        self.get_cfl()
        self.add_dict_info()
        print('- Write .dat files')
        self.write_grd(latlon = self.latlon)
        self.write_obc()
        self.write_sponge()
        self.write_cor()
        self.to_npy()

    @cached_property
    def Proj_2dm(self):
        '''Proj method for the 2dm input file'''
        return Proj(self.dm_projection)

    @cached_property
    def Proj_target(self):
        '''Proj method for coordinates we will use in the model'''
        return Proj(self.target_projection)

    @cached_property
    def Proj_depth(self):
        '''Proj method for coordinates depth is provided in'''
        return Proj(self.depth_projection)

    @property
    def sponge_nodes(self):
        '''
        Sponges from the obc
        - In BuildCase, we assume that sponges are always applied near the OBC, but remember that they can also be put in the domain.
        '''
        if self.sponge_factor > 0:
            sponges = []
            for nodestring in self.nodestrings:
                sponges.extend(nodestring.tolist())
            return np.array(sponges)
        else:
            return np.empty(0)

    def add_dict_info(self):
        """Add some info so that one can track who made it using what input"""
        self.info = {}
        self.info['created']    = datetime.now().strftime('%Y-%m-%d at %H:%M h')
        self.info['2dm file']   = self.filepath
        self.info['depth file'] = self.depth_file
        self.info['author']     = os.getlogin()
        self.info['directory']  = os.getcwd()
        self.info['scipt version'] = version_number
        self.info['casename']   = self.casename
        self.info['reference']  = self.target_projection

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

class RemapError(Exception):
    pass
