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
import matplotlib.pyplot as plt

from pyproj import Proj, transform
from scipy.interpolate import LinearNDInterpolator
from matplotlib.tri import Triangulation
from datetime import datetime
from numba import jit, njit
from fvtools.grid.fvcom_grd import GridLoader, InputCoordinates, Coordinates, OBC, PlotFVCOM, CropGrid, ExportGrid, CellTriangulation, PropertiesFromTGE, GridProximity, CoastLine, AnglesAndPhysics
from functools import cached_property
from fvtools.prepro._buildcase.depth import DepthHandler
from fvtools.prepro._buildcase.mesh_validation import validate_boundary_triangles
from fvtools.prepro._buildcase.plotting import BathymetryPlotter

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
    if sigma_file is None: 
        sigma_file = f'./input/{casename}_sigma.dat'
    
    # Copy the input to a dictionary
    vars = dir()
    kwargs = {}
    for variable in vars:
        kwargs[variable] = eval(variable)

    bc = BuildCase(**kwargs)
    return bc

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
                 sigma_file = None,
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
        self.verbose = False # defined since FVCOM_grid expects this one, set to False and we don't want to be overwhelmed with messages
        if latlon:
            print(f'\nBuilding case: {casename} from: {dmfile} with bathymetry from: {depth_file.split("/")[-1]} in spherical coordinates')
        else:
            print(f'\nBuilding case: {casename} from: {dmfile} with bathymetry from: {depth_file.split("/")[-1]} in carthesian coordinates')

        # Overwrite
        if sigma_file is None: sigma_file = f'./input/{casename}_sigma.dat'

        # read file, update input fields
        self._add_grid_parameters_2dm()
        self.casename = casename
        self.Proj = self._get_proj()
        self._project_xy()
        self.dm_projection, self.target_projection, self.depth_projection = dm_projection, target_projection, depth_projection
        self.min_depth, self.rx0max, self.SmoothFactor = min_depth, rx0max, SmoothFactor
        self.sponge_factor, self.sponge_radius = sponge_factor, sponge_radius
        print(self)

        updated_grid = validate_boundary_triangles(self)

        # Re-project so that xy has same coordinates as it will in operational mode
        self.re_project(self.target_projection)
        if not updated_grid:
            self.main()
        else:
            print('- The grid was changed, you may want to inspect it before running .main(). Writing changed grid to auto_cleaned_grid.2dm')

    def __str__(self):
        str_grid = f'''
---
Grid:
    # nodes:            {len(self.x)}
    # triangles:        {len(self.xc)}
    minimum angle:      {self.grid_angles.min():.2f}
    minimum resolution: {self.grid_res.min():.2f} m
    # triangles <35 degrees: {len(np.where(self.grid_angles.min(axis=1)<35)[0])}
'''
        str_end = f'''
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
        try: 
            str_sigma = f'''
Vertical layers
    # sigma layers:     {self.siglay.shape[1]}
    # sigma levels:     {self.siglev.shape[1]}
'''
            str_out = str_grid + str_sigma + str_end
        except:
            str_out = str_grid + str_end
        return str_out

    def main(self):
        '''
        Interpolate bathymetry to mesh, smooth it, write .dat input files
        '''
        print('- Compute nbsn, ntsn')
        self.setup_metrics()
        
        print('- Interpolate depth to mesh')
        self.make_depth()
        self.h = np.copy(self.h_raw)
        
        print('- Smooth topo')
        self.smooth_topo()
        self.show_depth()

        print('- Read sigma')
        self.read_sigma(self.sigmafile)
        
        print('- Compute CFL criteria')
        self.get_cfl()
        self.add_dict_info()

        print('- Write necessary .dat files')
        self.write_grd(latlon = self.latlon)
        self.write_obc()
        self.write_sponge()
        self.write_cor(latlon = self.latlon)

        print('- Dumping all setup info to M.npy')
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
        try:
            self.info['author']    = os.getlogin()
        except:
            self.info['author']    = 'compute node'
        try:
            self.info['directory'] = os.getcwd()
        except:
            self.info['directory'] = 'compute node'
            
        self.info['scipt version'] = version_number
        self.info['casename']   = self.casename
        self.info['reference']  = self.target_projection

    show_depth = BathymetryPlotter.show_depth

class RemapError(Exception):
    pass
