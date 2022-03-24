import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from netCDF4 import Dataset
from IPython.core.debugger import Tracer
from pyproj import Proj

class NEST_grid():
    'Represents NEST grid (FVCOM nesting zone)'

    def __init__(self, pathToFile=os.path.join(os.getcwd(), 'ngrd.mat')):
        '''Create FVCOM-grid object'''
        self.mfile=pathToFile
        self.add_grid_parameters(['xn', 'yn', 'lonn', 'latn', 'xc', 'yc','nid','cid'])

        self.add_grid_parameters(['nv',
                                  'xn', 
                                  'yn', 
                                  'lonn', 
                                  'latn', 
                                  'xc', 
                                  'yc',
                                  'lonc',
                                  'latc',
                                  'nid',
                                  'cid',
                                  'h',
                                  'h_center',
                                  'siglev',
                                  'siglev_center',
                                  'siglay',
                                  'siglay_center',])

    def add_grid_parameters(self, names):
        '''Read grid attributes from mfile and add them to FVCOM_grid object'''
        grid_mfile = loadmat(self.mfile)
        
        if type(names) is str:
            names=[names]
       
        for name in names:
            setattr(self, name, grid_mfile['ngrd'][0,0][name])
    
    def get_kb(self, sigmafile=None):
        '''Reads kb from the input/casename_sigma.dat file'''
        if sigmafile is None:
            casestr = get_cstr()
            sigmafile = os.path.join('input', casestr + '_sigma.dat') 
        
        
        fid = open(sigmafile, 'r')
        for line in fid:
            if 'NUMBER OF SIGMA LEVELS' in line:
                self.kb = int(line.split('=')[-1].rstrip('\r\n'))
        
        fid.close()

        return self.kb



class NorShelf_grd():
    '''ROMS-grid from met.no's NorShelf simulation'''

    def __init__(self, pathToFile='http://thredds.met.no/thredds/dodsC/sea_norshelf_his_agg'):
        '''Read grid information from met thredds-server.'''

        met_file = Dataset(pathToFile, 'r')

        self.pathToFile = pathToFile
        self.lon_rho = met_file.varibles['lon_rho'][:] 
        self.lat_rho = met_file.varibles['lat_rho'][:]
        self.lon_u = met_file.varibles['lon_u'][:]              
        self.lat_u = met_file.varibles['lat_u'][:]
        self.lon_v = met_file.varibles['lon_v'][:]              
        self.lat_v = met_file.varibles['lat_v'][:]








