# ----------------------------------------------------------------------------------
#              Dump data from a ROMS run to a FVCOM restart file
# ----------------------------------------------------------------------------------
import os
import pyproj
import fvtools.nesting.roms_nesting_fg as rn
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar as pb
import fvtools.nesting.vertical_interpolation as vi
from netCDF4 import Dataset
from datetime import datetime, timedelta
from fvtools.grid.fvcom_grd import FVCOM_grid
from time import gmtime, strftime
from scipy.spatial import cKDTree as KDTree

# ---
# Basically just the same as any other nestingfile.
# ---

def main(restartfile, mother, uv=False, smooth = None, avg = True):
    '''
    restartfile  - restart file formatted for FVCOM
    mother       - 'NK' for NorKyst-800 or 'NS' for NorShelf-2.4 km
    avg          - True if using average files
    '''
    # FVCOM grid object
    # ----
    M                 = FVCOM_grid(restartfile)
    restart_time      = netCDF4.num2date(Dataset(restartfile)['time'][0], \
                                         units = Dataset(restartfile)['time'].units)

    # Find the ROMS file and index to restart from
    # ----
    if mother == 'NS':
        start, stop       = prepare_search(restart_time)
        time, path, index = rn.make_fileList(start, stop, mother, avg)
        ind               = np.where(time==Dataset(restartfile)['time'][0])
        if ind[0].size == 0:
            raise OSError("Remember that the average files are published for 12:00 o'clock")
    
    else:
        # For NorKyst, we already know which file to read
        try:
            time_string       = "{0.year}-{0.month:02}-{0.day:02}".format(restart_time)   
            time, path, index = rn.make_fileList(time_string, time_string, mother)
            ind               = np.where(time==Dataset(restartfile)['time'][0])
            
            if ind[0].size == 0:
                raise OSError()

        except OSError:
            new_time_string   = "{0.year}-{0.month:02}-{0.day:02}".format(restart_time-timedelta(1))   
            time, path, index = rn.make_fileList(new_time_string, new_time_string, mother)
            ind               = np.where(time==Dataset(restartfile)['time'][0])
                
    # Trash the data we don't need
    # ----
    path              = path[ind[0][0]]
    index             = index[ind[0][0]]

    # ROMS grid object
    # ----
    ROMS_grd          = ROMS_grid(path)

    # Interpolation coefficients
    # ----
    print('\nFind nearest neighbors\n -------------------')
    N1                = nearest_neighbor(M, ROMS_grd, uv)

    print('\nAdd vertical interpolation coefficients\n -------------------')
    if uv:
        coords = ['rho','u','v']
    else:
        coords = ['rho']
    N1                = vi.add_vertical_interpolation2N1(N1, coords = coords)

    # Intepolate data and dump to the restartfile
    # ----
    print('\nInterpolate and store the ROMS data to the FVCOM restart file\n -------------------')
    interpolate_to_restart(restartfile, path, index, N1, uv)

    if smooth is not None:
        print('\nSmooth the initial field')
        

    print('Fin.')
    
def interpolate_to_restart(restartfile, path, index, N1, uv):
    '''
    Read data from the ROMS file and dump it to the FVCOM file
    '''
    # Get ROMS data
    roms_nc         = Dataset(path)
    ROMS_out        = get_roms_data(roms_nc, index, N1, uv)

    # Dump the grid angle to the grid file
    if uv:
        angle = N1.fvcom_angle # Angle to rotate u, v 
        angle = (angle[N1.M.tri[:, 0]-1] + angle[N1.M.tri[:, 1]-1] + angle[N1.M.tri[:, 2]-1]) / 3
        angle = angle 

    # Interpolate horizontally and vertically to the FVCOM grid
    print('... horizontal interpolation')
    ROMS_horizontal = horizontal_interpolation(ROMS_out, N1, uv)

    print('... vertical interpolation')
    FVCOM_in        = vertical_interpolation(ROMS_horizontal, N1, uv)

    # Dump to the restart file
    print('\nDump data to the restart file')
    restart         = Dataset(restartfile, 'r+')

    restart.variables['zeta'][0,:]       = FVCOM_in.zeta
    restart.variables['temp'][0,:,:]     = FVCOM_in.temp
    restart.variables['salinity'][0,:,:] = FVCOM_in.salt
    if uv:
        restart.variables['u'][0, :, :]      = FVCOM_in.u*np.cos(angle) - FVCOM_in.v*np.sin(angle)
        restart.variables['v'][0, :, :]      = FVCOM_in.u*np.sin(angle) + FVCOM_in.v*np.cos(angle)
        ubar, vbar = calc_uv_bar(restart.variables['u'][0, :, :],restart.variables['v'][0, :, :],N1.M)
        restart.variables['ua'][0,:]         = ubar
        restart.variables['va'][0,:]         = vbar

    restart.close()

def nearest_neighbor(M, ROMS_grd, uv):
    '''
    Nearest neighbor interpolation
    '''
    # Initiate the nearest ROMS object
    # ----
    N1 = NEAREST_ROMS(M, ROMS_grd)

    # Create a mask holding only ROMS point covering the FVCOM domain
    # ----
    N1.fv_rho_mask = ROMS_grd.crop_rho(xlim=[M.x.min() - 5000, M.x.max() + 5000],
                                       ylim=[M.y.min() - 5000, M.y.max() + 5000])

    if uv:
        N1.fv_u_mask   = ROMS_grd.crop_u(xlim=[M.x.min() - 5000, M.x.max() + 5000],
                                         ylim=[M.y.min() - 5000, M.y.max() + 5000])
        
        N1.fv_v_mask   = ROMS_grd.crop_v(xlim=[M.x.min() - 5000, M.x.max() + 5000],
                                         ylim=[M.y.min() - 5000, M.y.max() + 5000])    


    # create a cropped version of the mask
    # --------------------------------------------------------------------------------------------
    rho_i, rho_j = np.where(N1.fv_rho_mask)
    N1.m_ri  = min(rho_i); N1.x_ri = max(rho_i)   
    N1.m_rj  = min(rho_j); N1.x_rj = max(rho_j) 

    if uv:
        u_i, u_j = np.where(N1.fv_u_mask)
        N1.m_ui  = min(u_i); N1.x_ui = max(u_i)
        N1.m_uj  = min(u_j); N1.x_uj = max(u_j)

        v_i, v_j = np.where(N1.fv_v_mask)
        N1.m_vi  = min(v_i); N1.x_vi = max(v_i)
        N1.m_vj  = min(v_j); N1.x_vj = max(v_j)

    # The mask (to be used later on)
    N1.cropped_rho_mask = N1.fv_rho_mask[N1.m_ri:N1.x_ri+1, N1.m_rj:N1.x_rj+1]

    if uv:
        N1.cropped_u_mask   = N1.fv_u_mask[N1.m_ui:N1.x_ui+1, N1.m_uj:N1.x_uj+1]
        N1.cropped_v_mask   = N1.fv_v_mask[N1.m_vi:N1.x_vi+1, N1.m_vj:N1.x_vj+1]

    # Cropping the ROMS land mask
    # --------------------------------------------------------------------------------------------
    N1.Land_rho         = ROMS_grd.rho_mask[N1.fv_rho_mask]

    if uv:
        N1.Land_u           = ROMS_grd.u_mask[N1.fv_u_mask]
        N1.Land_v           = ROMS_grd.v_mask[N1.fv_v_mask]

    # Cropping the coordinates
    # --------------------------------------------------------------------------------------------
    x_rho          = ROMS_grd.x_rho[N1.fv_rho_mask][N1.Land_rho==0]
    y_rho          = ROMS_grd.y_rho[N1.fv_rho_mask][N1.Land_rho==0]

    if uv:
        x_u            = ROMS_grd.x_u[N1.fv_u_mask][N1.Land_u==0]
        y_u            = ROMS_grd.y_u[N1.fv_u_mask][N1.Land_u==0]

        x_v            = ROMS_grd.x_v[N1.fv_v_mask][N1.Land_v==0]
        y_v            = ROMS_grd.y_v[N1.fv_v_mask][N1.Land_v==0]

    # Build the KDTrees, find the nearest rho, u and v points
    # --------------------------------------------------------------------------------------------
    # Create position vectors
    # ----
    rho_points     = np.array([x_rho,y_rho]).transpose()

    if uv:
        u_points       = np.array([x_u,y_u]).transpose()
        v_points       = np.array([x_v,y_v]).transpose()

    fv_nodes       = np.array([M.x,M.y]).transpose()

    if uv:
        fv_cells       = np.array([M.xc, M.yc]).transpose()

    # Build the tree
    # ----
    print('Build the KDTree')
    rho_tree       = KDTree(rho_points)

    if uv:
        u_tree         = KDTree(u_points)
        v_tree         = KDTree(v_points)

    # Find the nearest fvcom nodes, store the indices
    # ----
    print('\nFind the nearest ocean points to each FVCOM point')
    print(' ... for rho')
    p, fv2rho      = rho_tree.query(fv_nodes)
    if uv:
        print(' ... for u')
        p, fv2u        = u_tree.query(fv_cells)
        print(' ... for v')
        p, fv2v        = v_tree.query(fv_cells)

    N1.rho_index   = fv2rho

    if uv:
        N1.u_index     = fv2u
        N1.v_index     = fv2v

    # Store stuff needed by other routines
    N1.M           = M
    N1.NEST        = M
    N1.ROMS_grd    = ROMS_grd
    N1.ROMS_depth(uv)
    if uv:
        N1.ROMS_angle()
    return N1

# Classes
# -----------------------------------------------------------------------
class NEAREST_ROMS():
    '''
    Object with indices for nearest neighbor interpolation from ROMS to FVCOM

    All grid details needed for the nesting should be found here.
    '''
    def __init__(self, M, ROMS_grd):
        self.rho_index        = np.empty([len(M.x)])

    def ROMS_depth(self, uv):
        self.fvcom_rho_dpt  = self.ROMS_grd.h_rho[self.fv_rho_mask][self.Land_rho==0][self.rho_index]
        if uv:
            self.fvcom_u_dpt    = self.ROMS_grd.h_u[self.fv_u_mask][self.Land_u==0][self.u_index]
            self.fvcom_v_dpt    = self.ROMS_grd.h_v[self.fv_v_mask][self.Land_v==0][self.v_index]

    def ROMS_angle(self):
        '''
        Calculate ROMS angle at FVCOM nodes.
        '''
        self.fvcom_angle    = self.ROMS_grd.angle[self.fv_rho_mask][self.rho_index]

class N4ROMS():
    '''
    Object with indices and coefficients for ROMS to FVCOM interpolation.

    All grid details needed for the nesting should be found here.
    '''
    def __init__(self, M, ROMS_grd):
        '''
        Initialize empty attributes
        '''
        self.rho_coef         = np.empty([len(M.x),  4])
        self.rho_index        = np.empty([len(M.x),  4])

        self.u_coef           = np.empty([len(M.xc),  4])
        self.u_index          = np.empty([len(M.xc),  4])

        self.v_coef           = np.empty([len(M.xc),  4])
        self.v_index          = np.empty([len(M.xc),  4])

    def save(self, name="Nearest4"):
        '''
        Save object to file.
        '''
        pickle.dump(self, open( name + ".p", "wb" ) )


# For ROMS mother grid stuff
# ----
class ROMS_grid():
    '''
    Object containing grid information about ROMS coastal ocean  model grid.
    '''
    def __init__(self, pathToROMS):
        """
        Read grid coordinates from nc-files.
        """
        print('\nROMS grid')
        self.nc      = pathToROMS
        self.name    = pathToROMS.split('/')[-1].split('.')[0]

        # Open the thredds file 
        ncdata       = Dataset(pathToROMS, 'r')
        
        # Open the file containing vertical grid information
        if 'norshelf' in pathToROMS:
            ncvert_path  = 'https://thredds.met.no/thredds/dodsC/sea_norshelf_files/norshelf_his_an_20190531T00Z.nc'
            ncvert       = Dataset(ncvert_path, 'r')

        else:
            ncvert       = Dataset(pathToROMS, 'r')

        # Verical grid info
        # ----
        self.Cs_r    = ncvert.variables.get('Cs_r')[:]
        self.angle   = ncdata.variables.get('angle')[:]
        self.h_rho   = ncdata.variables.get('h')[:]
        self.z_rho   = self.h_rho[:, :, None] * self.Cs_r

        # u and v variants
        # ----
        self.h_u     = (self.h_rho[:,1:]+self.h_rho[:,:-1])/2
        self.z_u     = (self.z_rho[:,1:,:]+self.z_rho[:,:-1,:])/2

        self.h_v     = (self.h_rho[1:,:]+self.h_rho[:-1,:])/2        
        self.z_v     = (self.z_rho[1:,:,:]+self.z_rho[:-1,:,:])/2

        # temp, salt and zeta
        # ----
        self.lon_rho = ncdata.variables.get('lon_rho')[:]
        self.lat_rho = ncdata.variables.get('lat_rho')[:]  

        self.lon_u   = ncdata.variables.get('lon_u')[:]
        self.lat_u   = ncdata.variables.get('lat_u')[:]  

        self.lon_v   = ncdata.variables.get('lon_v')[:]
        self.lat_v   = ncdata.variables.get('lat_v')[:]  

        # ROMS fractional landmask (from pp file). Let "1" indicate ocean.
        # ----
        self.rho_mask = ((ncdata.variables.get('mask_rho')[:]-1)*(-1)).astype(bool)
        self.u_mask = ((ncdata.variables.get('mask_u')[:]-1)*(-1)).astype(bool)
        self.v_mask = ((ncdata.variables.get('mask_v')[:]-1)*(-1)).astype(bool)
        ncdata.close()

        # Project to UTM33 coordinates
        # ----
        UTM33W       = pyproj.Proj(proj='utm', zone='33', ellps='WGS84')
        self.x_rho, self.y_rho = UTM33W(self.lon_rho, self.lat_rho, inverse=False)
        self.x_u,   self.y_u   = UTM33W(self.lon_u,   self.lat_u,   inverse=False)
        self.x_v,   self.y_v   = UTM33W(self.lon_v,   self.lat_v,   inverse=False)

    def crop_rho(self, xlim, ylim):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_rho >= xlim[0], self.x_rho <= xlim[1])
        ind2 = np.logical_and(self.y_rho >= ylim[0], self.y_rho <= ylim[1])
        return np.logical_and(ind1, ind2)

    def crop_v(self, xlim, ylim):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_v >= xlim[0], self.x_v <= xlim[1])
        ind2 = np.logical_and(self.y_v >= ylim[0], self.y_v <= ylim[1])
        return np.logical_and(ind1, ind2)

    def crop_u(self, xlim, ylim):
        """
        Find indices of grid points inside specified domain
        """
        ind1 = np.logical_and(self.x_u >= xlim[0], self.x_u <= xlim[1])
        ind2 = np.logical_and(self.y_u >= ylim[0], self.y_u <= ylim[1])
        return np.logical_and(ind1, ind2)

# Interpolation stuff
# -------------------------------------------------------------------------------------------
def prepare_search(restart_date):
    '''
    Search over a week to make sure that you will find the correct restart file
    '''
    start = restart_date-timedelta(days=3)

    start_string = "{0.year}-{0.month:02}-{0.day:02}".format(start)
    stop_string  = "{0.year}-{0.month:02}-{0.day:02}".format(restart_date)

    return start_string, stop_string


# Fetch data from the ROMS output and dump to the FVCOM grid
# ----------------------------------------------------------------------------------------
def get_roms_data(d, index, N1, uv):
    '''
    Dumps roms data from the netcdf file and prepares for interpolation
    '''

    # Initialize storage
    class dumped: pass

    # Selective load (load the domain within the given limits)
    # ----
    dumped.salt = d.variables.get('salt')[index, :, N1.m_ri:(N1.x_ri+1), N1.m_rj:(N1.x_rj+1)][:,N1.cropped_rho_mask][:,N1.Land_rho==0].transpose()
    dumped.temp = d.variables.get('temp')[index, :, N1.m_ri:(N1.x_ri+1), N1.m_rj:(N1.x_rj+1)][:,N1.cropped_rho_mask][:,N1.Land_rho==0].transpose()
    dumped.zeta = d.variables.get('zeta')[index,    N1.m_ri:(N1.x_ri+1), N1.m_rj:(N1.x_rj+1)][N1.cropped_rho_mask][N1.Land_rho==0]
    if uv:
        dumped.u    = d.variables.get('u')[index,    :, N1.m_ui:(N1.x_ui+1), N1.m_uj:(N1.x_uj+1)][:,N1.cropped_u_mask][:,N1.Land_u==0].transpose()
        dumped.v    = d.variables.get('v')[index,    :, N1.m_vi:(N1.x_vi+1), N1.m_vj:(N1.x_vj+1)][:,N1.cropped_v_mask][:,N1.Land_v==0].transpose()

    return dumped

def horizontal_interpolation(ROMS_out, N1, uv):
    '''
    Simple nearest neighbor interpolation algorithm.
    '''
    class HORZfield: pass
    HORZfield.zeta     = ROMS_out.zeta[N1.rho_index]
    HORZfield.temp     = ROMS_out.temp[N1.rho_index,:]
    HORZfield.salt     = ROMS_out.salt[N1.rho_index,:]
    if uv:
        HORZfield.u        = ROMS_out.u[N1.u_index,:]
        HORZfield.v        = ROMS_out.v[N1.v_index,:]
    return HORZfield

def vertical_interpolation(ROMS_data, N1, uv):
    '''
    Linear vertical interpolation of ROMS data to FVCOM-depths.
    '''
    class Data2FVCOM():
        pass

    Data2FVCOM.zeta = ROMS_data.zeta

    salt = np.flip(ROMS_data.salt, axis=1).T
    Data2FVCOM.salt = salt[N1.vi_ind1_rho, range(0, salt.shape[1])] * N1.vi_weigths1_rho + \
                      salt[N1.vi_ind2_rho, range(0, salt.shape[1])] * N1.vi_weigths2_rho 
     

    temp = np.flip(ROMS_data.temp, axis=1).T
    Data2FVCOM.temp = temp[N1.vi_ind1_rho, range(0, temp.shape[1])] * N1.vi_weigths1_rho + \
                      temp[N1.vi_ind2_rho, range(0, temp.shape[1])] * N1.vi_weigths2_rho 

    if uv:
        u = np.flip(ROMS_data.u, axis=1).T
        Data2FVCOM.u    = u[N1.vi_ind1_u, range(0, u.shape[1])] * N1.vi_weigths1_u + \
                          u[N1.vi_ind2_u, range(0, u.shape[1])] * N1.vi_weigths2_u 
 

        v = np.flip(ROMS_data.v, axis=1).T
        Data2FVCOM.v    = v[N1.vi_ind1_v, range(0, v.shape[1])] * N1.vi_weigths1_v + \
                          v[N1.vi_ind2_v, range(0, v.shape[1])] * N1.vi_weigths2_v 
    
    return Data2FVCOM

def calc_uv_bar(u,v,M):
    '''
    Calculate vertical average of velocity (ubar, vbar).
    '''
    h_uv    = np.squeeze(M.h_uv)
    siglevz = M.siglev_c.T * h_uv
    dz      = np.abs(np.diff(siglevz, axis=0))
    ubar    = np.sum(u*dz, axis=0) / h_uv
    vbar    = np.sum(v*dz, axis=0) / h_uv

    return ubar, vbar
