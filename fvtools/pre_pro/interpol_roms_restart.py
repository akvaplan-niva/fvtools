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
from time import gmtime, strftime
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.roms_grid import get_roms_grid
from scipy.spatial import cKDTree as KDTree

# ---
# Basically just the same as any other nestingfile.
# ---

def main(restartfile, mother, uv=False, avg=False, proj='epsg:32633'):
    '''
    restartfile  - restart file formatted for FVCOM
    mother       - 'HI-NK' or 'MET-NK' for NorKyst-800, 'H-NS' for hourly- or 'D-NS' for daily averaged NorShelf 2.4km files
    avg          - True if using average files, default: False
    proj         - set projection, default: epsg:32633 (UTM33)
    '''
    # FVCOM grid object
    # ----
    print('Load FVCOM restart file, find restart time')
    M            = FVCOM_grid(restartfile, reference=proj)
    restart_time = netCDF4.num2date(Dataset(restartfile)['time'][0], \
                                    units = Dataset(restartfile)['time'].units)

    print('Find ROMS source file')
    ROMS = get_roms_grid(mother, M.Proj)

    # Fields we will interpolate to the restart file
    # ----
    if uv:
        coords = ['rho','u','v']
    else:
        coords = ['rho']

    # Find the ROMS output file- and index in that file we want to start from
    # ---- 
    ROMS, index, path = find_roms_to_start_from(ROMS, restart_time, restartfile)

    # ROMS load the grid that covers our FVCOM domain
    # ----
    ROMS.load_grid(M.x, M.y)

    # Interpolation coefficients
    # ----
    print('\nFind nearest neighbors\n -------------------')
    N1 = N1ROMS(M.x, M.y, M.xc, M.yc, ROMS)
    N1.NEST = M
    
    print('\nAdd vertical interpolation coefficients\n -------------------')
    N1 = vi.add_vertical_interpolation2N1(N1, coords = coords)

    # Intepolate data and dump to the restartfile
    # ----
    print('\nInterpolate and store ROMS hydrography to the FVCOM restart file\n -------------------')
    interpolate_to_restart(restartfile, path, index, N1, uv)

    print('Fin.')
    
def find_roms_to_start_from(ROMS, restart_time, restartfile):
    # Search for the relevant time and index over more than one file, in case we use a forecast- or a HI-NorKyst file
    # ----
    start, stop       = prepare_search(restart_time)
    time, path, index = rn.make_fileList(start, stop, ROMS)
    ind               = np.where(time==Dataset(restartfile)['time'][0])

    # Check if this index actually exists
    # ----
    if ind[0].size == 0:
        if 'NS' in mother:
            raise CouldNotFindRestartTime("Remember that the average files are published for 12:00 o'clock")
        else:
            raise CouldNotFindRestartTime("We were not able to find a suitable restart time")

    # The file and index of that file we need to read, is therefore
    # ----
    path  = path[ind[0][0]]
    index = index[ind[0][0]]

    return ROMS, index, path

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
    dumped.salt = d.variables.get('salt')[index, :, N1.m_ri:(N1.x_ri+1), N1.m_rj:(N1.x_rj+1)][:, N1.ROMS.cropped_rho_mask][:,N1.Land_rho==0].transpose()
    dumped.temp = d.variables.get('temp')[index, :, N1.m_ri:(N1.x_ri+1), N1.m_rj:(N1.x_rj+1)][:, N1.ROMS.cropped_rho_mask][:,N1.Land_rho==0].transpose()
    dumped.zeta = d.variables.get('zeta')[index,    N1.m_ri:(N1.x_ri+1), N1.m_rj:(N1.x_rj+1)][N1.ROMS.cropped_rho_mask][N1.Land_rho==0]
    if uv:
        dumped.u    = d.variables.get('u')[index, :, N1.m_ui:(N1.x_ui+1), N1.m_uj:(N1.x_uj+1)][:, N1.ROMS.cropped_u_mask][:, N1.Land_u==0].transpose()
        dumped.v    = d.variables.get('v')[index, :, N1.m_vi:(N1.x_vi+1), N1.m_vj:(N1.x_vj+1)][:, N1.ROMS.cropped_v_mask][:, N1.Land_v==0].transpose()

    return dumped


def interpolate_to_restart(restartfile, path, index, N1, uv):
    '''
    Read data from the ROMS file and dump it to the FVCOM file
    '''
    # Get ROMS data
    roms_nc         = Dataset(path)
    ROMS_out        = get_roms_data(roms_nc, index, N1, uv)

    # Dump the grid angle to the grid file
    if uv:
        # rotation factor
        angle = N1.fvcom_angle[N1.M.tri]

    # Interpolate horizontally and vertically to the FVCOM grid
    print('- horizontal interpolation')
    ROMS_horizontal = horizontal_interpolation(ROMS_out, N1, uv)

    print('- vertical interpolation')
    FVCOM_in        = vertical_interpolation(ROMS_horizontal, N1, uv)

    # Dump to the restart file
    print('\nDump to the restart file')
    restart         = Dataset(restartfile, 'r+')

    restart.variables['zeta'][0,:]       = FVCOM_in.zeta
    restart.variables['temp'][0,:,:]     = FVCOM_in.temp
    restart.variables['salinity'][0,:,:] = FVCOM_in.salt
    if uv:
        restart.variables['u'][0, :, :]      = FVCOM_in.u*np.cos(angle) - FVCOM_in.v*np.sin(angle)
        restart.variables['v'][0, :, :]      = FVCOM_in.u*np.sin(angle) + FVCOM_in.v*np.cos(angle)
        ubar, vbar = calc_uv_bar(restart.variables['u'][0, :, :], restart.variables['v'][0, :, :], N1.M)
        restart.variables['ua'][0,:]         = ubar
        restart.variables['va'][0,:]         = vbar

    restart.close()


# Interpolation stuff
# -------------------------------------------------------------------------------------------
def prepare_search(restart_date, days = 3):
    '''
    Search over a some days to make sure that you will find the correct restart file
    '''
    start = restart_date-timedelta(days =days)

    start_string = "{0.year}-{0.month:02}-{0.day:02}".format(start)
    stop_string  = "{0.year}-{0.month:02}-{0.day:02}".format(restart_date)

    return start_string, stop_string


# Fetch data from the ROMS output and dump to the FVCOM grid
# ----------------------------------------------------------------------------------------
def horizontal_interpolation(ROMS_out, N1, uv):
    '''
    Simple nearest neighbor interpolation algorithm.
    '''
    class HORZfield: pass
    HORZfield.zeta = ROMS_out.zeta[N1.rho_index]
    HORZfield.temp = ROMS_out.temp[N1.rho_index,:]
    HORZfield.salt = ROMS_out.salt[N1.rho_index,:]
    if uv:
        HORZfield.u = ROMS_out.u[N1.u_index,:]
        HORZfield.v = ROMS_out.v[N1.v_index,:]
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

def calc_uv_bar(u, v, M):
    '''
    Calculate vertical average of velocity (ubar, vbar).
    '''
    h_uv    = np.squeeze(M.h_uv)
    siglevz = M.siglev_c.T * h_uv
    dz      = np.abs(np.diff(siglevz, axis=0))
    ubar    = np.sum(u*dz, axis=0) / h_uv
    vbar    = np.sum(v*dz, axis=0) / h_uv
    return ubar, vbar

class N1ROMS:
    '''
    Object that finds the nearest
    '''
    def __init__(self, x, y, xc, yc, ROMS):
        '''
        Initialize empty attributes
        '''
        self.x = x;   self.y = y   # for FVCOM nodes
        self.xc = xc; self.yc = yc # for FVCOM cells
        self.ROMS = ROMS
        self._cropped_mask()

    @property
    def fv_nodes(self):
        return np.array([self.x, self.y]).transpose()

    @property
    def fv_cells(self):
        return np.array([self.xc, self.yc]).transpose()

    # Masking to only retain ROMS points covering FVCOM domain
    @property
    def fv_rho_mask(self):
        return self.ROMS.fv_rho_mask

    @property
    def fv_u_mask(self):
        return self.ROMS.fv_u_mask

    @property
    def fv_v_mask(self):
        return self.ROMS.fv_v_mask

    # Quick access to depth at FVCOM points
    @property
    def fvcom_rho_dpt(self):
        return self.ROMS.h_rho[self.fv_rho_mask][self.Land_rho==0][self.rho_index]

    @property
    def fvcom_u_dpt(self):
        return self.ROMS.h_u[self.fv_u_mask][self.Land_u==0][self.u_index]

    @property 
    def fvcom_v_dpt(self):
        return self.ROMS.h_v[self.fv_v_mask][self.Land_v==0][self.v_index]

    # ROMS grid angle relative to FVCOM
    @property
    def fvcom_angle(self):
        '''
        ROMS angle at FVCOM nodes.
        '''
        return self.ROMS.angle[self.fv_rho_mask][self.rho_index]

    # ROMS land points identifier
    @property
    def Land_rho(self):
        return self.ROMS.rho_mask[self.fv_rho_mask]

    @property
    def Land_u(self):
        return self.ROMS.u_mask[self.fv_u_mask]

    @property
    def Land_v(self):
        return self.ROMS.v_mask[self.fv_v_mask]
    
    # Positions of ocean rho, u and v points as FVCOM-covernig arrays
    @property
    def x_rho(self):
        return self.ROMS.x_rho[self.fv_rho_mask][self.Land_rho==0]
    
    @property
    def y_rho(self):
        return self.ROMS.y_rho[self.fv_rho_mask][self.Land_rho==0]
    
    @property
    def x_u(self):
        return self.ROMS.x_u[self.fv_u_mask][self.Land_u==0]
    
    @property
    def y_u(self):
        return self.ROMS.y_u[self.fv_u_mask][self.Land_u==0]

    @property
    def x_v(self):
        return self.ROMS.x_v[self.fv_v_mask][self.Land_v==0]
    
    @property
    def y_v(self):
        return self.ROMS.y_v[self.fv_v_mask][self.Land_v==0]

    @property
    def rho_tree(self):
        return KDTree(np.array([self.x_rho, self.y_rho]).transpose())

    @property
    def u_tree(self):
        return KDTree(np.array([self.x_u, self.y_u]).transpose())

    @property
    def v_tree(self):
        return KDTree(np.array([self.x_v, self.y_v]).transpose())

    @property
    def rho_index(self):
        _, ind = self.rho_tree.query(self.fv_nodes)
        return ind
    
    @property
    def u_index(self):
        _, ind = self.u_tree.query(self.fv_cells)
        return ind

    @property
    def v_index(self):
        _, ind = self.v_tree.query(self.fv_cells)
        return ind

    def _cropped_mask(self):
        # of rho mask
        i, j = np.where(self.fv_rho_mask)
        self.m_ri = min(i); self.x_ri = max(i)
        self.m_rj = min(j); self.x_rj = max(j)

        # of u mask
        i, j = np.where(self.fv_u_mask)
        self.m_ui = min(i); self.x_ui = max(i)
        self.m_uj = min(j); self.x_uj = max(j)

        i, j = np.where(self.fv_v_mask)
        self.m_vi = min(i); self.x_vi = max(i)
        self.m_vj = min(j); self.x_vj = max(j)


class CouldNotFindRestartTime(Exception): pass