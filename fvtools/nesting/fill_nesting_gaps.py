'''
Routine that performs a tidal analysis, and subsequently use this to fill gaps in the nestingfile
 - Harmonic analysis of ua, va, u, v and zeta
 - Linear interpolation of temp, salinity, weight_cell, weight_node and non-harmonic portions of the other variables.

Known slow stuff:
 - anything to do with netCDF read timeseries 
   - from nesting file
   - write to netCDF
 We therefore keep all relevant variables in memory in this routine. This comes at a cost wrt running on login nodes, so please keep this routine to work nodes.


## Example use:
## file: run_gap_filler.py

 import sys
 sys.path.append('/cluster/home/hes001/fvtools')
 import fvtools.nesting.fill_nesting_gaps as fg
 fg.main('PO10_v2_nest_feb.nc', nprocs = None)

## file: run_python.sh (on Betzy using 4 nodes, normal partition)

#!/bin/bash
#
#SBATCH --account=nn9238k
#SBATCH --job-name=fillgaps
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
##SBATCH --qos=devel
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=ALL
##SBATCH --mem-per-cpu=1G

module load IPython/7.18.1-GCCcore-10.2.0
module load GEOS/3.9.1-GCC-10.2.0

source /cluster/work/users/hes001/pyipy/bin/activate

python run_gap_filler.py
'''


import multiprocessing as mp
import progressbar as pb
import numpy as np 
import utide

from fvtools.grid.tools import num2date, date2num
from fvtools.grid.fvcom_grd import FVCOM_grid
from functools import cached_property
from time import gmtime, strftime
from dataclasses import dataclass
from netCDF4 import Dataset

def main(nestingfile, nprocs = 20, constits = ['K2','S2','M2','N2','K1','P1','O1','Q1']):
    '''
    Fill missing dates in the nestingfile using results from a utide harmonical analysis and a linear interpolation of non-tidal contributions
    ----
    - nestingfile: file with gaps
    - nprocs:      number of processors to use (None for all available CPUs, but this should just be done on a work node)
    - constits:    sets the major linear component by default, can also be set "auto" to let utide decide what to do itself

    Will use a shitload of memory, this routine should be used on work nodes - not on a simple login node.
    '''
    new_timeseries, inds_copy, inds_overlap_orig, inds_fill = get_new_timeseries(nestingfile)
    M = FVCOM_grid(nestingfile)
    create_nc_forcing_file(f'{M.casename}_no_gaps.nc', M, nestingfile, date2num(new_timeseries), M.reference)

    # utide settings
    solver_settings      = {'epoch': '1858-11-17', 'verbose': False, 'constit' : constits}
    reconstruct_settings = {'epoch': '1858-11-17', 'verbose': False}

    # do the filling business
    F = FillGaps(nestingfile, f'{M.casename}_no_gaps.nc',
                 new_timeseries, inds_copy, inds_overlap_orig, inds_fill, 
                 solver_settings, reconstruct_settings)
    F.fill_gaps(nprocs)

def get_new_timeseries(nestingfile):
    '''
    receives time vector, identifies where we are short of data
    '''
    with Dataset(nestingfile, 'r') as f:
        timeseries = num2date(Itime=f['Itime'][:], Itime2=f['Itime2'][:])

    # Find timestep in forcing file, create new and continous timeseries (constant dt)
    dt = np.diff(timeseries)
    new_timeseries = [timeseries.min() + i*dt.min() for i in range(1+int((timeseries.max()-timeseries.min())/dt.min()))]

    # Figure out which indices in the new forcing file can use existing data, and which we will fill using harmonic analysis/interpolation algorithms
    _, inds_overlap_orig, indices_to_copy_from_original = np.intersect1d(timeseries, new_timeseries, return_indices = True)
    _, indices_to_fill, _ = np.intersect1d(new_timeseries, np.setdiff1d(new_timeseries, timeseries), return_indices = True)

    # sanity checks
    assert len(indices_to_copy_from_original) == len(timeseries), 'Unexpected error, the new timeseries does not cover the old timeseries'
    assert len(new_timeseries)-len(indices_to_fill) == len(timeseries), 'Unexpected error, the gaps we fill cover more or less than the timeseries'
    return new_timeseries, indices_to_copy_from_original, inds_overlap_orig, indices_to_fill

class FillGaps:
    '''
    Class that performs tidal analysis on nestingfile, uses this (and a linear interpolation) to fill gaps
    '''
    def __init__(self, original_file, new_file, 
                       new_timeseries,
                       inds_copy, inds_overlap_orig, inds_fill,
                       solver_settings, reconstruct_settings,
                ):
        '''
        input:
        - original_file   - original forcing file
        - new_file        - new forcing file (seems tedious)
        - new_timeseries  - full timeseries to be used in the new forcing file
        - inds_copy       - data indices corresponding to times in new_timeseries we will copy directly from the original forcing file
        - inds_fill       - indices we will have to fill using the harmonic analysis and a linear interpolation of fields
        '''
        self.original_file = original_file
        self.new_file  = new_file
        self.new_time  = np.array(date2num(new_timeseries))
        self.inds_copy = inds_copy
        self.inds_overlap_orig = inds_overlap_orig
        self.inds_fill = inds_fill
        self.solver_settings = solver_settings
        self.reconstruct_settings = reconstruct_settings
    
    @cached_property
    def original_time(self):
        with Dataset(self.original_file, 'r') as d:
            time = d['time'][:]
        return time

    @cached_property
    def nodes(self):
        with Dataset(self.original_file, 'r') as d:
            n_nodes = d.dimensions['node'].size
        return n_nodes
    
    @cached_property
    def cells(self):
        with Dataset(self.original_file, 'r') as d:
            n_cells = d.dimensions['nele'].size
        return n_cells

    def fill_gaps(self, nprocs):
        '''
        Loops over all nodes and cells in the FVCOM forcing file to fill gaps
        '''
        # Prepare workers
        # ----
        manager = mp.Manager()
        q = manager.Queue()

        if nprocs is None:
            print(f'- Number of processors: {mp.cpu_count()}')
            pool = mp.Pool(mp.cpu_count())
        else:
            print(f'- Number of processors: {nprocs}')
            assert nprocs >=3, 'Total number of processors must be greater than 2'
            pool = mp.Pool(nprocs)

        # Put listener (write to netCDF) to work
        # ----
        watcher = pool.apply_async(write_to_file_in_memory, (self.new_file, self.cells+self.nodes, q))

        # Fire off workers
        # ----
        jobs = []
        with Dataset(self.original_file, 'r') as original:
            salt = original['salinity'][:]
            temp = original['temp'][:]
            zeta = original['zeta'][:]
            ua = original['ua'][:]
            va = original['va'][:]
            u = original['u'][:]
            v = original['v'][:]
            weight_cell = original['weight_cell'][:]
            weight_node = original['weight_node'][:]
            lat = original['lat'][:]
            latc = original['latc'][:]
            for i in range(self.nodes):
                job = pool.apply_async(self.handle_node_data, (zeta[:,i], temp[:,:,i], salt[:,:,i], weight_node[:,i], lat[i], i, q))
                jobs.append(job)

            for i in range(self.cells):
                job = pool.apply_async(self.handle_cell_data, (ua[:,i], va[:,i], u[:,:,i], v[:,:,i], weight_cell[:,i], latc[i], i, q))
                jobs.append(job)

        # Collect results from the workers through the pool result queue
        # ----
        for job in jobs:
            job.get()

        # Now we are done, retire the listener
        # ----
        q.put('retire')
        pool.close()
        pool.join()

    # Versions where processors do not to interact with netCDF
    def handle_node_data(self, zeta, temp, salinity, weight_node, latitude, i, q):
        '''
        Fill gaps in node data
        '''
        # Might make sense to just load everything to memory from the getgo
        node = NodeData(original_time = self.original_time,
                        new_time = self.new_time, 
                        zeta = zeta, 
                        temp = temp, 
                        salinity = salinity,
                        weight_node = weight_node,
                        idx = i,
                        lat = latitude,
                        copy_inds = self.inds_copy,
                        inds_overlap_orig = self.inds_overlap_orig,
                        replace_inds = self.inds_fill,
                        solver_settings = self.solver_settings, 
                        reconstruct_settings = self.reconstruct_settings)
        node.fill_gaps()
        q.put(node)

    def handle_cell_data(self, ua, va, u, v, weight_cell, latitude, i, q):
        '''
        Harmonic analysis of cell based data
        '''
        cell = CellData(new_time = self.new_time,
                        original_time = self.original_time, 
                        ua = ua, 
                        va = va, 
                        u = u, 
                        v = v,
                        weight_cell = weight_cell,
                        lat = latitude,
                        copy_inds = self.inds_copy,
                        inds_overlap_orig = self.inds_overlap_orig,
                        replace_inds = self.inds_fill,
                        idx = i,
                        solver_settings = self.solver_settings, 
                        reconstruct_settings = self.reconstruct_settings)
        cell.fill_gaps()
        q.put(cell)

def write_to_netCDF(new_file, tot_iterations, q):
    '''
    Write fixed data to the forcing file
    - very slow since writing to netCDF across time dimensions is _really_ slow.
    '''
    bar = pb.ProgressBar(widgets = ['Fill gaps in nesting file:', pb.Percentage(), pb.Bar(), pb.AdaptiveETA()], 
                         maxval = tot_iterations)
    bar.start()
    i = 0
    with Dataset(new_file, 'r+') as out:
        while True:
            data = q.get()
            if data == 'retire':
                break
                bar.finish()

            if type(data) == CellData:
                bar.update(i)
                i+=1
                out['ua'][:, data.idx]   = data.ua
                out['va'][:, data.idx]   = data.va
                out['u'][:, :, data.idx] = data.u
                out['v'][:, :, data.idx] = data.v
                out['weight_cell'][:, data.idx] = data.weight_cell

            elif type(data) == NodeData:
                bar.update(i)
                i+=1
                out['zeta'][:, data.idx] = data.zeta
                out['temp'][:, :, data.idx]  = data.temp
                out['salinity'][:, :, data.idx] = data.salinity
                out['weight_node'][:, data.idx] = data.weight_node
                out['hyw'][:, :, data.idx] = np.zeros((data.salinity.shape[0], data.salinity.shape[1]+1)) # time, siglev (= siglay + 1)

def write_to_file_in_memory(new_file, tot_iterations, q):
    '''
    Write fixed data to the forcing file
    - very slow since writing to netCDF across time dimensions is _really_ slow.
    '''
    bar = pb.ProgressBar(widgets = ['Fill gaps in nesting file:', pb.Percentage(), pb.Bar(), pb.AdaptiveETA()], 
                         maxval = tot_iterations)
    bar.start()
    i = 0

    # DeTided will (temporarilly) store the full data
    DeTided = DeTidedData(new_file)

    while True:
        data = q.get()
        if data == 'retire':
            bar.finish()
            print('- Write to the new nestingfile')
            with Dataset(new_file, 'r+') as out:
                for field in ['ua', 'va', 'u', 'v', 'weight_cell', 'weight_node', 'zeta', 'temp', 'salinity', 'hyw']:
                    out[field][:] = getattr(DeTided, field)
            print('Done.')
            break
            
        if type(data) == CellData:
            bar.update(i)
            i+=1
            DeTided.ua[:, data.idx]   = data.ua
            DeTided.va[:, data.idx]   = data.va
            DeTided.u[:, :, data.idx] = data.u
            DeTided.v[:, :, data.idx] = data.v
            DeTided.weight_cell[:, data.idx] = data.weight_cell

        elif type(data) == NodeData:
            bar.update(i)
            i+=1
            DeTided.zeta[:, data.idx] = data.zeta
            DeTided.temp[:, :, data.idx]  = data.temp
            DeTided.salinity[:, :, data.idx] = data.salinity
            DeTided.weight_node[:, data.idx] = data.weight_node
            DeTided.hyw[:, :, data.idx] = np.zeros((data.salinity.shape[0], data.salinity.shape[1]+1)) # time, siglev (= siglay + 1)

class DeTidedData:
    '''
    Temporary storage for fields that are in memory (we write directly from this one to the output netCDF once we make it that far)
    '''
    def __init__(self, new_file):
        '''
        We use the dimensions in new_file as templates for how large the temporary storages should be
        '''
        with Dataset(new_file, 'r+') as out:
            for field in ['ua', 'va', 'u', 'v', 'weight_cell', 'weight_node', 'zeta', 'temp', 'salinity', 'hyw']:
                setattr(self, field, np.zeros(out[field].shape))

@dataclass
class CellData:
    '''
    handles the gap plugging of cell data (works on one node at the time)
    '''
    # Time of original and complete timeseries
    new_time: np.array(0)
    original_time: np.array(0)
    ua:  np.array(0)
    va:  np.array(0)
    u:   np.array(0)
    v:   np.array(0)
    weight_cell: np.array(0)
    lat: 0
    copy_inds: np.array(0)
    inds_overlap_orig: np.array(0)
    replace_inds: np.array(0)
    idx: 0
    solver_settings: {}
    reconstruct_settings: {}

    # Don't let any of these add attributes to the class, we want to keep these lightweight since they are passed among processors
    def fill_gaps(self):
        '''
        Does the harmonic analysis, fills gaps
        '''
        # update ua, va
        self.ua, self.va = self.handle_uv(self.ua, self.va)

        # update u, v
        new_u = np.nan*np.ones((len(self.new_time), self.u.shape[1]))
        new_v = np.nan*np.ones((len(self.new_time), self.u.shape[1]))
        for i in range(self.u.shape[1]):
            new_u[:,i], new_v[:,i] = self.handle_uv(self.u[:,i], self.v[:,i])
        self.u, self.v = new_u, new_v
        self.weight_cell = self.time_interp(self.weight_cell)

        assert np.isnan(self.ua).any() == False, 'Found nans in ua field'
        assert np.isnan(self.va).any() == False, 'Found nans in va field'
        assert np.isnan(self.u).any() == False, 'Found nans in u field'
        assert np.isnan(self.v).any() == False, 'Found nans in v field'
        assert np.isnan(self.weight_cell).any() == False, 'Found nans in nesting weight field'

    def handle_uv(self, u_in, v_in):
        '''
        Fill gaps in u_in and v_in
        '''
        # Harmonic analysis of tidal current
        coef = self.harmonic_analysis(u_in, v_in)

        # Retrieve detided signal
        u_detided, v_detided = self.detide(u_in, v_in, coef) # duplicate reconstruction of timeseries, see if it can be avoided

        # Get pure tidal signal
        u_tide, v_tide = self.tidal_current(self.new_time, coef)

        # Find sum of detided and tide
        u = self.time_interp(u_detided)
        v = self.time_interp(v_detided)
        u_tot = u + u_tide
        v_tot = v + v_tide

        # Use u_tot and v_tot to replace missing time indices
        u_tot[self.copy_inds] = u_in[self.inds_overlap_orig] # dette er vel strengt tatt _alle_ indeksene i disse matrisene?
        v_tot[self.copy_inds] = v_in[self.inds_overlap_orig]
        return u_tot, v_tot

    def harmonic_analysis(self, u, v):
        '''
        Compute coef
        '''
        coef = utide.solve(self.original_time, u, v, lat = self.lat, **self.solver_settings)
        coef['umean'], coef['vmean'], coef['uslope'], coef['vslope'] = 0, 0, 0, 0
        return coef

    def detide(self, u, v, coef):
        '''
        detide a velocity signal
        '''
        u_t, v_t = self.tidal_current(self.original_time, coef)
        return u-u_t, v-u_t

    def tidal_current(self, time, coef):
        '''
        Returns detided timeseries
        '''
        tide = utide.reconstruct(time, coef, **self.reconstruct_settings)
        return tide['u'], tide['v']

    def time_interp(self, var):
        '''
        Copied from Achims detide
        '''
        # implicit: time is always first dimension in these files
        var_out = np.nan * np.zeros((len(self.new_time), *var.shape[1:]))
        for idx, _ in np.ndenumerate(var[0]):
            var_out[(slice(None), ) + idx] = np.interp(self.new_time, self.original_time, var[(slice(None), ) + idx])
        return var_out

@dataclass
class NodeData:
    '''
    handles the gap plugging of node data
    '''
    original_time: np.array(0)
    new_time: np.array(0)
    zeta: np.array(0)
    temp: np.array(0)
    salinity: np.array(0)
    weight_node: np.array(0)
    idx: 0
    lat: 0
    copy_inds: np.array(0)
    inds_overlap_orig: np.array(0)
    replace_inds: np.array(0)
    solver_settings: {}
    reconstruct_settings: {}

    def fill_gaps(self):
        '''
        Does the harmonic analysis, fills gaps
        '''
        # update zeta
        self.zeta = self.handle_wave(self.zeta)

        # Potential to handle temperature as wave as well
        new_temp = np.nan * np.ones((len(self.new_time), self.temp.shape[1]))
        new_salinity = np.nan * np.ones((len(self.new_time), self.temp.shape[1]))

        for i in range(self.temp.shape[1]):
            new_temp[:, i]     = self.time_interp(self.temp[:, i])
            new_salinity[:, i] = self.time_interp(self.salinity[:, i])

        self.temp = new_temp
        self.salinity = new_salinity
        self.weight_node = self.time_interp(self.weight_node)

        assert np.isnan(self.temp).any() == False, 'Found nans in temperature field'
        assert np.isnan(self.salinity).any() == False, 'Found nans in salinity field'
        assert np.isnan(self.zeta).any() == False, 'Found nans in sea surface field'
        assert np.isnan(self.weight_node).any() == False, 'Found nans in nesting weight field'

    def handle_wave(self, scalar_in):
        '''
        Fill gaps in u_in and v_in
        '''
        # Harmonic analysis of tidal current
        coef = self.harmonic_analysis(scalar_in)

        # Retrieve detided signal
        scalar_detided = self.detide(scalar_in, coef)

        # Get pure tidal signal
        scalar_tide = self.tidal_wave(self.new_time, coef)

        # Find sum of detided and tide
        scalar = self.time_interp(scalar_detided)
        scalar_tot = scalar + scalar_tide

        # Use u_tot and v_tot to replace missing time indices
        scalar_tot[self.copy_inds] = scalar_in[self.inds_overlap_orig]
        return scalar_tot

    def harmonic_analysis(self, scalar):
        '''
        Compute coef
        '''
        coef = utide.solve(self.original_time, scalar, lat = self.lat, **self.solver_settings)
        coef['mean'], coef['slope'] = 0, 0
        return coef

    def detide(self, scalar, coef):
        '''
        detide a velocity signal
        '''
        scalar_tide = self.tidal_wave(self.original_time, coef)
        return scalar - scalar_tide

    def tidal_wave(self, time, coef):
        '''
        Returns detided timeseries
        '''
        tide = utide.reconstruct(time, coef, **self.reconstruct_settings)
        return tide['h']

    def time_interp(self, var):
        '''
        Copied from Achims detide
        '''
        # implicit: time is always first dimension in these files
        var_out = np.nan * np.zeros((len(self.new_time), *var.shape[1:]))
        for idx, _ in np.ndenumerate(var[0]):
            var_out[(slice(None), ) + idx] = np.interp(self.new_time, self.original_time, var[(slice(None), ) + idx])
        return var_out

# Dump!
def create_nc_forcing_file(name, M, sourcefile, timesteps, epsg):
    '''
    Creates empty nc file formatted to fit FVCOM open boundary ocean forcing
    '''
    print(f'Writing {name}')
    with Dataset(sourcefile, 'r') as source:
        with Dataset(name, 'w', format='NETCDF4') as nc:
            # Write global attributes
            # ----
            nc.title        = 'FVCOM nesting File'
            nc.institution  = 'Akvaplan-niva AS'
            nc.source       = 'FVCOM grid (unstructured) nesting file'
            nc.created      = source.created
            nc.mother_model = source.mother_model
            nc.gaps_filles  = f'Gaps filled using utide: {strftime("%Y-%m-%d %H:%M:%S", gmtime())}'
            nc.interpolation_projection = source.interpolation_projection

            # Create dimensions
            # ----
            nc.createDimension('time', 0)
            nc.createDimension('node', len(M.x))
            nc.createDimension('nele', len(M.xc))
            nc.createDimension('three', 3)
            nc.createDimension('siglay', len(M.siglay[0,:]))
            nc.createDimension('siglev', len(M.siglev[0,:]))

            # Create variables and variable attributes
            # ----------------------------------------------------------
            time               = nc.createVariable('time', 'single', ('time',))
            time.units         = 'days since 1858-11-17 00:00:00'
            time.format        = 'modified julian day (MJD)'
            time.time_zone     = 'UTC'

            Itime              = nc.createVariable('Itime', 'int32', ('time',))
            Itime.units        = 'days since 1858-11-17 00:00:00'
            Itime.format       = 'modified julian day (MJD)'
            Itime.time_zone    = 'UTC'

            Itime2             = nc.createVariable('Itime2', 'int32', ('time',))
            Itime2.units       = 'msec since 00:00:00'
            Itime2.time_zone   = 'UTC'

            # positions
            # ----
            # node
            lon                = nc.createVariable('lon', 'single', ('node',))
            lat                = nc.createVariable('lat', 'single', ('node',))
            x                  = nc.createVariable('x', 'single', ('node',))
            y                  = nc.createVariable('y', 'single', ('node',))
            h                  = nc.createVariable('h', 'single', ('node',))

            # center
            lonc               = nc.createVariable('lonc', 'single', ('nele',))
            latc               = nc.createVariable('latc', 'single', ('nele',))
            xc                 = nc.createVariable('xc', 'single', ('nele',))
            yc                 = nc.createVariable('yc', 'single', ('nele',))
            hc                 = nc.createVariable('h_center', 'single', ('nele',))

            # grid parameters
            # ----
            nv                 = nc.createVariable('nv', 'int32', ('three', 'nele',))

            # node
            lay                = nc.createVariable('siglay','single',('siglay','node',))
            lev                = nc.createVariable('siglev','single',('siglev','node',))

            # center
            lay_center         = nc.createVariable('siglay_center','single',('siglay','nele',))
            lev_center         = nc.createVariable('siglev_center','single',('siglev','nele',))

            # Weight coefficients (since we are Ming from ROMS)
            # ----
            wc                 = nc.createVariable('weight_cell','single',('time','nele',))
            wn                 = nc.createVariable('weight_node','single',('time','node',))

            # time dependent variables
            # ----
            zeta               = nc.createVariable('zeta', 'single', ('time', 'node',))
            ua                 = nc.createVariable('ua', 'single', ('time', 'nele',))
            va                 = nc.createVariable('va', 'single', ('time', 'nele',))
            u                  = nc.createVariable('u', 'single', ('time', 'siglay', 'nele',))
            v                  = nc.createVariable('v', 'single', ('time', 'siglay', 'nele',))
            temp               = nc.createVariable('temp', 'single', ('time', 'siglay', 'node',))
            salt               = nc.createVariable('salinity', 'single', ('time', 'siglay', 'node',))
            hyw                = nc.createVariable('hyw', 'single', ('time', 'siglev', 'node',))

            # copy the grid metrics
            # ----
            for var in ['lat', 'lon', 'x', 'y', 'latc', 'lonc', 'xc', 'yc', 'nv', 'h', 'h_center', 'siglev', 'siglay', 'siglev_center', 'siglay_center']:
                nc.variables[var][:] = source.variables[var][:]

            # Initialize the time (so that we can multiprocess the download)
            for counter, fvcom_time in enumerate(timesteps):
                nc.variables['time'][counter] = fvcom_time
                nc.variables['Itime'][counter] = np.floor(fvcom_time)
                nc.variables['Itime2'][counter] = np.round((fvcom_time - np.floor(fvcom_time)) * 60 * 60 * 1000, decimals = 0)*24