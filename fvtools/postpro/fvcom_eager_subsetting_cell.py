#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import datetime
import time
import sys
import collections
import xarray as xr

from netCDF4 import Dataset
import numpy as np

import fvtools.grid.fvcom_grd
from fvtools.grid import tools

def extract2file_subset_cell_var(
    fileList="fileList.txt",
    output_file="subset_output",
    start_time=None,
    stop_time=None,
    subset=None,
    variables=None,# e.g., ['u', 'v', 'ua', 'va']
    cell_depths=None, # sigma layer depth at subset nodes read from grd info
    include_depth=False, # write sigma layer depths on subset cells
    include_ids=False  # subset cell ids
):
    '''
    Extract selected FVCOM cell-based variables on a subset of cells
    and save to separate NetCDF files (no vertical interpolation).
    '''

    # Accept single variable string
    if isinstance(variables, str):
        variables = [variables]

    supported_vars = ['u', 'v', 'ua', 'va']
    if variables is None:
        raise ValueError("Please specify one or more variables to extract.")
    for v in variables:
        if v not in supported_vars:
            raise ValueError("Unsupported variable: {}".format(v))

    # Load file list
    fl = tools.Filelist(fileList, start_time=start_time, stop_time=stop_time)   

    # Prepare iteration
    first_time = True
    already_read = ""
    outputs = {}

    grouped_indices = collections.defaultdict(list)
    for file, idx in zip(fl.path, fl.index):
        grouped_indices[file].append(idx)

    file_indices = {}
    for file, indices in grouped_indices.items():
        if indices:
            file_indices[file] = (min(indices), max(indices))


    write_index = 0
    #for file_name, index, fvtime, counter in zip(fl.path, fl.index, fl.time, range(len(fl.time))):
    for file_name, index_range in file_indices.items():

        start_idx, end_idx = index_range # index for reading data
        end_idx = end_idx + 1
        chunk_size = end_idx - start_idx # chunk size in time dimension
        
        print(f"Reading time indices: {start_idx}-{end_idx} and Writing to indices: {write_index}-{write_index + chunk_size}")
        wr_stidx = write_index # index for writing to file
        wr_edidx = write_index + chunk_size
        
        if file_name != already_read:
            # Use xarray and load into memory
            #d = xr.load_dataset(file_name, decode_times=False)   # eager load
            d = xr.open_dataset(file_name, decode_times=False)   # lazy load
            print("📂 Reading:", file_name)
        already_read = file_name

        if first_time:
            if subset is None:
                total_cells = d['ua'].shape[1]  # xarray DataArray, same as before
                subset = np.arange(total_cells, dtype=int)
            else:
                subset = np.array(subset, dtype=int)

            numberOfgridPoints = len(subset)           

            # Create output files
            for var in variables:
                if var in d.variables:
                    shape = d.variables[var].shape
                    if len(shape) == 3:  # 3D (space, depth, time)
                        nz = shape[1]
                        nc = Dataset(f"{output_file}_{var}.nc", "w", format="NETCDF4")
                        nc.createDimension("time", None)
                        nc.createDimension("space", numberOfgridPoints)
                        nc.createDimension("siglay", nz)
                        nc.createVariable("fvcom_time", "f4", ("time",))
                        nc.createVariable("xc", "f4", ("space",))[:] = d.variables["xc"][subset]
                        nc.createVariable("yc", "f4", ("space",))[:] = d.variables["yc"][subset]
                        if include_ids and ("cell_id" not in nc.variables):
                            assert len(subset) == numberOfgridPoints  # use your variable name
                            vni = nc.createVariable("cell_id", "i4", ("space",))
                            vni.long_name = "cell index used for sebsetting"
                            vni[:] = np.asarray(subset, dtype=np.int32)
                        if include_depth and (cell_depths is not None) and ("depth" not in nc.variables):
                            nz = cell_depths.shape[1]
                            assert cell_depths.shape == (numberOfgridPoints, nz)
                            vnd = nc.createVariable("depth", "f4", ("space", "siglay"))
                            vnd.long_name = "depth at sigma layers for subset cells"
                            vnd[:] = np.asarray(cell_depths, dtype=np.float32)

                        var_out = nc.createVariable(var, "f4", ("space", "siglay", "time"))
                        outputs[var] = (nc, var_out)
                    elif len(shape) == 2:  # 2D (space, time)
                        nc = Dataset(f"{output_file}_{var}.nc", "w", format="NETCDF4")
                        nc.createDimension("time", None)
                        nc.createDimension("space", numberOfgridPoints)
                        nc.createVariable("fvcom_time", "f4", ("time",))
                        nc.createVariable("xc", "f4", ("space",))[:] = d.variables["xc"][subset]
                        nc.createVariable("yc", "f4", ("space",))[:] = d.variables["yc"][subset]
                        if include_ids and ("cell_id" not in nc.variables):
                            assert len(subset) == numberOfgridPoints  # use your variable name
                            vni = nc.createVariable("cell_id", "i4", ("space",))
                            vni.long_name = "cell index used for sebsetting"

                        var_out = nc.createVariable(var, "f4", ("space", "time"))
                        outputs[var] = (nc, var_out)
                    else:
                        print("⚠️ Unsupported shape for variable '{}': {}".format(var, shape))
                else:
                    print("⚠️ Variable '{}' not found in file.".format(var))
            d.close()
            first_time = False

        #for var in outputs:
        ds = xr.open_dataset(file_name, decode_times=False)  
        indexers = {'time': slice(start_idx, end_idx),'nele': subset.tolist()}
        for var in variables:    
            nc, var_out = outputs[var]
            
            if len(var_out.shape) == 3:
                # Read all data for this file at once
                # Use the dictionary in the .isel() method                
                ds_loaded = ds[var_out.name].load()
                ds_sel = ds_loaded.isel(indexers)                    
                
                target_dims = ['nele', 'siglay', 'time']
                ds_towrite = ds_sel.transpose(*target_dims)
            
                # Write to the output file at the correct time indices
                var_out[:, :, wr_stidx:wr_edidx] = ds_towrite

            if len(var_out.shape) == 2:
                # Read all data for this file at once
                # Use the dictionary in the .isel() method
                ds_loaded = ds[var_out.name].load()
                ds_sel = ds_loaded.isel(indexers)            
                
                target_dims = ['nele', 'time']                
                ds_towrite = ds_sel.transpose(*target_dims)
            
                # Write to the output file at the correct time indices
                var_out[:, wr_stidx:wr_edidx] = ds_towrite

            
            # Writing time 
            ds_time_loaded = ds['time'].load()
            ds_time_sel = ds_time_loaded.isel({'time': slice(start_idx, end_idx)})
            nc.variables["fvcom_time"][wr_stidx:wr_edidx] = ds_time_sel

        write_index += chunk_size # updating time index for writing to file    

    for nc, _ in outputs.values():
        nc.close()

