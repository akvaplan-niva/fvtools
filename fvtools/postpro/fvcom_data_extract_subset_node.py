#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import datetime
import time
import matplotlib.mlab as mlab
import sys

from netCDF4 import Dataset
import netCDF4
import numpy as np
import matplotlib.mlab as mp
import time


import fvtools.grid.fvcom_grd
from fvtools.grid import tools

 # extract a single variable or multi varibles
def extract2file_subset_node_var(
    fileList="fileList.txt",
    output_file="subset_output",
    start_time=None,
    stop_time=None,
    subset=None,
    variables=None  # e.g., ['zeta', 'zisf','temp','salinity','meltrate']
):
    '''
    Extract selected FVCOM node-based variables on a subset of nodes
    and save to separate NetCDF files (no vertical interpolation).
    '''

    # Support single string as variable
    if isinstance(variables, str):
        variables = [variables]

    supported_vars = ['zeta', 'zisf', 'meltrate', 'temp', 'salinity']
    if variables is None:
        raise ValueError("Please specify one or more variables to extract.")
    for v in variables:
        if v not in supported_vars:
            raise ValueError(f"Unsupported variable: {v}")

    # Load file list
    fl = tools.Filelist(fileList, start_time=start_time, stop_time=stop_time)

    # Prepare first file
    first_time = True
    already_read = ""
    outputs = {}
    print(f"Shape: {subset.shape}")

    for file_name, index, fvtime, counter in zip(fl.path, fl.index, fl.time, range(len(fl.time))):
        if file_name != already_read:
            d = Dataset(file_name, 'r')
            print(f"📂 Reading: {file_name}")
        already_read = file_name

        # Determine subset only once
        if first_time:
            if subset is None:
                total_nodes = d.variables['zeta'].shape[1]
                subset = np.arange(total_nodes, dtype=int)
            else:
                subset = np.array(subset, dtype=int)

            numberOfgridPoints = len(subset)

            # Create output files for selected variables
            for var in variables:
                if var in d.variables:
                    shape = d.variables[var].shape
                    if len(shape) == 3:  # 3D (space, depth, time) — T/S
                        nz = shape[1]
                        nc = Dataset(f"{output_file}_{var}.nc", "w", format="NETCDF4")
                        nc.createDimension("time", None)
                        nc.createDimension("space", numberOfgridPoints)
                        nc.createDimension("siglay", nz)
                        nc.createVariable("fvcom_time", "f4", ("time",))
                        nc.createVariable("lon", "f4", ("space",))[:] = d.variables["lon"][subset]
                        nc.createVariable("lat", "f4", ("space",))[:] = d.variables["lat"][subset]
                        nc.createVariable("x", "f4", ("space",))[:] = d.variables["x"][subset]
                        nc.createVariable("y", "f4", ("space",))[:] = d.variables["y"][subset]
                        var_out = nc.createVariable(var, "f4", ("space", "siglay", "time"))
                        outputs[var] = (nc, var_out)
                    elif len(shape) == 2:  # 2D (space, time)
                        nc = Dataset(f"{output_file}_{var}.nc", "w", format="NETCDF4")
                        nc.createDimension("time", None)
                        nc.createDimension("space", numberOfgridPoints)
                        nc.createVariable("fvcom_time", "f4", ("time",))
                        nc.createVariable("lon", "f4", ("space",))[:] = d.variables["lon"][subset]
                        nc.createVariable("lat", "f4", ("space",))[:] = d.variables["lat"][subset]
                        nc.createVariable("x", "f4", ("space",))[:] = d.variables["x"][subset]
                        nc.createVariable("y", "f4", ("space",))[:] = d.variables["y"][subset]
                        var_out = nc.createVariable(var, "f4", ("space", "time"))
                        outputs[var] = (nc, var_out)
                    else:
                        print(f"⚠️ Unsupported shape for variable '{var}': {shape}")
                else:
                    print(f"⚠️ Variable '{var}' not found in file.")

            first_time = False

        # Write to each selected variable file
        for var in variables:
            if var in d.variables and var in outputs:
                nc, var_out = outputs[var]
                if len(var_out.shape) == 3:
                    var_out[:, :, counter] = d.variables[var][index, :, subset].T
                elif len(var_out.shape) == 2:
                    var_out[:, counter] = d.variables[var][index, subset]
                nc.variables["fvcom_time"][counter] = d.variables["time"][index]

    # Close all output files
    for nc, _ in outputs.values():
        nc.close()
    d.close()