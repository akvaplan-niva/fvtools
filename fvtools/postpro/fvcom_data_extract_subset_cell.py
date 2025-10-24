#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import datetime
import time
import sys

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
    variables=None  # e.g., ['u', 'v', 'ua', 'va'] 
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

    for file_name, index, fvtime, counter in zip(fl.path, fl.index, fl.time, range(len(fl.time))):
        if file_name != already_read:
            d = Dataset(file_name, 'r')
            print("📂 Reading:", file_name)
        already_read = file_name

        if first_time:
            if subset is None:
                total_cells = d.variables['ua'].shape[1]
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
                        var_out = nc.createVariable(var, "f4", ("space", "siglay", "time"))
                        outputs[var] = (nc, var_out)
                    elif len(shape) == 2:  # 2D (space, time)
                        nc = Dataset(f"{output_file}_{var}.nc", "w", format="NETCDF4")
                        nc.createDimension("time", None)
                        nc.createDimension("space", numberOfgridPoints)
                        nc.createVariable("fvcom_time", "f4", ("time",))
                        nc.createVariable("xc", "f4", ("space",))[:] = d.variables["xc"][subset]
                        nc.createVariable("yc", "f4", ("space",))[:] = d.variables["yc"][subset]
                        var_out = nc.createVariable(var, "f4", ("space", "time"))
                        outputs[var] = (nc, var_out)
                    else:
                        print("⚠️ Unsupported shape for variable '{}': {}".format(var, shape))
                else:
                    print("⚠️ Variable '{}' not found in file.".format(var))

            first_time = False

        # Write data
        for var in variables:
            if var in d.variables and var in outputs:
                nc, var_out = outputs[var]
                if len(var_out.shape) == 3:
                    var_out[:, :, counter] = d.variables[var][index, :, subset].T
                elif len(var_out.shape) == 2:
                    var_out[:, counter] = d.variables[var][index, subset]
                nc.variables["fvcom_time"][counter] = d.variables["time"][index]

    for nc, _ in outputs.values():
        nc.close()
    d.close()
