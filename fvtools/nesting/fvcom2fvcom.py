import os
import datetime
import sys
import time
from argparse import ArgumentParser

#from IPython.core.debugger import Tracer
from netCDF4 import Dataset
import netCDF4
import numpy as np

import grid.fvcom_grd as fvg
from grid.tools import Filelist
from nesting.vertical_interpolation import calc_interp_matrices

def make_nestfile(ngrd        = 'ngrd.npy',
                  fileList    = "fileList.txt",
                  output_file = "fvcom_nest.nc",
                  start_time  = None,
                  stop_time   = None,
                  vertical_interpolation = False,
                  tides = False):
    '''
    Extract subset of FVCOM T and S data and export to netcdf file.
    '''

    fl = Filelist(fileList, start_time=start_time, stop_time=stop_time)

    # Get indices of grid points
    ngrd     = fvg.NEST_grid(ngrd)
    ngrd.nid = np.squeeze(ngrd.nid)
    ngrd.cid = np.squeeze(ngrd.cid)

    # Check that positions match
    check_grid(ngrd, fl)

    if vertical_interpolation:
        # Calculate indices/weights for vertical interpolation
        zn = ngrd.siglayz
        zn_mother = ngrd.siglayz_mother
        ind1_n, ind2_n, w1_n, w2_n = calc_interp_matrices(-zn_mother.T, -zn.T)

        zc = ngrd.siglayz_uv
        zc_mother = ngrd.siglayz_uv_mother
        ind1_c, ind2_c, w1_c, w2_c = calc_interp_matrices(-zc_mother.T, -zc.T)


    # Read data from FVCOM result files and write to nesting nc-file
    first_time=1
    already_read=""
    for file_name,index,fvtime,counter in zip(fl.path,fl.index,fl.time,range(0,len(fl.time))):
        if file_name != already_read:
            d=Dataset(file_name,'r')
            print(file_name)

        already_read = file_name

        if first_time: # Initialize nc-files
            out_nest = Dataset(output_file, 'w', format='NETCDF4')
            kb=ngrd.siglev.shape[1]

            # Create dimensions
            timedim = out_nest.createDimension('time', 0)
            nodedim = out_nest.createDimension('node', len(ngrd.nid))
            celldim = out_nest.createDimension('nele', len(ngrd.cid))
            threedim = out_nest.createDimension('three', 3)
            levdim = out_nest.createDimension('siglev',kb)
            laydim = out_nest.createDimension('siglay',kb-1)

            # Create variables
            time = out_nest.createVariable('time', 'single',('time',))
            time.units = "days since 1858-11-17 00:00:00"
            time.format = "modified julian day (MJD)"
            time.time_zone = "UTC"

            Itime = out_nest.createVariable('Itime', 'int32',('time',))
            Itime.units = "days since 1858-11-17 00:00:00"
            Itime.format = "modified julian day (MJD)"
            Itime.time_zone = "UTC"
            Itime2 = out_nest.createVariable('Itime2', 'int32',('time',))
            Itime2.units = "msec since 00:00:00"
            Itime2.time_zone = "UTC"
            lon = out_nest.createVariable('lon', 'single',('node',))
            lat = out_nest.createVariable('lat', 'single', ('node',))
            lonc = out_nest.createVariable('lonc', 'single',('nele',))
            latc = out_nest.createVariable('latc', 'single', ('nele',))
            x = out_nest.createVariable('x', 'single', ('node',))
            y = out_nest.createVariable('y', 'single', ('node',))
            h = out_nest.createVariable('h', 'single', ('node',))
            h_center =  out_nest.createVariable('h_center', 'single', ('nele',))
            nv = out_nest.createVariable('nv', 'int32', ('three', 'nele',))
            siglay = out_nest.createVariable('siglay', 'single', ('siglay', 'node',))
            siglev = out_nest.createVariable('siglev', 'single', ('siglev', 'node',))
            siglay_center = out_nest.createVariable('siglay_center', 'single', ('siglay', 'nele',))
            siglev_center = out_nest.createVariable('siglev_center', 'single', ('siglev', 'nele',))
            xc = out_nest.createVariable('xc', 'single', ('nele',))
            yc = out_nest.createVariable('yc', 'single', ('nele',))
            zeta = out_nest.createVariable('zeta', 'single', ('time', 'node',))
            ua = out_nest.createVariable('ua', 'single', ('time', 'nele',))
            va = out_nest.createVariable('va', 'single', ('time', 'nele',))
            u = out_nest.createVariable('u', 'single', ('time', 'siglay', 'nele',))
            v = out_nest.createVariable('v', 'single', ('time', 'siglay', 'nele',))
            temp = out_nest.createVariable('temp', 'single', ('time', 'siglay', 'node',))
            salt = out_nest.createVariable('salinity', 'single', ('time', 'siglay', 'node',))
            hyw = out_nest.createVariable('hyw', 'single', ('time', 'siglev', 'node',))
            #tracer = out_nest.createVariable('tracer_c', 'f4', ('node', 'siglay', 'time'))

            # Write data to grid fields (h, lon, lat, x, y and z)
            # ----
            nv[:]   = ngrd.nv.transpose()
            lon[:]  = ngrd.lonn
            lat[:]  = ngrd.latn
            x[:]    = ngrd.xn
            y[:]    = ngrd.yn
            lonc[:] = ngrd.lonc
            latc[:] = ngrd.latc
            xc[:]   = ngrd.xc
            yc[:]   = ngrd.yc
            h[:]    = ngrd.h
            h_center[:] = ngrd.hc
            siglay[:]   = ngrd.siglay.transpose()
            siglay_center[:] = ngrd.siglay_center.transpose()
            siglev[:]        = ngrd.siglev.transpose()
            siglev_center[:] = ngrd.siglev_center.transpose()

            first_time = 0

        # Read data from current time step
        time[counter] = d.variables['time'][index]
        Itime[counter] = d.variables['Itime'][index]
        Itime2[counter] = d.variables['Itime2'][index]
        zeta[counter, :] = np.transpose(d.variables['zeta'][index, :][ngrd.nid])
        ua[counter, :] = np.transpose(d.variables['ua'][index, :][ngrd.cid])
        va[counter, :] = np.transpose(d.variables['va'][index, :][ngrd.cid])



        if vertical_interpolation:
            print('Do vertical interpolation')
            ui = d.variables['u'][index, :, :][:, ngrd.cid]
            ui = ui[ind1_c, range(0, ngrd.cid.shape[0])]*w1_c + ui[ind2_c, range(0, ngrd.cid.shape[0])]*w2_c
            u[counter, :, :] = ui
            vi = d.variables['v'][index, :, :][:, ngrd.cid]
            vi = vi[ind1_c, range(0, ngrd.cid.shape[0])]*w1_c + vi[ind2_c, range(0, ngrd.cid.shape[0])]*w2_c
            v[counter, :, :] = vi
            if not tides:
                tempi = d.variables['temp'][index, :, :][:, ngrd.nid]
                tempi = tempi[ind1_n, range(0, ngrd.nid.shape[0])]*w1_n + tempi[ind2_n, range(0, ngrd.nid.shape[0])]*w2_n
                temp[counter, :, :] = tempi
                salti = d.variables['salinity'][index, :, :][:, ngrd.nid]
                salti = salti[ind1_n, range(0, ngrd.nid.shape[0])]*w1_n + salti[ind2_n, range(0, ngrd.nid.shape[0])]*w2_n
                salt[counter, :, :] = salti

        else:
            u[counter, :, :] = d.variables['u'][index, :, :][:, ngrd.cid]
            v[counter, :, :] = d.variables['v'][index, :, :][:, ngrd.cid]
            if not tides:
                temp[counter, :, :] = d.variables['temp'][index, :, :][:, ngrd.nid]
                salt[counter, :, :] = d.variables['salinity'][index, :, :][:, ngrd.nid]
        #tracer[:,:,counter] = d.variables.get('tracer_c')[:][index, :, ngrd.nid]

    out_nest.close()
    d.close()


def main():
    '''Make nesting files (fvcom to fvcom nesting)'''
    ngrd, fileList, output_file, start_time, stop_time, vert_interp, tides = parse_command_line()
    make_nestfile(ngrd=ngrd,
                  fileList=fileList,
                  output_file = output_file,
                  start_time  = start_time,
                  stop_time   = stop_time,
                  vertical_interpolation=vert_interp,
                  tides = tides)


def parse_command_line():
    '''Parse command line arguments'''
    parser = ArgumentParser(description='Make nesting files (fvcom to fvcom nesting)')
    parser.add_argument("-ngrd", "-n", help='Path to ngrd mat-file', default="ngrd.mat")
    parser.add_argument("-fileList", "-f", help='Path to fileList for source simulation', default="fileList.txt")
    parser.add_argument("-outfile", "-o", help='Name of output file (with path)')
    parser.add_argument("-start_time", "-s", help='First point in time to include (yyyy-mm-dd-hh)')
    parser.add_argument("-stop_time", "-e", help='Last point in time to include (yyyy-mm-dd-hh)')
    parser.add_argument("-vertical_interpolation", "-v", help='Set to True to do vertical interpolation')
    parser.add_argument("-tides", "-t", help='Set to True to not read temp and salinity', default = False)
    args = parser.parse_args()
    return args.ngrd, args.fileList, args.outfile, args.start_time, args.stop_time, args.vertical_interpolation, args.tides


def check_grid(ngrd, fl):
    '''
    Kill routine if an invalid ngrd is provided
    '''
    if len(fl.path)<1:
        raise InputError('\n---------------\nNo files available in filelist for the specified period.\n'+\
                         'Make sure that the mother model covers the time period you wish to model.')
    M     = fvg.FVCOM_grid(fl.path[0], verbose = False)

    # Extract nodes and cells that ngrd refers to
    xn_nest, yn_nest = [M.x[ngrd.nid], M.y[ngrd.nid]]
    xc_nest, yc_nest = [M.xc[ngrd.cid], M.yc[ngrd.cid]]

    # Compare them
    max_node_diff    = np.max(np.sqrt((xn_nest-ngrd.xn[:,0])**2+(yn_nest-ngrd.yn[:,0])**2))
    max_cell_diff    = np.max(np.sqrt((xc_nest-ngrd.xc[:,0])**2+(yc_nest-ngrd.yc[:,0])**2))

    # Allow some roundoff-error
    if max_node_diff > 1 or max_cell_diff > 1:
        M.plot_grid()
        plt.triplot(ngrd_new.xn[:,0], ngrd_new.yn[:,0], ngrd_new.nv, c = 'k', label = 'nest grid')
        plt.scatter(M.x[ngrd.nid], M.y[ngrd.nid], c = 'r', zorder = 20, label = 'positions where you try to extract\ndata from the mother model')
        plt.legend(loc = 'upper right')
        plt.draw()
        raise InputError(f'\nPositions in {ngrd.filepath} does not correspond to the positions you try to read from {M.filepath}.\n'+
                         'You are most likely using the wrong ngrd-file, or you have provided a wrong filelist for this nest.')

class InputError(Exception):
    pass

if __name__ == '__main__':
    main()
