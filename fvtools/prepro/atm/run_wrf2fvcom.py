#!/global/apps/python/2.7.3/bin/python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import atm.atm_grids as wrf
import grid.fvcom_grd as fv
import atm.interp_tools as interp


def main():
    '''Make atmospheric forcing files.'''
    gridfile, savepath, start, stop = parse_command_line()
    
    # Read FVCOM grid
    FVCOM_grd = fv.FVCOM_grid(gridfile)

    # Read WRF grid
    WRF_grd = wrf.WRF_grid()

    # Create Nearest4 interpolation coefficients
    print('Creating nearest4 coefficients for interpolation')
    N4 = interp.nearest4(FVCOM_grd, WRF_grd)

    # Do interpolation and write to file
    interp.create_nc_forcing_file(savepath, FVCOM_grd)
    
    print('\nInterpolating and dumping to nc-file')
    interp.wrf2fvcom(N4, outfile=savepath, start_time=start, stop_time=stop)
    interp.wrf_prec2fvcom(N4, outfile=savepath, start_time=start, stop_time=stop)
    interp.wrf_fluxes2fvcom(N4, outfile=savepath, start_time=start, stop_time=stop)

def parse_command_line():
    ''' 
    Parse command line arguments
    '''

    parser = ArgumentParser(description='Make FVCOM atmospheric forcing files from WRF data')
    parser.add_argument("-gridfile", "-g", help='Name of matlab grid file with path (e.g. M.mat, M.npy)', 
                          default = 'M.mat')
    parser.add_argument("-outfile", "-o", help='Name of output file (with path)')
    parser.add_argument("-start_time", "-s", help='First point in time to include (yyyy-mm-dd-hh)')
    parser.add_argument("-stop_time", "-e", help='Last point in time to include (yyyy-mm-dd-hh)')
    args = parser.parse_args()

    return args.gridfile, args.outfile, args.start_time, args.stop_time


if __name__ == '__main__':
    main()



