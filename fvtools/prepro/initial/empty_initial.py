import netCDF4
import datetime
import numpy as np
from fvtools.grid.tools import date2num

# Make an empty restartfile
def make_initial_file(M, initial_time, obc_type = 0):
    '''
    Empty restartfile for this experiment
    - M:            FVCOM_grid
    - initial_time: Date string 'yyyy-mm-dd-hh'
    - obc_type:     1 (fvcom2fvcom), 2 (??), 3 (??)
    '''
    print(f"- Create {M.casename}_initial.nc\n")
    nums   = [int(number) for number in initial_time.split('-')]
    fvtime = date2num([datetime.datetime(nums[0], nums[1], nums[2], nums[3], tzinfo = datetime.timezone.utc)])

    requested_by_restart = {
        'zeta': (('time', 'node'), 'single'),
        'salinity': (('time', 'siglay', 'node'), 'single'),
        'temp': (('time', 'siglay', 'node'), 'single'),
        'iint': (('time', ), 'int32'),
        'ua': (('time', 'nele'), 'single'),
        'va': (('time', 'nele'), 'single'),
        'u': (('time', 'siglay', 'nele'), 'single'),
        'v': (('time', 'siglay', 'nele'), 'single'),
        'tauc': (('time', 'nele'), 'single'),
        'omega': (('time', 'siglev', 'node'), 'single'),
        'ww': (('time', 'siglay', 'nele'), 'single'),
        'viscofm': (('time', 'siglay', 'nele'), 'single'),
        'viscofh': (('time', 'siglay', 'node'), 'single'),
        'km': (('time', 'siglev', 'node'), 'single'),
        'kh': (('time', 'siglev', 'node'), 'single'),
        'kq': (('time', 'siglev', 'node'), 'single'),
        'q2': (('time', 'siglev', 'node'), 'single'),
        'q2l': (('time', 'siglev', 'node'), 'single'),
        'l': (('time', 'siglev', 'node'), 'single'),
        'cor': (('nele', ), 'single'),
        'cc_sponge': (('nele', ), 'single'),
        'et': (('time', 'node', ), 'single'),
        'tmean1': (('siglay', 'node'), 'single'),
        'smean1': (('siglay', 'node'), 'single'),
        'obc_nodes': (('nobc', ), 'int32'),
        'obc_type': (('nobc', ), 'int32'),
        'rho1': (('time', 'siglay', 'node'), 'single'),
        'rmean1': (('siglay', 'node'), 'single'),
        'zice': (('siglay', 'node'), 'single')
        }

    grid_fields = {
        'x': (('node',), 'single'),
        'y': (('node',), 'single'),
        'xc': (('nele',), 'single'),
        'yc': (('nele',), 'single'),
        'lat': (('node',), 'single'),
        'lon': (('node', ), 'single'),
        'latc': (('nele',), 'single'),
        'lonc': (('nele',), 'single'),
        'h': (('node',), 'single'),
        'h_center': (('nele',), 'single'),
        'nv': (('three', 'nele'), 'int32'),
        'siglay': (('siglay', 'node'), 'single'),
        'siglev': (('siglev', 'node'), 'single'),
        'siglay_center': (('siglay', 'nele'), 'single'),
        'siglev_center': (('siglev', 'nele'), 'single')
        }

    aliases = {'nv': 'tri',
               'h_center': 'hc'}

    # Write to the initial file
    with netCDF4.Dataset(f'{M.casename}_initial.nc', 'w') as initial:
        timedim  = initial.createDimension('time', 0)
        nodedim  = initial.createDimension('node', len(M.x))
        celldim  = initial.createDimension('nele', len(M.xc))
        threedim = initial.createDimension('three', 3)
        levdim   = initial.createDimension('siglev', M.siglev.shape[-1])
        laydim   = initial.createDimension('siglay', M.siglev.shape[-1]-1)
        datestr  = initial.createDimension('DateStrLen', 26)
        nobc     = initial.createDimension('nobc', len(M.obc_nodes))

        time             = initial.createVariable('time', 'single', ('time',))
        time.units       = 'days since 1858-11-17 00:00:00'
        time.format      = 'modified julian day (MJD)'
        time.time_zone   = 'UTC'

        Itime            = initial.createVariable('Itime', 'int32', ('time',))
        Itime.units      = 'days since 1858-11-17 00:00:00'
        Itime.format     = 'modified julian day (MJD)'
        Itime.time_zone  = 'UTC'

        Itime2           = initial.createVariable('Itime2', 'int32', ('time',))
        Itime2.units     = 'msec since 00:00:00'
        Itime2.time_zone = 'UTC'

        # Create variables in the netCDF file
        for key in requested_by_restart.keys():
            initial.createVariable(key, requested_by_restart[key][1], requested_by_restart[key][0])

        for key in grid_fields.keys():
            initial.createVariable(key, grid_fields[key][1], grid_fields[key][0])

        # Dump grid info to the netCDF file
        for key in grid_fields.keys():
            if key not in aliases.keys():
                if key in ['siglev', 'siglay', 'siglev_center', 'siglay_center']:
                    initial[key][:] = getattr(M, key).T
                else:
                    initial[key][:] = getattr(M, key)

            else:
                if key =='nv':
                    initial[key][:] = getattr(M, aliases[key]).T + 1 # Offset for Fortran indexing
                else:
                    initial[key][:] = getattr(M, aliases[key])

        # Set initial values to zero, give obc_nodes and set OBC type to desired value
        for key in requested_by_restart.keys():
            if key == 'obc_type':
                initial[key][:] = obc_type

            elif key == 'obc_nodes':
                initial[key][:] = np.array(M.obc_nodes) + 1 # Offset for Fortran indexing

            else:
                initial[key][:] = 0

        # Set initial time
        initial['time'][:] = fvtime[0]
        initial['Itime'][:] = int(fvtime[0])
        initial['Itime2'][:] = int((fvtime[0]-int(fvtime[0]))*24*60*60*1000)
    return f'{M.casename}_initial.nc'