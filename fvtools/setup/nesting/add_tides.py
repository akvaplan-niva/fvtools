from netCDF4 import Dataset
import numpy as np
import sys

from fvcom_pytools.grid import fvcom_grd as fvg
from fvcom_pytools.post_processing import fvcom_make_file_list as mfl
from time import gmtime, strftime

def initial(inifile, tidedirs, casename):
    ''' Adds tides to a restart file made from daily averaged data

        inifile - restart file name
        tidedirs - folders with 2D tidal data
        casename - '''

    print('Create filelist for tidal data')
    fvtime, fvpath, fvind = mfl.make_fileList(casename, tidedirs)
    fvtime = np.array(fvtime)

    print('Read ' + inifile)
    nc = Dataset(inifile, 'r+')
    datime = nc.variables['Itime'][:] + nc.variables['Itime2'][:] / 86400000.
    zeta_da = nc.variables['zeta'][:]
    ua_da = nc.variables['ua'][:]
    va_da = nc.variables['va'][:]
    u_da = nc.variables['u'][:]
    v_da = nc.variables['v'][:]

    i = np.where(fvtime == datime)
    tidefile = fvpath[int(i[0])]
    print('Read ' + tidefile + ' and add tidal and daily averaged data')
    nct = Dataset(tidefile, 'r')
    ttime = nct.variables['Itime'][:] + nct.variables['Itime2'][:] / 86400000.
    i = np.where(ttime == datime)
    i = int(i[0])
    zeta_t = nct.variables['zeta'][i,:]
    ua_t = nct.variables['ua'][i,:]
    va_t = nct.variables['va'][i,:]
    zeta = zeta_t + zeta_da
    ua = ua_da + ua_t
    va = va_da + va_t
    sh = u_da.shape
    u = np.zeros(u_da.shape)
    v = np.zeros(v_da.shape)
    for n in np.arange(sh[1]):
        u[:,n,:] = u_da[:,n,:] + ua_t
        v[:,n,:] = v_da[:,n,:] + va_t
    nct.close()

    print('Write to ' + inifile)
    nc.variables['zeta'][:] = zeta
    nc.variables['ua'][:] = ua
    nc.variables['va'][:] = va
    nc.variables['u'][:] = u
    nc.variables['v'][:] = v

    nc.close()

def nest(nestgrd, nestfile, nestfile_da, tidedirs, casename, dayblock = 30, dt = 1/24):
    ''' Adds tides to a daily averaged nesting file
        nestgrd - nesting grid
        nestfile - name of output nest file
        nestfile_da - existing daily averaged nestring file
        tidedirs - folders with 2D nesting data
        casename -
        dayblock - how many days to process at a time
        dt - timestep in output nest file. '''

    print('Read NEST grid')
    ngrd = fvg.NEST_grid(nestgrd)
    nid = np.squeeze(ngrd.nid)
    cid = np.squeeze(ngrd.cid)
    print('Create filelist for tidal data')
    fvtime, fvpath, fvind = mfl.make_fileList(casename, tidedirs)
    fvtime = np.array(fvtime)
    da = Dataset(nestfile_da, 'r')
    ntime = da.variables['Itime'][:] + da.variables['Itime2'][:] / 86400000.
    print('Create empty nest file from data in daily averaged nest file')
    create_nesting_file(nestfile, nestfile_da)

    dtime = ntime[(ntime >= fvtime[0]) & (ntime <= fvtime[-1])]
    time = fvtime[(fvtime >= dtime[0]) & (fvtime <= dtime[-1])]
    tend = time[-1]

    # Read weights from daily averaged file
    wc = da.variables['weight_cell'][0,:]
    wn = da.variables['weight_node'][0,:]

    nc = Dataset(nestfile, 'r+')

    t1 = time[0]
    nblocks = int(np.ceil((tend - t1) / dayblock))
    nb = 0
    c0 = 0
    while True:
        nb += 1
        if nb > nblocks:
            break
        print('Block ' + str(nb) + ' of ' + str(nblocks))
        t2 = t1 + dayblock
        if t2 > tend:
            t2 = tend
        zeta_t = np.empty((int((t2-t1) / dt), len(nid)))
        ua_t = np.empty((int((t2-t1) / dt), len(cid)))
        va_t = np.empty((int((t2-t1) / dt), len(cid)))
        ttime = np.empty(int((t2-t1) / dt))
        tmp = np.where((fvtime>=t1) & (fvtime<=t2))
        ind = tmp[0]
        ind0 = ind[0]
        i0 = 0

        print('Load data from 2D tidal simulation')
        while True:
            if i0 >= int((t2-t1) / dt):
                break
            file = fvpath[ind0]
            print(file)
            df = Dataset(file, 'r')
            timetmp = df.variables['Itime'][:] + df.variables['Itime2'][:] / 86400000.
            zetatmp = df.variables['zeta'][(timetmp >= t1) & (timetmp < t2),nid]
            uatmp = df.variables['ua'][(timetmp >= t1) & (timetmp < t2),cid]
            vatmp = df.variables['va'][(timetmp >= t1) & (timetmp < t2),cid]
            timetmp = timetmp[(timetmp >= t1) & (timetmp < t2)]
            zeta_t[i0:i0+len(timetmp),:] = zetatmp
            ua_t[i0:i0+len(timetmp),:] = uatmp
            va_t[i0:i0+len(timetmp),:] = vatmp
            ttime[i0:i0+len(timetmp)] = timetmp
            i0 = i0 + len(timetmp)
            ind0 = ind0 + len(timetmp)
            #if ind0 >= len(fvtime):
            #    df.close()
            #    break
            df.close()

        print('Load data from daily average nestingfile')
        datime = da.variables['Itime'][:] + da.variables['Itime2'][:] / 86400000.
        zeta_da = da.variables['zeta'][(datime >= t1) & (datime <= t2), :]
        ua_da = da.variables['ua'][(datime >= t1) & (datime <= t2), :]
        va_da = da.variables['va'][(datime >= t1) & (datime <= t2), :]
        u_da = da.variables['u'][(datime >= t1) & (datime <= t2), :, :]
        v_da = da.variables['v'][(datime >= t1) & (datime <= t2), :, :]
        temp_da = da.variables['temp'][(datime >= t1) & (datime <= t2), :, :]
        salinity_da = da.variables['salinity'][(datime >= t1) & (datime <= t2), :, :]
        datime = datime[(datime >= t1) & (datime <= t2)]
        t1 = t2

        print('Interpolate in time and combine with tidal data')
        zeta_i = time_interp(zeta_da, datime, ttime)
        zeta = zeta_i + zeta_t
        ua_i = time_interp(ua_da, datime, ttime)
        ua = ua_i + ua_t
        va_i = time_interp(va_da, datime, ttime)
        va = va_i + va_t
        u_i = time_interp(u_da, datime, ttime)
        v_i = time_interp(v_da, datime, ttime)
        u = np.zeros(u_i.shape)
        v = np.zeros(v_i.shape)
        sh = u_i.shape
        for n in np.arange(sh[1]):
            u[:,n,:] = u_i[:,n,:] + ua_t
            v[:,n,:] = v_i[:,n,:] + va_t
        temp = time_interp(temp_da, datime, ttime)
        salinity = time_interp(salinity_da, datime, ttime)
        sh_t = temp.shape
        hyw = np.zeros((sh_t[0], sh_t[1]+1, sh_t[2]))
        weight_cell = np.zeros((sh_t[0], sh[2]))
        weight_node = np.zeros((sh_t[0], sh_t[2]))
        for tt in np.arange(sh_t[0]):
            weight_cell[tt,:] = wc
            weight_node[tt,:] = wn

        print('Write to netcdf file')
        counter = np.arange(c0, c0 + len(ttime[ttime > 0.0]))
        nc.variables['time'][counter] = ttime[ttime > 0.0]
        nc.variables['Itime'][counter] = np.floor(ttime[ttime > 0.0])
        nc.variables['Itime2'][counter] = (ttime[ttime > 0.0] - np.floor(ttime[ttime > 0.0])) * 86400000
        nc.variables['zeta'][counter,:] = zeta[ttime > 0.0,:]
        nc.variables['ua'][counter,:] = ua[ttime > 0.0,:]
        nc.variables['va'][counter,:] = va[ttime > 0.0,:]
        nc.variables['u'][counter,:,:] = u[ttime > 0.0,:,:]
        nc.variables['v'][counter,:,:] = v[ttime > 0.0,:,:]
        nc.variables['temp'][counter,:,:] = temp[ttime > 0.0,:,:]
        nc.variables['salinity'][counter,:,:] = salinity[ttime > 0.0,:,:]
        nc.variables['hyw'][counter,:,:] = hyw[ttime > 0.0,:,:]
        nc.variables['weight_cell'][counter,:] = weight_cell[ttime > 0.0,:]
        nc.variables['weight_node'][counter,:] = weight_node[ttime > 0.0,:]
        c0 = c0 + len(ttime)

    nc.close()
    da.close()

def time_interp(var, time_in, time_out):
    """
    Linearly interpolates data in daily averaged nesting file from time_in to time_out

    This code assumes implicitly: time is always first dimension in these files
    """
    var_out = np.nan * np.zeros((len(time_out), *var.shape[1:]))
    for idx, _ in np.ndenumerate(var[0]):
        var_out[(slice(None), ) + idx] = np.interp(time_out, time_in, var[(slice(None), ) + idx])

    return var_out

def create_nesting_file(name, nest_da):
    '''
    Creates empty nc file formatted to fit FVCOM open boundary ocean forcing
    '''
    nc = Dataset(name, 'w', format='NETCDF4')
    da = Dataset(nest_da, 'r')

    # Write global attributes
    # ----
    nc.title       = 'FVCOM Nesting File'
    nc.institution = 'Akvaplan-niva AS'
    nc.source      = 'FVCOM grid (unstructured) nesting file'
    nc.created     = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # Create dimensions
    # ----
    nc.createDimension('time', 0)
    nc.createDimension('node', len(da.variables['x'][:]))
    nc.createDimension('nele', len(da.variables['xc'][:]))
    nc.createDimension('three', 3)
    nc.createDimension('siglay', len(da.variables['siglay'][:]))
    nc.createDimension('siglev', len(da.variables['siglev'][:]))

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

    # Weight coefficients (since we are nesting from ROMS)
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

    # dump the grid metrics
    # ----
    nc.variables['lat'][:]           = da.variables['lat'][:]
    nc.variables['lon'][:]           = da.variables['lon'][:]
    nc.variables['latc'][:]          = da.variables['latc'][:]
    nc.variables['lonc'][:]          = da.variables['lonc'][:]
    nc.variables['x'][:]             = da.variables['x'][:]
    nc.variables['y'][:]             = da.variables['y'][:]
    nc.variables['xc'][:]            = da.variables['xc'][:]
    nc.variables['yc'][:]            = da.variables['yc'][:]
    nc.variables['h'][:]             = da.variables['h'][:]
    nc.variables['h_center'][:]      = da.variables['h_center'][:]
    nc.variables['nv'][:]            = da.variables['nv'][:]
    nc.variables['siglev'][:]        = da.variables['siglev'][:]
    nc.variables['siglay'][:]        = da.variables['siglay'][:]
    nc.variables['siglev_center'][:] = da.variables['siglev_center'][:]
    nc.variables['siglay_center'][:] = da.variables['siglay_center'][:]

    nc.close()
    da.close()
