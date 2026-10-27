"""FVCOM river forcing output: NetCDF, namelist, and QC plots.

The numerical data and NetCDF variables are unchanged. Existing public
FVCOM_rivers.dump(), .write_namelist(), .fix_nordic() and
BuildRivers.show_forcing() APIs remain available.
"""
import os
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import matplotlib.pyplot as plt

from fvtools.prepro._buildrivers.constants import version


class RiverOutputMixin:
    def dump(self, path='input/riverdata.nc'):
        """
        Write the riverdata.nc file for river forcing
        """
        # Initialize file
        print(f'Writing {path}')
        with netCDF4.Dataset(os.fspath(path), 'w') as d:
            # Set dimensions
            d.createDimension('time', None)
            d.createDimension('rivers', len(self.xriv))
            d.createDimension('DateStrLen', 26)
            d.createDimension('namelen', 80)
    
            # Add netcdf information
            d.source      = 'Akvaplan-niva BuildRiver, version '+str(version)
            try:
                user = os.getlogin()
            except:
                try:
                    user = os.getcwd().split('home/')[1].split('/')[0]
                except:
                    user = 'unknown user'

            
            d.history     = f'Created {datetime.now().strftime('%Y-%m-%d at %H:%M h')} by  {user}'
            d.description = 'River forcing (temperature and runoff) for FVCOM 4.x'
    
            # Create variables:
            # - time
            time = d.createVariable('time', 'single', ('time',))
            time.long_name   = 'time'
            time.units       = 'days since ' + str(datetime(1858, 11, 17, 0, 0, 0))
            time.format      = 'modified julian day (MJD)'
            time.time_zone   = 'UTC'
    
            # - Itime
            Itime = d.createVariable('Itime', 'int32', ('time',))
            Itime.long_name   = 'integer days'
            Itime.units       = 'days since ' + str(datetime(1858, 11, 17, 0, 0, 0))
            Itime.format      = 'modified julian day (MJD)'
            Itime.time_zone   = 'UTC'
    
            # - Itime2
            Itime2           = d.createVariable('Itime2', 'int32', ('time',))
            Itime2.long_name = 'integer milliseconds'
            Itime2.units     = 'msec since 00:00:00'
            Itime2.time_zone = 'UTC'
    
            # - river_flux
            flux             = d.createVariable('river_flux', 'single', ('time','rivers'))
            flux.long_name   = 'river runoff volume flux, m**-3 s**-1'
            flux.units       = 'm^3s^-1'
    
            # - river_temp
            temp             = d.createVariable('river_temp', 'single', ('time','rivers'))
            temp.long_name   = 'river runoff temperature'
            temp.units       = 'Celsius'
    
            # - river_salt
            salt             = d.createVariable('river_salt', 'single', ('time','rivers'))
            salt.long_name   = 'river runoff salinity'
            salt.units       = 'PSU'
    
            # - river_names
            names            = d.createVariable('river_names', 'S1', ('rivers', 'namelen'))
    
            # Dump data:
            salt[:]   = np.zeros(self.RiverTemp.shape)
            temp[:]   = self.RiverTemp
            flux[:]   = self.RiverTransport
            time[:]   = self.model_time
            Itime[:]  = np.floor(self.model_time)
            Itime2[:] = np.round((self.model_time - np.floor(self.model_time)) * 60 * 60 * 1000, decimals = 0)*24
    
            # Dump river names
            # --> Make sure that each rivername has 80 character
            names._Encoding = 'ascii'
            for i, name in enumerate(self.river_names):
                if len(name) > 80:
                    this_name = name[:80]
                else:
                    this_name  = name + (80-len(name))*' '
                this_name = self.fix_nordic(this_name)
                names[i,:] = np.array(this_name, dtype = 'S80')

    def fix_nordic(self, this_name):
        '''
        FVCOM does not accept norwegian letters
        '''
        this_name = this_name.replace('å','a')
        this_name = this_name.replace('Å','A')
        this_name = this_name.replace('ø','o')
        this_name = this_name.replace('Ø','O')
        this_name = this_name.replace('æ','e')
        this_name = this_name.replace('Æ','E')
        return this_name

    def write_namelist(self, namelist = 'input/RiverNamelist.nml', riverfile = 'riverdata.nc'):
        """
        Write a namelist to accompany the netCDF file
        """
        difference = np.round(-np.diff(self.M.siglev_c, axis = 1), 6)
        with open(namelist, 'w+') as f:
            print(f'  - Writing namelist for {len(self.river_names)} rivers')
            for i, river in enumerate(self.river_names):
                river = self.fix_nordic(river)
                f.write(' &NML_RIVER\n')
                f.write(f" RIVER_NAME = '{river}',\n")
                f.write(f" RIVER_FILE = '{riverfile}',\n")
                f.write(f' RIVER_GRID_LOCATION = {self.mesh_location[i]+1},\n')
                vertical_dist = np.array2string(
                        difference[self.mesh_location[i],:], 
                        separator = ' ', 
                        edgeitems = 6,
                        precision = 5, 
                        floatmode = 'fixed'
                    ).replace('\n','')[1:-1]
                f.write(f' RIVER_VERTICAL_DISTRIBUTION = {vertical_dist}\n')
                f.write('/\n')


def write_forcing_files(forcing, directory='input'):
    """Write matching FVCOM files to one directory and return their paths.

    Default `input/` calls old no-argument methods, preserving compatibility
    with existing integrations and mock objects. Other destinations are
    opt-in and require the new optional path argument to `dump()`.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    nc_path = folder / 'riverdata.nc'
    nml_path = folder / 'RiverNamelist.nml'
    if folder == Path('input'):
        forcing.dump()
        forcing.write_namelist()
    else:
        forcing.dump(nc_path)
        forcing.write_namelist(nml_path)
    return nc_path, nml_path



def show_forcing(obj, M, Rivers, save_dir=None):
    """
    Simple figures to see that the routine got the basics right
    """
    plt.figure()
    M.georeference()
    plt.plot(obj.x_land, obj.y_land, 'k.', label = 'land nodes', zorder = 1)
    plt.scatter(
        obj.xriv, obj.yriv, 
        s = np.mean(obj.RiverTransport, axis = 0), 
        c = np.mean(obj.RiverTransport, axis = 0), 
        zorder = 5, 
        transform = Rivers.crs
        )
    plt.title('Average transport')
    plt.axis('equal')
    plt.colorbar(label = r'm$^3$ s$^{-1}$')
    if save_dir is not None:
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(folder / 'river_mean_transport.png', dpi=160, bbox_inches='tight')
    plt.show(block = False)

    plt.figure()
    M.georeference()
    plt.scatter(obj.xriv, obj.yriv, obj.RiverTemp.max(axis = 0), c = obj.RiverTemp.max(axis = 0), cmap = 'inferno', zorder = 5, transform = Rivers.crs)
    plt.title('Max temperature in model period')
    plt.axis('equal')
    plt.colorbar(label = 'degrees celcius')
    if save_dir is not None:
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(folder / 'river_max_temperature.png', dpi=160, bbox_inches='tight')
    plt.show(block = False)
