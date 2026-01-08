import netCDF4 as nc
import numpy as np
from fvtools.util.time import date2num 

class InputGrid:
    def __init__(self, LON, LAT, M, landmask=None):
        """
        M is FVCOM_grid object
        landmask: (1=ocean, 0=land) , dimensions matching LAT and LON
        """
        # Store the coordinate arrays
        self.LON = LON
        self.LAT = LAT
        
        # Store the M object itself
        self.M = M

        # gamma is the angle for transforming U,V from LAT/LON to Northing/Easting coordinates
        self._gamma = None
        if landmask is None:
            # Creates an array of 1s (ocean) with the same shape as LON
            self.landmask = np.ones(self.LON.shape)
        else:
            self.landmask = landmask
        
        # Use the M object to calculate projected coordinates        
        self.x, self.y = self.M.Proj(self.LON, self.LAT)
        
        # xex is expected as we are using arome_interpolators which takes this as an input for masking puposes if I am not wrong. 
        self.xex = self.x 
        
        # Calculate cell centers 
        self.x_center = 0.5 * (self.x[:-1, :-1] + self.x[1:, 1:])
        self.y_center = 0.5 * (self.y[:-1, :-1] + self.y[1:, 1:])

    def reproject_new_coords(self, new_lon, new_lat): 
        # This may not be needed now! I have added this just in case. 
        return self.M.Proj(new_lon, new_lat)

    def crop_grid(self, xlim, ylim):
        '''Returns mask for the standard grid'''
        return (self.x >= xlim[0]) & (self.x <= xlim[1]) & \
               (self.y >= ylim[0]) & (self.y <= ylim[1])

    def crop_extended(self, xlim, ylim):
        '''Returns mask used by fv_domain_mask_ex, dont ask me why, but it is needed in class for the code to work'''
        return self.crop_grid(xlim, ylim)

    @property
    def gamma(self):
        """Find the angle for conversion of u,v in Polar to northign/easting"""
        if self._gamma is None:
            # Shift North to find the True North vector in projected space
            x_n, y_n = self.M.Proj(self.LON, self.LAT + 0.01)
            dx = x_n - self.x
            dy = y_n - self.y
            self._gamma = np.arctan2(dx, dy)
        return self._gamma

class SurfaceForcingGenerator:
    def __init__(self, default_interp=None, convert_wind=True):
        """
        default_interp: if the  weight indices object for interpolation is uniform across files/varaibles
        """
        self.default_interp = default_interp
        self.convert_wind = convert_wind

    def apply_interpolation(self, data_2d, interp_obj, mode='node'):
        """Applies weights from a specific interp object"""
        # Crop data to the FVCOM domain using the mask in the specific object
        flat_data = data_2d[interp_obj.fv_domain_mask]
        
        # Route based on the attribute names in the provided object
        if mode == 'rad':
            idx, coef = interp_obj.nindex_rad, interp_obj.ncoef_rad
        elif mode == 'cell':
            idx, coef = interp_obj.cindex, interp_obj.ccoef
        else:
            idx, coef = interp_obj.nindex, interp_obj.ncoef
            
        return np.sum(flat_data[idx] * coef, axis=1)

    def process(self, input_files, output_path):
        var_routing = {
            'tas':   {'fv_name': 'air_temperature',   'type': 'node'},
            'uas':   {'fv_name': 'uwind_speed',       'type': 'cell'},
            'vas':   {'fv_name': 'vwind_speed',       'type': 'cell'},
            'rsds':  {'fv_name': 'short_wave',        'type': 'rad'},
            'rlds':  {'fv_name': 'long_wave',         'type': 'rad'},
            'psl':   {'fv_name': 'air_pressure',      'type': 'node'}            
        }

        # Determine grid size from the first available interp object
        first_key = list(input_files.keys())[0]
        sample_interp = input_files[first_key]['interp']
        node_count = len(sample_interp.FVCOM_grd.x)
        cell_count = len(sample_interp.FVCOM_grd.xc)

        # Create Output File
        out_ds = nc.Dataset(output_path, 'w', format='NETCDF4')
        out_ds.createDimension('time', None)
        out_ds.createDimension('node', node_count)
        out_ds.createDimension('nele', cell_count)

        # Initialize Variables
        out_ds.createVariable('time', 'f4', ('time',))
        for var_id, config in var_routing.items():
            dim = 'nele' if config['type'] == 'cell' else 'node'
            out_ds.createVariable(config['fv_name'], 'f4', ('time', dim))
        
        # Special variables or derived varaibles not in var_routing
        out_ds.createVariable('precip', 'f4', ('time', 'node'))
        out_ds.createVariable('evap', 'f4', ('time', 'node'))

        # Get Time data from 'tas' , here it is assumed the time is uniform, but if some varaibles are taken from other data source, input files may differ in time. This is going to be the challenge when dealign with data from multiple sources. Also its good to look inside 'from fvtools.util.time' . this module is important for the reference time for FVCOM. 
        with nc.Dataset(input_files['tas']['path'], 'r') as sample:
            num_steps = len(sample.variables['time'])
            raw_times = sample.variables['time'][:]
            time_units = sample.variables['time'].units

        #  Processing Loop in time
        for t in range(num_steps):
            print(f"Processing step {t+1}/{num_steps}", end='\r')
            
            # Time Conversion using your time.py date2num
            dt = nc.num2date(raw_times[t], time_units)
            out_ds.variables['time'][t] = date2num(dt.year, dt.month, dt.day, dt.hour)

            # Iterating through variables
            for var_id, config in var_routing.items():
                if var_id in input_files:
                    f_info = input_files[var_id]


                    # --- WIND ROTATION BEFORE INTERPOLATION ---
                    if self.convert_wind and var_id in ['uas', 'vas']:                        
                        if var_id == 'uas':
                            with nc.Dataset(input_files['uas']['path'], 'r') as u_src, \
                                 nc.Dataset(input_files['vas']['path'], 'r') as v_src:
                                
                                # Load raw 2D geographic data
                                u_geo_2d = np.squeeze(u_src.variables['uas'][t, :, :])
                                v_geo_2d = np.squeeze(v_src.variables['vas'][t, :, :])
                                
                                # Get the grid object for rotation
                                grid_obj = f_info.get('grid')
                                if grid_obj is not None:
                                    # ROTATE ON 2D GRID FIRST
                                    gamma = grid_obj.gamma
                                    u_grid_2d = u_geo_2d * np.cos(gamma) - v_geo_2d * np.sin(gamma)
                                    v_grid_2d = u_geo_2d * np.sin(gamma) + v_geo_2d * np.cos(gamma)
                                    
                                    # Now interpolate the already-rotated components
                                    u_final = self.apply_interpolation(u_grid_2d, input_files['uas']['interp'], mode='cell')
                                    v_final = self.apply_interpolation(v_grid_2d, input_files['vas']['interp'], mode='cell')
                                    
                                    # Store them in the output file
                                    out_ds.variables['uwind_speed'][t, :] = u_final
                                    out_ds.variables['vwind_speed'][t, :] = v_final
                                else:
                                    # Fallback if no grid provided
                                    out_ds.variables['uwind_speed'][t, :] = self.apply_interpolation(u_geo_2d, f_info['interp'], mode='cell')
                                    out_ds.variables['vwind_speed'][t, :] = self.apply_interpolation(v_geo_2d, f_info['interp'], mode='cell')
                        
                        # Skip 'vas' and 'uas' coming in the following steps when convert_wind  is True
                        continue
                        
                    with nc.Dataset(f_info['path'], 'r') as src:
                        #data = src.variables[var_id][t, :, :]
                        data = np.squeeze(src.variables[var_id][t, :, :])
                        # Pass the specific interp object for this variable
                        result = self.apply_interpolation(data, f_info['interp'], mode=config['type'])
                        
                        # Apply conversions
                        if var_id == 'tas': result -= 273.15
                        elif var_id == 'psl': result /= 100.0
                        
                        out_ds.variables[config['fv_name']][t, :] = result

            # Precipitation as a derived variables (check for the arithmatic, here everything is just added , sign conventions not checked )
            p_sum = np.zeros(node_count)
            for p_var in ['prra', 'prsn', 'licalvf']:
                if p_var in input_files:
                    f_info = input_files[p_var]
                    with nc.Dataset(f_info['path'], 'r') as src:
                        #p_data = src.variables[p_var][t, :, :]
                        p_data = np.squeeze(src.variables[p_var][t, :, :])                                               
                        p_sum += self.apply_interpolation(p_data, f_info['interp'], mode='node')
            out_ds.variables['precip'][t, :] = p_sum / 1000.0

            # Evaporation, if not entered in input file, it creates a zero vector 
            if 'evap' in input_files:
                f_info = input_files['evap']
                with nc.Dataset(f_info['path'], 'r') as src:
                    #e_data = src.variables['evap'][t, :, :]
                    e_data = np.squeeze(src.variables['evap'][t, :, :])
                    out_ds.variables['evap'][t, :] = self.apply_interpolation(e_data, f_info['interp'], mode='node')
            else:
                out_ds.variables['evap'][t, :] = 0.0

        out_ds.close()