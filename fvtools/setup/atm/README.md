# Scripts that interpolate results from a weather model to the FVCOM grid
Atmospheric forcing can be prepared from MetCoOp-AROME or a local ERA5 file.

## read_metCoop
A script that reads weather data from the norwegian met-office thredds server and interpolates the results to a experiment_atm.nc file

```python
import fvtools.setup.atm.read_metCoop as rm 
rm.main('M.npy', 'experiment_atm.nc', '2018-01-01-00', '2018-02-01-00')
```

## read_era5
Interpolate a processed local ERA5 NetCDF (including `lsm`) to an FVCOM atmospheric forcing file:

```python
import fvtools.setup.atm.read_era5 as re5
re5.main('M.npy', 'experiment_era5_atm.nc', 'ERA5.nc', '2023-08-01-00', '2023-08-02-00')
```
