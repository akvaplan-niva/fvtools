# Scripts that interpolate results from a weather model to the FVCOM grid
We currently just have data to force using the metCoop-arome model.

## read_metCoop
A script that reads weather data from the norwegian met-office thredds server and interpolates the results to a experiment_atm.nc file

```python
import fvcom_pytools.atm.read_metCoop as rm 
rm.main('M.npy', 'experiment_atm.nc', '2018-01-01-00', '2018-02-01-00')
```
