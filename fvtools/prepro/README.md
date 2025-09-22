# Scripts used to prepare an experiment
These should really be co-located with nesting and atmospheric forcing

[[_TOC_]]

## Overview
### BuildCase
Reads .2dm file, checks that the triangles are legal (and remove illegal ones), interpolates bathymetry to the mesh, smooths to comply with hydrostatic constitency criteria, writes .2dm files and returns a M.npy-mesh file.

Example use:
```python
import pre_pro.BuildCase as bc
bc.main('M.npy', 'depth_data.npy')
```

### BuildRivers
Creates river forcing
```python
import pre_pro.BuildRivers as br

# If you prepare a new experiment (ROMS nested)
br.main('2018-01-01-00', '2018-02-01-00')

# If you prepare a fvcom-nested experiment
br.main('2018-01-01-00', '2018-02-01-00', temp = 'fvcom_mother_temperatures.npy')

# Temperatures for big models are stored on the Stokes and Betzy
#   Stokes: /data/FVCOM/Setup_Files/Rivers/Raw_Temperatures/
#   Betzy:  /cluster/shared/NS9067K/apn_backup/FVCOM/Setup_Files/Rivers/Raw_Temperatures/
```

### JulianTimeElev
Reads a M.npy file and interpolates tidal harmonics from the TPXO database to the FVCOM-obc. 

The routine reads the start data (year, month, day) and number of days to create forcing for (e.g. 365). Needs an FVCOM_grid object as input.
```python
import pre_pro.JulianTimeElev as je 
from grid.fvcom_grd import FVCOM_grid
M = FVCOM_grid('M.npy')
je.main(M, 2018, 10, 1, 300, netcdf_name = 'tides.nc')
```

There is something odd about the interpolation routines in the module that accesses tidal data and interpolates it to the mesh (pyTMD), it is therefore encouraged to check the quality of the forcing before moving on

```python
je.ModifyForcing('tides.nc', 'M.npy')
```

### interpol_restart
Interpolates inital fields from a FVCOM mother to you restartfile
```python
import pre_pro.interpol_restart as ir 
ir.main(childfn = 'my_experiment_restart_0001.nc', filelist = 'filelist.txt', vinterp = True, speed = True)
```
Switches_
- vinterp: True to vertically interpolate fields to the child model
- speed:   True if you also want to include speed


### interpol_roms_restart
Interpolated initial fields from a ROMS model to your restartfile
```python
import pre_pro.interpol_roms_restart as ir 
ir.main('my_experiment_restart_0001.nc', 'MET-NK', uv = True, latlon = True)
```
Switches:
- uv: True if you want to interpolate a velocity field to the mesh
- latlon: True if you run FVCOM in spherical model, will otherwise correct currents for meridional convergence in UTM coordinates

