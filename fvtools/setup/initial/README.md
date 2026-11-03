# FVCOM initial/restart modules

The public restart modules are:

- `fvtools.setup.initial.fvcom` (FVCOM source)
- `fvtools.setup.initial.roms` (ROMS source)
- `fvtools.setup.initial.cmems` (GLORYS/CMEMS source)
- `fvtools.setup.initial.tpxo` (TPXO barotropic tides)

## GLORYS/CMEMS restart + TPXO

First interpolate a local, combined GLORYS file (`thetao`, `so`, `uo`, `vo`, `zos`, `deptho`) to an existing FVCOM restart. For daily CMEMS data, use `time_method='nearest'` when exact time matching is unavailable:

```python
import fvtools.setup.initial.cmems as cmems_restart
cmems_restart.main(restartfile='input/casename_restart.nc', fvcom_grid='M.npy',
                   glorys_file='input/cmems/glorys.nc', uv=True, latlon=True,
                   time_method='nearest')
```

Then add TPXO to the same file (once). Supplying only `zeta` adds elevation; supplying both `u` and `v` also adds depth-uniform currents, requiring TPXO `hu/hv` bathymetry:

```python
import fvtools.setup.initial.tpxo as tpxo_restart
tpxo_restart.main('input/casename_restart.nc', harmonics_files={'zeta': 'tpxo/h.nc'})
```

`cmems_restart.main` can also create a new restart if passed `restart_date='YYYY-MM-DD-HH'`. Do not add TPXO twice; the update is atomic and checks its marker.
