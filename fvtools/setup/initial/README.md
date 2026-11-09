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

## Optional parent-driven continuity correction

The ROMS, FVCOM and CMEMS/GLORYS initializers can optionally adjust the
barotropic velocity so the child restart's interior-node continuity tendency
matches the parent model's sea-surface-height tendency. The derivative is
computed at the restart time from the three nearest distinct parent snapshots
using a quadratic time interpolant (centered when possible, one-sided near an
endpoint).

Enable it only when initializing velocities:

```python
import fvtools.setup.initial.roms as roms_restart
import fvtools.setup.initial.fvcom as fvcom_restart
import fvtools.setup.initial.cmems as cmems_restart

roms_restart.main(..., uv=True, continuity_correction=True)
fvcom_restart.main(..., speed=True, continuity_correction=True)
cmems_restart.main(..., uv=True, continuity_correction=True)
```

The option is disabled by default. The default maximum velocity increment is
0.25 m/s; projection limits can be passed through `continuity_options={...}`.
The projection constrains interior nodes only. Every cell touching an exterior
mesh edge is fixed, preserving boundary-edge transport, and the same velocity
increment is added to every sigma layer to preserve vertical shear. It does not
enforce OBC-node continuity or momentum balance. A failed solvability,
residual, 2D/3D consistency or correction-speed check aborts the atomic update.
Treat this as an experimental initialization correction and inspect the
reported residuals and velocity increments before production use.
