# fvtools examples

The notebooks here are examples, not packaged command-line entry points.

- `make_roms_forced_experiment.ipynb`: FVCOM grid preparation, ROMS nesting and `fvtools.setup.initial.roms` restart, river and atmospheric forcing.
- `identify_rivers_download_temperatures.ipynb`: river outlets and temperature climatology. Its imports from `fvtools.setup.rivers.NVEforcing` remain valid.

## CMEMS/GLORYS or NWSHELF + TPXO (nesting)

```python
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.setup.download.cmems import download_glorys, download_nwshelf
import fvtools.setup.nesting.cmems as cmems_nest
import fvtools.setup.nesting.tpxo as tpxo_nest

M = FVCOM_grid('M.npy')

# Choose one source:
download_glorys(M, '2024-04-01', '2024-04-30', outfile='input/cmems/glorys.nc')
# download_nwshelf(M, '2024-10-01', '2024-10-03', outfile='input/cmems/nwshelf.nc')

# The same nesting interpolator reads either combined file:
cmems_nest.main('M.npy', 'ngrd.npy', 'input/cmems/glorys.nc', 'input/casename_nest.nc', latlon=True)
tpxo_nest.main('input/casename_nest.nc', harmonics_files={'zeta': 'tpxo/h.nc'})
```

Both downloaders produce one local NetCDF with `thetao`, `so`, `uo`, `vo`, `zos` and static `deptho`, using the variable names expected by `fvtools.setup.nesting.cmems`. NWSHELF additionally retains `wo`, `ubar` and `vbar`; the latter are the 2D barotropic current components. The NWSHELF output uses timestamps common to all downloaded hourly datasets.

`cmems_nest` interpolates the source timestamps without time interpolation. Add TPXO only once afterwards. For TPXO currents provide both U/V harmonic files and `tpxo_bathymetry='tpxo/grid.nc'` (containing `hu/hv`); projected FVCOM nesting currently supports elevation-only TPXO.

## CMEMS/GLORYS + TPXO (restart)

```python
import fvtools.setup.initial.cmems as cmems_restart
import fvtools.setup.initial.tpxo as tpxo_restart

cmems_restart.main(restartfile='input/casename_restart.nc', fvcom_grid='M.npy',
                   glorys_file='input/cmems/glorys.nc', uv=True, latlon=True,
                   time_method='nearest')
tpxo_restart.main('input/casename_restart.nc', harmonics_files={'zeta': 'tpxo/h.nc'})
```

Here the FVCOM restart file must already exist. Add `restart_date='YYYY-MM-DD-HH'` to the CMEMS call to create it. For TPXO U/V currents, supply U/V harmonic files and TPXO bathymetry; its velocity rotation follows GLORYS restart metadata, or an explicit `latlon` argument if necessary.
