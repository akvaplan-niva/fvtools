# Experimental FVCOM initial continuity projection

`fvtools.setup.initial.continuity` is a **read-only diagnostic plus an optional
copy-on-write restart correction**. It uses the same Cartesian half-edge
finite-volume divergence geometry as FVCOM's `EXTEL_EDGE` (through `M.T`).

**Not a generally balanced restart solver.** Only the *interior-node* mass
continuity constraint is imposed. It does **not** enforce the full momentum
balance, OBC nodal continuity, spherical or wet/dry geometry, energy stability,
or stability of a subsequent FVCOM integration. No existing restart is edited.
Use it as a controlled experiment on the Cartesian, fully wet Grid B case.

## Mathematical definition

For cell `j`, `D_j = mean(h+zeta)` over its three vertices,
`Q_j = D_j * (ua_j,va_j)`. The half-edge operator `B` returns the same nodal
mass-flux sum as the Cartesian `EXTEL_EDGE`, before special boundary handling.
For each *interior* node `i`, require

```
   - B[Q + delta_Q]_i / ART1_i  =  (d zeta / d t)_parent,i
```

Minimize `sum_j A_j D_j |delta_u_j|^2` with
`delta_Q_j = D_j * delta_u_j` (`A_j`: triangle area).
This is the barotropic kinetic-energy **norm of the correction**, *not* a
constraint that total kinetic energy must decrease. The momentum/energy
changes are reported for review.

**Boundary safeguards:** Any triangle touching an exterior mesh edge, whether
open or closed, has `delta_u = delta_v = 0`. Therefore its OBC edge flux
remains exactly unchanged. Boundary nodal continuity equations are **not**
solved. If interior targets cannot be met with those restrictions, or the
velocity correction exceeds the safety threshold, the function raises an
exception and writes nothing. It never clips a correction.

## Jupyter example, using Grid B snapshots

Both snapshots must be on the **same FVCOM node ordering**, from the **same
parent source**, exactly 3600 seconds apart, with the first snapshot matching
the time in the restart. A one-hour finite difference represents an *average*
tendency over the interval, not the instantaneous tendency at the first time.
A centered difference using snapshots bracketing the restart is preferable.

```python
from netCDF4 import Dataset
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.setup.initial.diagnostics import finite_difference_eta_tendency
from fvtools.setup.initial.continuity import correct_restart_continuity

M = FVCOM_grid('M.npy')
with Dataset('input/grid_B_initial_v2.nc') as before, \
        Dataset('SmartValues_initial.nc') as after:
    target = finite_difference_eta_tendency(
        before['zeta'][0, :], after['zeta'][0, :], seconds=3600)

report = correct_restart_continuity(
    FVCOM_grid=M,
    restartfile='input/grid_B_initial_v2.nc',
    outputfile='input/grid_B_initial_continuity.nc',  # NEW path only
    eta_tendency=target,
    max_velocity_correction=0.25,  # m/s; fail if exceeded
)

for name, value in report.attrs.items():
    print(name, value)
```

To preview **without writing anything**:

```python
from fvtools.setup.initial.continuity import project_initial_continuity
with Dataset('input/grid_B_initial_v2.nc') as ds:
    report = project_initial_continuity(
        M, zeta=ds['zeta'][0,:], ua=ds['ua'][0,:],
        va=ds['va'][0,:], eta_tendency=target)
```

For QC, plot `report['initial_continuity_residual']`,
`report['corrected_continuity_residual']` on a common color scale, and
`report['velocity_correction_speed']` on the **element** grid. The reported
RMS residuals are weighted by nodal control-volume area. Evaluate the
`barotropic_kinetic_energy_change_pct` and momentum proxies as well.

The new restart includes **all** source NetCDF variables/attributes, except:
`ua`, `va`, `u`, and `v` at the selected timestep are updated by the same
horizontal velocity correction for all sigma layers. Thus vertical shear and
2D/3D depth-mean consistency are preserved (within NetCDF precision).
`zeta`, temperature, salinity and all other fields remain unchanged.

If the solver rejects the target, do not relax its safety thresholds blindly:
examine the estimated parent `d(zeta)/dt`, boundary conditions, chosen timestep,
and the reported velocity scale. An initial continuity correction can reduce a
spurious fast-mode adjustment, but cannot guarantee geostrophic/hydrostatic
balance or eliminate a 3D nesting instability.

## Tests

```powershell
uv run --no-sync python -m pytest tests/test_initial_continuity.py -q
```
