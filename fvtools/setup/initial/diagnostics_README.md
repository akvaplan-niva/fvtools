# Restart balance diagnostics

This diagnostic reads a complete, unmodified FVCOM restart and uses
`FVCOM_grid.T` to reconstruct the Cartesian part of `EXTEL_EDGE`'s nodal
control-volume transport divergence. It does not fix initial conditions.

```python
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.setup.initial.diagnostics import diagnose_restart

M = FVCOM_grid('M.npy')
result = diagnose_restart(M, 'input/grid_B_restart.nc')
print(result.attrs)
M.plot_field(result['eta_tendency_from_transport'].values)
# Or using xarray/matplotlib for cell-centered fields:
M.plot_contour(result['geostrophic_proxy_acceleration'].values)
```

`eta_tendency_from_transport` is an **implied** d(zeta)/dt, not a residual.
For an actual continuity residual, provide independent parent-model zeta
tendency on the *same FVCOM nodes* and at the same time. For example:

```python
from fvtools.setup.initial.diagnostics import finite_difference_eta_tendency
parent_tendency = finite_difference_eta_tendency(zeta_parent_before,
                                                zeta_parent_after, seconds=3600)
result = diagnose_restart(M, 'input/grid_B_restart.nc',
                          eta_tendency=parent_tendency)
M.plot_field(result['continuity_residual'].values)
```

The finite difference should ideally bracket the restart time; parent model
fields must be horizontally interpolated to FVCOM nodes first. Check the
reference epoch and time units. The OBC and land nodes are intentionally
masked in nodal continuity because FVCOM handles these separately.

`geostrophic_proxy_acceleration` estimates `|-g grad(zeta)+f(v,-u)|` using
piecewise-linear triangle gradients. It is **not** a full momentum residual;
it ignores tendency, advection, friction, density and other forces. In a
strongly ageostrophic flow it can be large even for a physically consistent
initial state. If the mesh does not contain valid latitude, supply `coriolis`
explicitly in s^-1; otherwise the proxy is omitted.

The 2D/3D check uses cell sigma-interface thickness fractions; it reads the
3D velocities in chunks. The output separates `mean_2d_3d_error` (internal
consistency) from `surface_shear` (real shear relative to the 3D mean).
`obc_outward_transport_m3_s` is an integral of the outward normal transport
on exterior mesh edges where both endpoints have `ISONB==2`.

Assumptions: full, uncropped FVCOM projected mesh, matching node/element
numbering and geometry; metric x/y in metres. The transport calculation
matches the Cartesian part of FVCOM's `EXTEL_EDGE`; builds compiled with
`SPHERICAL` include extra corrections and require separate validation. The
momentum proxy is diagnostic only. Nothing is written to the restart.
