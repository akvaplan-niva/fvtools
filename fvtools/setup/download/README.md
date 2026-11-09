# External forcing downloads

- `fvtools.setup.download.cmems.download_glorys`: download GLORYS daily fields and static bathymetry into one combined NetCDF.
- `fvtools.setup.download.cmems.download_nwshelf`: download NWSHELF hourly SSH, temperature, salinity, 3D currents, 2D barotropic currents and static bathymetry into one combined NetCDF.
- `fvtools.setup.download.era5.main`: download ERA5 atmospheric forcing data.

Both CMEMS download functions produce a combined file containing the core variable names expected by `fvtools.setup.nesting.cmems`: `zos`, `thetao`, `so`, `uo`, `vo` and `deptho`.

NWSHELF additionally retains `wo`, `ubar` and `vbar`. Its 3D products have a shorter historical coverage than some 2D products, so the merged file contains only timestamps common to all requested dynamic datasets. Large 4D variables are copied in blocks to limit memory use.

ERA5 post-processing lives in `fvtools.setup.atm.era5_processing`.
