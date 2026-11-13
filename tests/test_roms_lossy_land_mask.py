"""ROMS land masks must survive lossy compression.

MET's NorKyst v3 files are written with lossy compression (``ncks --ppc``),
which stores some water points of ``mask_rho``/``mask_u``/``mask_v`` as
1.0000038 instead of 1. Reading the mask must still count those as water,
or a nesting zone over them fails with a LandError.
"""

import numpy as np
from netCDF4 import Dataset

from fvtools.grid.roms_grid import METNorKystV3

LOSSY_ONE = np.float32(1.0000038)


def _write_roms_grid(path, ny=4, nx=5, nlev=3):
    lossy = np.ones((ny, nx))
    lossy[0, 0] = 0.0                       # land
    lossy[1, 1] = LOSSY_ONE                 # water, stored lossily
    lossy[2, 3] = 1.0 - 3.8e-6              # water, stored lossily below 1
    with Dataset(path, "w") as nc:
        nc.createDimension("eta_rho", ny)
        nc.createDimension("xi_rho", nx)
        nc.createDimension("s_rho", nlev)
        nc.createDimension("s_w", nlev + 1)
        rho = ("eta_rho", "xi_rho")
        for name in ("lon_rho", "lon_u", "lon_v"):
            nc.createVariable(name, "f8", rho)[:] = np.tile(np.arange(nx, dtype=float), (ny, 1))
        for name in ("lat_rho", "lat_u", "lat_v"):
            nc.createVariable(name, "f8", rho)[:] = 60 + np.tile(np.arange(ny, dtype=float)[:, None], (1, nx))
        nc.createVariable("h", "f8", rho)[:] = 100.0
        nc.createVariable("angle", "f8", rho)[:] = 0.0
        nc.createVariable("s_rho", "f8", ("s_rho",))[:] = np.linspace(-0.833, -0.167, nlev)
        nc.createVariable("Cs_r", "f8", ("s_rho",))[:] = np.linspace(-0.833, -0.167, nlev)
        nc.createVariable("s_w", "f8", ("s_w",))[:] = np.linspace(-1, 0, nlev + 1)
        nc.createVariable("Cs_w", "f8", ("s_w",))[:] = np.linspace(-1, 0, nlev + 1)
        nc.createVariable("hc", "f8")[:] = 100.0
        nc.createVariable("Vstretching", "i4")[:] = 5
        for name in ("mask_rho", "mask_u", "mask_v"):
            nc.createVariable(name, "f4", rho)[:] = lossy
    return lossy


def test_lossy_water_points_are_water(tmp_path):
    path = tmp_path / "norkyst_grid.nc"
    stored = _write_roms_grid(path)

    grid = METNorKystV3()
    grid.load_grid_from_nc(str(path))

    land = stored == 0
    for name in ("rho_mask", "u_mask", "v_mask"):
        mask = np.asarray(getattr(grid, name))
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, land, err_msg=name)
