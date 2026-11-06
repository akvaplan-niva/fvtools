"""Representation of the GLORYS rectilinear grid and model bathymetry."""

import numpy as np
import xarray as xr

from .common.base import BaseGrid


class GLORYS(BaseGrid):
    def __init__(self, ncglorys, *, bathymetry=None):
        """Read dynamic GLORYS data, optionally using the static ``deptho`` field.

        ``ncglorys`` and ``bathymetry`` accept paths or xarray Datasets.
        The static dataset may cover a larger domain than the dynamic one.

        If ``bathymetry`` is omitted, preserve the historical approximation
        based on the deepest wet model z-level for backwards compatibility.
        """
        self._deptho = None
        if isinstance(ncglorys, xr.Dataset):
            self._load_dataset_grid(ncglorys)
        else:
            with xr.open_dataset(ncglorys) as df:
                self._load_dataset_grid(df)

        if bathymetry is not None:
            if isinstance(bathymetry, xr.Dataset):
                self._load_static_bathymetry(bathymetry)
            else:
                with xr.open_dataset(bathymetry) as df:
                    self._load_static_bathymetry(df)

    def _load_dataset_grid(self, df):
        self.depth = np.asarray(df.depth.values, dtype=np.float64)
        self.lon, self.lat = np.meshgrid(
            np.asarray(df.longitude.values), np.asarray(df.latitude.values)
        )
        self.zlevels = np.broadcast_to(
            self.depth[:, None, None],
            (len(self.depth), *self.lon.shape),
        )
        # The salinity mask describes valid 3-D source data, not exact bottom
        # depth; keep it separate from the static model bathymetry.
        salinity = df.so.isel(time=0).transpose('depth', 'latitude', 'longitude')
        self.land_3d = ~np.isfinite(np.asarray(salinity.values))
        if self.land_3d.shape != self.zlevels.shape:
            raise ValueError(
                'GLORYS salinity mask does not match its (depth,lat,lon) grid'
            )

        # A combined download already includes the static bathymetry.
        # Keep bathymetry=... for users of older two-file workflows.
        if 'deptho' in df:
            self._load_static_bathymetry(df)

    def _load_static_bathymetry(self, df):
        if 'deptho' not in df:
            raise ValueError("Static GLORYS dataset has no 'deptho' variable")
        da = df['deptho']
        units = str(da.attrs.get('units', 'm')).lower().strip()
        if units not in ('m', 'meter', 'meters', 'metre', 'metres'):
            raise ValueError(f'GLORYS deptho must be in metres; got {units!r}')

        # Permit singleton auxiliary dimensions (but not extra dimensions
        # carrying variable bathymetry), and keep singleton spatial axes.
        extra = [d for d in da.dims if d not in ('latitude', 'longitude')]
        if any(da.sizes[d] != 1 for d in extra):
            raise ValueError('GLORYS deptho has unexpected non-singleton dimensions')
        da = da.isel({d: 0 for d in extra}, drop=True)
        if set(da.dims) != {'latitude', 'longitude'}:
            raise ValueError('GLORYS deptho must have latitude and longitude dimensions')
        da = da.transpose('latitude', 'longitude')
        for axis in ('latitude', 'longitude'):
            coordinates = np.asarray(da[axis].values, dtype=float)
            if not np.isfinite(coordinates).all():
                raise ValueError(f'Invalid static GLORYS {axis} coordinates')
        da = da.sortby('latitude').sortby('longitude')
        for axis in ('latitude', 'longitude'):
            if np.any(np.diff(np.asarray(da[axis].values, dtype=float)) <= 0):
                raise ValueError(f'Duplicate static GLORYS {axis} coordinates')

        # Match real grid coordinates; a static global file can cover a much
        # larger area than the subset of dynamic forcing data. Reject shifted
        # or different-resolution grids rather than silently interpolating.
        try:
            selected = da.sel(
                latitude=xr.DataArray(self.lat[:, 0], dims='latitude'),
                longitude=xr.DataArray(self.lon[0, :], dims='longitude'),
                method='nearest',
                tolerance=1e-4,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                'Static deptho coordinates do not match the dynamic GLORYS grid'
            ) from exc

        deptho = np.ma.filled(
            np.ma.asarray(selected.values, dtype=np.float64), np.nan
        )
        if deptho.shape != self.lon.shape:
            raise ValueError('Static GLORYS deptho has an incompatible grid shape')
        wet = ~self.land_3d[0]
        invalid_wet = wet & (~np.isfinite(deptho) | (deptho <= 0))
        if invalid_wet.any():
            raise ValueError(
                f'Static GLORYS deptho missing/invalid at '
                f'{np.count_nonzero(invalid_wet)} wet dynamic grid points'
            )
        self._deptho = np.where(wet, deptho, np.nan)

    @property
    def bathymetry_source(self):
        return 'deptho' if self._deptho is not None else 'deepest_wet_z_level'

    @property
    def depth_below_geoid(self):
        """Static model deptho if supplied; otherwise deepest wet z-level."""
        if self._deptho is not None:
            return self._deptho
        wet_levels = np.where(~self.land_3d, self.zlevels, -np.inf)
        deepest = wet_levels.max(axis=0)
        return np.where(np.isfinite(deepest), deepest, np.nan)

    @property
    def land_mask(self):
        """True for land points in the dynamic GLORYS source grid."""
        return np.isnan(self.depth_below_geoid)

    @property
    def h_rho(self):
        """ROMS-compatible name for GLORYS bathymetry (positive down)."""
        return self.depth_below_geoid
