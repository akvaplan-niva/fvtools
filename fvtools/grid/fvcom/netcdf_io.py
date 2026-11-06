"""FVCOM NetCDF I/O and field processing.

Import this module through :class:`fvtools.grid.fvcom_grd.FVCOM_grid`.
The classes below are implementation mixins, not a separate grid API.

Layers:
    NetCDFSpatialIO: read data and crop mesh dimensions.
    NetCDFSessionIO: file-handle ownership and reuse.
    NetCDFFieldProcessing: convert/mask values, interpolate vertically.
    FVCOMNetCDFIO: validate a request and orchestrate the layers.

The public ``load_netCDF(filepath, field, time, sig=None, *, z=None,
mask_dry=True)`` interface is unchanged.
"""

from contextlib import contextmanager
import os

import numpy as np
from netCDF4 import Dataset

# Tune these after benchmarking representative Olivia NetCDF files.
MAX_TEMPORARY_BYTES = 32 * 1024 * 1024
DENSE_READ_FACTOR = 2
DEFAULT_BLOCK_WIDTH = 4096


class NetCDFSpatialIO:
    """Read NetCDF variables and efficiently subset mesh points."""

    def _crop_indices(self, dimensions):
        """Return global mesh indices, or None on an uncropped grid."""
        if 'node' in dimensions:
            indices = np.asarray(self.cropped_nodes)
        elif 'nele' in dimensions:
            indices = np.asarray(self.cropped_cells)
        else:
            return None
        # FVCOM_grid uses a scalar array as its 'not cropped' sentinel.
        return indices if indices.ndim == 1 else None

    @staticmethod
    def _nc_spatial_dim(dimensions):
        return next((d for d in ('node', 'nele') if d in dimensions), None)

    @staticmethod
    def _nc_vertical_dim(dimensions):
        return next((d for d in ('siglay', 'siglev') if d in dimensions), None)

    @staticmethod
    def _spatial_result_axis(selection, source_axis):
        """Locate the spatial axis after integer-indexed dimensions vanish."""
        return sum(
            not isinstance(selector, (int, np.integer))
            for selector in selection[:source_axis]
        )

    @staticmethod
    def _read_spatial_window(variable, selection, axis, lo, hi):
        """Read one contiguous spatial window, without mutating selection."""
        selectors = list(selection)
        selectors[axis] = slice(lo, hi)
        return variable[tuple(selectors)]

    @classmethod
    def _read_nc_subset(cls, variable, selection, axis, indices):
        """Read an indexed mesh subset using a small number of spatial slices.

        Supports contiguous, dense and sparse indices; preserves original
        order and duplicates. Limits large temporary hyperslab reads.
        """
        indices = np.asarray(indices, dtype=np.intp)
        if indices.ndim != 1:
            raise ValueError('Subgrid indices must be a 1-D array')
        if np.any(indices < 0) or np.any(indices >= variable.shape[axis]):
            raise IndexError('Subgrid indices outside NetCDF spatial dimension')

        result_axis = cls._spatial_result_axis(selection, axis)
        if indices.size == 0:
            return cls._read_spatial_window(variable, selection, axis, 0, 0)

        if indices.size == 1 or np.all(np.diff(indices) == 1):
            return cls._read_spatial_window(
                variable, selection, axis, int(indices[0]), int(indices[-1]) + 1
            )

        unique, inverse = np.unique(indices, return_inverse=True)
        first, last = int(unique[0]), int(unique[-1])
        other_count = np.prod([
            int(size) for i, size in enumerate(variable.shape)
            if i != axis and not isinstance(selection[i], (int, np.integer))
        ], dtype=np.int64)
        bytes_per_spatial_point = int(other_count) * max(int(variable.dtype.itemsize), 8)

        span = last - first + 1
        if (span <= DENSE_READ_FACTOR * unique.size
                and span * bytes_per_spatial_point <= MAX_TEMPORARY_BYTES):
            data = cls._read_spatial_window(variable, selection, axis, first, last + 1)
            return np.take(data, indices - first, axis=result_axis)

        selected = cls._read_sparse_nc_blocks(
            variable, selection, axis, unique, bytes_per_spatial_point,
            result_axis
        )
        # Unique indices were sorted for efficient disk access. Restore the
        # caller-supplied order, including repeated indices.
        return np.take(selected, inverse, axis=result_axis)

    @classmethod
    def _read_sparse_nc_blocks(cls, variable, selection, axis, unique,
                               bytes_per_point, result_axis):
        """Combine bounded windows into one array ordered by unique indices."""
        chunks = variable.chunking() if hasattr(variable, 'chunking') else None
        chunk_width = chunks[axis] if isinstance(chunks, (tuple, list)) else None
        preferred_width = int(chunk_width) if chunk_width is not None else DEFAULT_BLOCK_WIDTH
        maximum_width = max(1, MAX_TEMPORARY_BYTES // max(1, bytes_per_point))
        block_width = max(1, min(preferred_width, maximum_width))

        ids = unique // block_width
        boundaries = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, unique.size]
        parts = []
        for i, j in zip(boundaries[:-1], boundaries[1:]):
            group = unique[i:j]
            lo, hi = int(group[0]), int(group[-1]) + 1
            block = cls._read_spatial_window(variable, selection, axis, lo, hi)
            parts.append(np.take(block, group - lo, axis=result_axis))

        if len(parts) == 1:
            return parts[0]
        concatenate = (
            np.ma.concatenate if any(np.ma.isMaskedArray(p) for p in parts)
            else np.concatenate
        )
        return concatenate(parts, axis=result_axis)

    def _read_nc_variable(self, dataset, name, time, sig_index=None):
        """Read a NetCDF variable, using slices for subgrid I/O.

        Returns (raw data, remaining dimensions). Does not perform
        wet/dry masking, NaN conversion or vertical interpolation.
        """
        variable = dataset[name]
        dimensions = variable.dimensions
        selection = [slice(None)] * len(dimensions)

        if 'time' in dimensions:
            if time is None:
                raise ValueError(f'{name!r} requires a time index')
            selection[dimensions.index('time')] = time

        vertical = self._nc_vertical_dim(dimensions)
        if sig_index is not None and vertical is not None:
            selection[dimensions.index(vertical)] = sig_index

        spatial = self._nc_spatial_dim(dimensions)
        cropped = self._crop_indices(dimensions)
        remaining_dimensions = tuple(
            dim for dim, selector in zip(dimensions, selection)
            if not isinstance(selector, (int, np.integer))
        )
        if spatial is not None and cropped is not None:
            raw = self._read_nc_subset(
                variable, selection, dimensions.index(spatial), cropped
            )
        else:
            raw = variable[tuple(selection)]
        return raw, remaining_dimensions


class NetCDFSessionIO:
    """Own and optionally reuse a single NetCDF Dataset handle."""

    @contextmanager
    def netcdf_session(self):
        """Reuse a single open NetCDF file across sequential load_netCDF calls.

        Usage for movies::

            with grid.netcdf_session():
                animation.save(filename, writer=writer)

        The handle follows filenames, closing the previous file when the
        animation advances to another file, and always closes on exit.
        Ordinary load_netCDF calls still open and close independently.
        """
        if getattr(self, '_nc_session', None) is not None:
            raise RuntimeError('Nested netcdf_session() is not supported')
        session = {'path': None, 'dataset': None}
        self._nc_session = session
        try:
            yield self
        finally:
            try:
                if session['dataset'] is not None:
                    session['dataset'].close()
            finally:
                self._nc_session = None

    @contextmanager
    def _netcdf_handle(self, filepath):
        """Use a session-owned Dataset, if present, else a temporary one."""
        session = getattr(self, '_nc_session', None)
        if session is None:
            with Dataset(filepath, 'r') as dataset:
                yield dataset
            return

        path = os.fspath(filepath)
        if session['dataset'] is None or session['path'] != path:
            if session['dataset'] is not None:
                session['dataset'].close()
                session['dataset'] = None
                session['path'] = None
            session['dataset'] = Dataset(path, 'r')
            session['path'] = path
        yield session['dataset']


class NetCDFFieldProcessing:
    """Mask invalid/dry data and transform its vertical coordinate."""

    @staticmethod
    def _as_array(data):
        """Convert masked/non-finite numeric values to NaN where possible.

        Preserve float32 for unmasked or masked float32 data. For masked
        integers, promote to float so that missing values can be NaN.
        """
        arr = np.ma.asarray(data)
        if arr.dtype.kind in 'fc':
            return np.asarray(np.ma.masked_invalid(arr).filled(np.nan))
        if arr.dtype.kind in 'iub' and np.ma.is_masked(arr):
            return np.asarray(arr.astype(float).filled(np.nan))
        return np.asarray(arr)

    def _get_dry_mask(self, dataset, spatial, time, dry_masks):
        """Load and cache the dry-point mask for one spatial staggering."""
        wet_name = 'wet_nodes' if spatial == 'node' else 'wet_cells'
        if wet_name not in dataset.variables:
            return None
        if wet_name not in dry_masks:
            wet, remaining_dims = self._read_nc_variable(dataset, wet_name, time)
            wet = self._as_array(wet)
            if remaining_dims != (spatial,):
                raise ValueError(
                    f'{wet_name} must have one remaining {spatial!r} dimension; '
                    f'found {remaining_dims}'
                )
            dry_masks[wet_name] = (~np.isfinite(wet)) | (wet == 0)
        return dry_masks[wet_name]

    @staticmethod
    def _apply_wet_mask(data, remaining_dims, dry, name):
        """Broadcast a 1-D dry-point mask over all non-spatial dimensions."""
        if dry is None:
            return data
        spatial = 'node' if 'node' in remaining_dims else 'nele'
        axis = remaining_dims.index(spatial)
        if data.shape[axis] != dry.size:
            raise ValueError(f'Wet/dry mask shape mismatch for {name!r}')
        shape = [1] * data.ndim
        shape[axis] = dry.size
        return np.where(dry.reshape(shape), np.nan, data)

    def _prepare_nc_field(self, raw, remaining_dims, dry, name):
        """Convert NetCDF fill values to NaN and apply a wet/dry mask."""
        data = self._as_array(raw)
        return self._apply_wet_mask(data, remaining_dims, dry, name)

    def _transform_vertical_field(self, data, remaining_dims, name,
                                  sig=None, z=None):
        """Apply a vertical maximum or sigma-to-z interpolation (no I/O)."""
        vertical = self._nc_vertical_dim(remaining_dims)
        if z is None:
            if sig == 'max' and vertical is not None:
                axis = remaining_dims.index(vertical)
                return np.ma.masked_invalid(data).max(axis=axis).filled(np.nan)
            return data

        spatial = self._nc_spatial_dim(remaining_dims)
        if vertical is None or spatial is None or data.ndim != 2:
            raise ValueError(
                f'Cannot interpolate {name!r} to z: expected one vertical '
                'and one spatial dimension'
            )
        data = np.moveaxis(data, remaining_dims.index(vertical), 0)
        depth_attr = {
            ('siglay', 'node'): 'siglayz',
            ('siglay', 'nele'): 'siglayz_uv',
            ('siglev', 'node'): 'siglevz',
            ('siglev', 'nele'): 'siglevz_uv',
        }[(vertical, spatial)]
        depths = -np.asarray(getattr(self, depth_attr)).T
        if data.shape != depths.shape:
            raise ValueError(
                f'Shape mismatch for {name!r}: data {data.shape}, '
                f'sigma depths {depths.shape}'
            )
        return self.interpolate_to_z(data, z, depths=depths)


class FVCOMNetCDFIO(NetCDFSpatialIO, NetCDFSessionIO, NetCDFFieldProcessing):
    """Public NetCDF loading interface used by FVCOM_grid."""

    def _load_prepared_variable(self, dataset, name, time, sig_index,
                                mask_dry, prepared_cache, dry_masks):
        """Read and mask a field, caching only during this public call.

        No frame-to-frame field cache is retained: the same variable name
        may have different values at each time step.
        """
        key = (name, sig_index)
        if key not in prepared_cache:
            raw, dimensions = self._read_nc_variable(
                dataset, name, time, sig_index
            )
            spatial = self._nc_spatial_dim(dimensions)
            dry = None
            if (mask_dry and spatial is not None
                    and 'time' in dataset[name].dimensions
                    and name not in ('wet_nodes', 'wet_cells')):
                dry = self._get_dry_mask(dataset, spatial, time, dry_masks)
            prepared_cache[key] = (
                self._prepare_nc_field(raw, dimensions, dry, name), dimensions
            )
        return prepared_cache[key]

    def load_netCDF(self, filepath, field, time, sig=None, *, z=None,
                    mask_dry=True):
        """Read and process one or more FVCOM fields at one time step.

        The existing public API is unchanged. 'field' may be a name or a
        sequence of names; sequences return a dictionary. 'sig' is a layer
        index, 'max' or None. z is metres below the instantaneous surface.
        All variables and wet/dry masks are read from one Dataset handle.
        When z is specified, self.zeta is updated for this time step.
        """
        if z is not None and sig is not None:
            raise ValueError('Specify either sig or z, not both')
        if isinstance(sig, str) and sig != 'max':
            raise ValueError("sig must be an integer, None or 'max'")
        if sig is not None and not isinstance(sig, (int, np.integer, str)):
            raise TypeError("sig must be an integer, None or 'max'")
        if z is not None and (not np.isscalar(z) or not np.isfinite(z) or z < 0):
            raise ValueError('z must be a nonnegative, finite depth in metres')

        single = isinstance(field, str)
        fields = [field] if single else list(field)
        if not fields or not all(isinstance(name, str) for name in fields):
            raise ValueError('field must be a variable name or a list of names')

        with self._netcdf_handle(filepath) as dataset:
            for name in fields:
                if name not in dataset.variables:
                    raise KeyError(f'Variable {name!r} not found in {filepath}')

            dry_masks = {}
            prepared_cache = {}

            if z is not None:
                if 'zeta' not in dataset.variables:
                    raise KeyError('z interpolation requires zeta in the NetCDF file')
                self.zeta, _ = self._load_prepared_variable(
                    dataset, 'zeta', time, None, mask_dry,
                    prepared_cache, dry_masks
                )

            result = {}
            sig_index = sig if isinstance(sig, (int, np.integer)) else None
            for name in fields:
                data, remaining_dims = self._load_prepared_variable(
                    dataset, name, time, sig_index, mask_dry,
                    prepared_cache, dry_masks
                )
                result[name] = self._transform_vertical_field(
                    data, remaining_dims, name, sig=sig, z=z
                )

        return result[field] if single else result