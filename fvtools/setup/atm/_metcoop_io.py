"""MetCoOp-specific bounded spatial reads; shared THREDDS connection logic."""

from __future__ import annotations

import numpy as np

from fvtools.util.thredds_io import (
    SourcePool, is_retryable_netcdf_error as is_retryable_error,
)


def read_masked(dataset, field, prefix, mask):
    """Read a bounding rectangle from the source, then apply its original mask.

    ``prefix`` contains the time and optional singleton axes. The trailing two
    source axes must correspond to the original 2-D spatial mask. The boolean
    mask keeps original C-order point ordering and masked-array semantics.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError('AROME spatial mask must be a nonempty 2-D boolean mask')
    rows, columns = np.nonzero(mask)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(columns.min()), int(columns.max()) + 1
    key = (*prefix, slice(r0, r1), slice(c0, c1))
    subset = dataset.variables[field][key]
    return subset[..., mask[r0:r1, c0:c1]]
