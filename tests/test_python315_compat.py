"""Python 3.15 support contract, independent from a network connection."""
import sys

import numpy as np
import pytest


def test_numba_smoke_compiles_on_active_python():
    from numba import njit
    import numba
    from packaging.version import Version

    # This assertion intentionally reports an outdated environment before JIT tests.
    assert Version(numba.__version__) >= Version('0.68.0')

    @njit
    def square_and_sum(values):
        total = 0.0
        for value in values:
            total += value * value
        return total

    values = np.array([1., 2., 3.], dtype=np.float64)
    assert square_and_sum(values) == pytest.approx(14.)
    assert square_and_sum.nopython_signatures


def test_python_315_binary_stack_imports():
    if sys.version_info[:2] != (3, 15):
        pytest.skip('Binary-stack validation is specific to CPython 3.15')

    # Import-only smoke: complete functionality is covered by the normal test suite.
    import cartopy  # noqa: F401
    import cftime  # noqa: F401
    import matplotlib  # noqa: F401
    import netCDF4  # noqa: F401
    import pyproj  # noqa: F401
    import scipy  # noqa: F401
    import shapely  # noqa: F401
