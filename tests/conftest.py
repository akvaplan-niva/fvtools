"""Small reusable grids and optional externally supplied real-world fixtures."""

from pathlib import Path
import os

import numpy as np
import pytest


TESTS_DIR = Path(__file__).resolve().parent
SMALL_2DM = TESTS_DIR / "data" / "mini_nodestrings.2dm"


@pytest.fixture(scope="session")
def reference_paths():
    """Resolve the paired mesh supplied for regression testing.

    The 31 MB object-array M.npy is deliberately *not* tracked in Git.
    Load it only when it comes from a trusted source (NumPy uses pickle).
    """
    names = (
        ("FVTOOLS_TEST_2DM", "v2_with_nodestrings_with_nest.2dm"),
        ("FVTOOLS_TEST_M_NPY", "M.npy"),
    )
    paths = []
    for env, filename in names:
        path = Path(os.environ.get(env, str(TESTS_DIR / "reference" / filename)))
        if not path.is_file():
            pytest.skip(f"Optional reference file missing: {path} (set {env})")
        paths.append(path)
    return tuple(paths)


@pytest.fixture
def small_grid_file(tmp_path):
    """Make a self-contained, small M.npy compatible with FVCOM_grid."""
    from fvtools.grid.fvcom_grd import FVCOM_grid

    grid = FVCOM_grid(str(SMALL_2DM))
    tri, x, y, nodestrings = grid.tri, grid.x, grid.y, grid.nodestrings
    siglev = np.tile([0.0, -0.5, -1.0], (len(x), 1))
    siglay = (siglev[:, :-1] + siglev[:, 1:]) / 2
    info = {"casename": "mini_nodestrings", "reference": "epsg:32633"}
    payload = {
        "x": x,
        "y": y,
        "lon": np.array([15.0, 15.01, 15.01, 15.0]),
        "lat": np.array([69.0, 69.0, 69.001, 69.001]),
        "h": np.array([5.0, 6.0, 7.0, 8.0]),
        "nv": tri,
        "siglay": siglay,
        "siglev": siglev,
        "nodestrings": [np.asarray(ns) for ns in nodestrings],
        "obc_nodes": [node for ns in nodestrings for node in ns],
        "reference": "epsg:32633",
        "info": info,
    }
    path = tmp_path / "M.npy"
    np.save(path, payload)
    return path
