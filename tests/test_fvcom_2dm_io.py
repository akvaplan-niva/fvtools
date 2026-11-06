"""Regression tests for the SMS/2dm I/O now owned by FVCOM_grid."""

from pathlib import Path

import numpy as np
import pytest

from fvtools.grid.fvcom_grd import FVCOM_grid


def _mesh(nodestrings):
    x = np.array([0.0, 100.0, 100.0, 0.0])
    y = np.array([0.0, 0.0, 100.0, 100.0])
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    obc_nodes = sorted(set().union(*(set(int(n) for n in ns) for ns in nodestrings))) if nodestrings else []
    mesh = FVCOM_grid(x=x, y=y, tri=tri, obc_nodes=obc_nodes)
    mesh.nodestrings = [np.asarray(ns, dtype=np.int64) for ns in nodestrings]
    return mesh


@pytest.mark.parametrize("nodestrings", [
    [],
    [[0, 1, 2]],
    [[0, 1, 2, 0]],
    [[0, 1], [2, 3]],
])
def test_2dm_roundtrip(tmp_path, nodestrings):
    original = _mesh(nodestrings)
    stem = tmp_path / "a.mesh.with.periods"
    original.write_2dm(name=stem)
    path = Path(str(stem) + ".2dm")
    assert path.is_file()
    loaded = FVCOM_grid(str(path))

    np.testing.assert_array_equal(loaded.tri, original.tri)
    np.testing.assert_allclose(loaded.x, original.x)
    np.testing.assert_allclose(loaded.y, original.y)
    assert len(loaded.nodestrings) == len(nodestrings)
    for actual, expected in zip(loaded.nodestrings, nodestrings):
        np.testing.assert_array_equal(actual, expected)
    if not nodestrings:
        assert not any(line.startswith("NS ") for line in path.read_text().splitlines())


def test_long_nodestring_split_across_ns_records(tmp_path):
    # A nodestring with >10 nodes must be correctly reconstructed.
    n = 13
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 100.0 * np.cos(theta)
    y = 100.0 * np.sin(theta)
    tri = np.array([[0, i, i + 1] for i in range(1, n-1)], dtype=np.int64)
    boundary = np.arange(n, dtype=np.int64)
    mesh = FVCOM_grid(x=x, y=y, tri=tri, obc_nodes=boundary.tolist())
    mesh.nodestrings = [boundary]
    stem = tmp_path / "long"
    mesh.write_2dm(stem)
    text = (tmp_path / "long.2dm").read_text()
    assert text.count("\nNS  ") == 2
    loaded = FVCOM_grid(str(tmp_path / "long.2dm"))
    np.testing.assert_array_equal(loaded.nodestrings[0], boundary)


def test_nonconsecutive_nd_ids(tmp_path):
    # Read the file's actual ND IDs rather than relying on line positions.
    source = """MESH2D
MESHNAME "nonsequential"
E3T 1 40 30 20 1
ND 40 0 0 0.00000001
ND 20 100 0 0.00000001
ND 30 0 100 0.00000001
NS  40 20 -30 1
"""
    path = tmp_path / "nonsequential.2dm"
    path.write_text(source)
    mesh = FVCOM_grid(str(path))
    np.testing.assert_array_equal(mesh.tri, [[0, 2, 1]])
    np.testing.assert_array_equal(mesh.nodestrings[0], [0, 1, 2])
    np.testing.assert_array_equal(mesh.x, [0, 100, 0])


def test_unknown_node_id_raises(tmp_path):
    source = """MESH2D
E3T 1 1 2 9 1
ND 1 0 0 0
ND 2 1 0 0
ND 3 0 1 0
"""
    path = tmp_path / "invalid.2dm"
    path.write_text(source)
    # The error is meaningful regardless of the first letter's case.
    with pytest.raises(ValueError, match="(?i)unknown node ID"):
        FVCOM_grid(str(path))
