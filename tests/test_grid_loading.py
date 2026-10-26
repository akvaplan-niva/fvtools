"""Fast grid-loader tests using a four-node mesh and M.npy fixture."""

from pathlib import Path

import numpy as np

from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.grid.fvgrid import read_sms_mesh

SMALL_2DM = Path(__file__).resolve().parent / "data" / "mini_nodestrings.2dm"


def test_sms_mesh_node_and_nodestring_order():
    tri, node_ids, x, y, z, _, nodestrings = read_sms_mesh(
        str(SMALL_2DM), nodestrings=True
    )

    np.testing.assert_array_equal(tri, [[0, 1, 2], [0, 2, 3]])
    np.testing.assert_array_equal(node_ids, [1, 2, 3, 4])
    np.testing.assert_array_equal(x, [500000, 500100, 500100, 500000])
    np.testing.assert_array_equal(y, [7700000, 7700000, 7700100, 7700100])
    np.testing.assert_array_equal(z, np.zeros(4))
    assert len(nodestrings) == 2
    np.testing.assert_array_equal(nodestrings[0], [0, 1])
    np.testing.assert_array_equal(nodestrings[1], [2, 3])


def test_fvcom_grid_loads_sms_mesh_and_boundaries():
    grid = FVCOM_grid(str(SMALL_2DM))
    assert grid.node_number == 4
    assert grid.cell_number == 2
    np.testing.assert_array_equal(grid.tri, [[0, 1, 2], [0, 2, 3]])
    assert grid.obc_nodes == [0, 1, 2, 3]
    assert [list(nodes) for nodes in grid.nodestrings] == [[0, 1], [2, 3]]


def test_fvcom_grid_loads_small_npy(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))

    assert grid.casename == "mini_nodestrings"
    assert grid.reference.lower() == "epsg:32633"
    assert grid.node_number == 4
    assert grid.cell_number == 2
    assert grid.obc_nodes == [0, 1, 2, 3]
    assert [list(nodes) for nodes in grid.nodestrings] == [[0, 1], [2, 3]]
    np.testing.assert_allclose(grid.h, [5.0, 6.0, 7.0, 8.0])
    np.testing.assert_allclose(grid.siglev[:, 0], 0.0)
    np.testing.assert_allclose(grid.siglev[:, -1], -1.0)
    np.testing.assert_allclose(grid.siglay, [[-0.25, -0.75]] * 4)
