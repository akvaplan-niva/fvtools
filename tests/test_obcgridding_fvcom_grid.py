"""Regression tests for the FVCOM_grid-based obcgridding I/O workflow."""

import numpy as np

from fvtools.prepro.nesting import obcgridding as module


def test_main_uses_fvcom_grid_for_input_and_output(tmp_path, monkeypatch):
    """Output must have no obsolete NS records and retain the new connectivity."""
    initial_x = np.array([0., 1., 1., 0.])
    initial_y = np.array([0., 0., 1., 1.])
    initial_tri = np.array([[0, 1, 2], [0, 2, 3]])
    original_nodestrings = [np.array([0, 1, 2])]
    created = []
    writes = []

    class FakeFVCOMGrid:
        def __init__(self, meshname):
            created.append(meshname)
            self.x = initial_x.copy()
            self.y = initial_y.copy()
            self.tri = initial_tri.copy()
            self.nodestrings = [ns.copy() for ns in original_nodestrings]
            self.obc_nodes = [0, 1, 2]

        def write_2dm(self, name=None):
            writes.append((name, self.x.copy(), self.y.copy(), self.tri.copy(),
                           list(self.obc_nodes), list(self.nodestrings)))

    expanded_x = np.array([0., 1., 1., 0., 0.5])
    expanded_y = np.array([0., 0., 1., 1., -1.])
    expanded_tri = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]])
    calls = []

    def fake_obcgridding(x, y, tri, nodestrings, res, rows):
        calls.append((x.copy(), y.copy(), tri.copy(), [ns.copy() for ns in nodestrings], res, rows))
        return np.column_stack((expanded_x, expanded_y)), expanded_tri

    monkeypatch.setattr(module, 'FVCOM_grid', FakeFVCOMGrid)
    monkeypatch.setattr(module, 'obcgridding', fake_obcgridding)

    mesh = tmp_path / 'domain.v2.2dm'
    assert module.main(mesh, res=200, rows=3) is None

    assert created == [str(mesh)]
    assert len(calls) == 1
    x, y, tri, nodestrings, res, rows = calls[0]
    np.testing.assert_array_equal(x, initial_x)
    np.testing.assert_array_equal(y, initial_y)
    np.testing.assert_array_equal(tri, initial_tri)
    np.testing.assert_array_equal(nodestrings[0], original_nodestrings[0])
    assert (res, rows) == (200, 3)

    assert len(writes) == 1
    output_name, x, y, tri, obc_nodes, nodestrings = writes[0]
    assert output_name == str(tmp_path / 'domain.v2_with_nest')
    np.testing.assert_array_equal(x, expanded_x)
    np.testing.assert_array_equal(y, expanded_y)
    np.testing.assert_array_equal(tri, expanded_tri)
    assert obc_nodes == []
    assert nodestrings == []


def test_open_obc_geometry_unchanged():
    x = np.array([0., 1., 0.])
    y = np.array([0., 0., 1.])
    triangles = np.array([[0, 1, 2]])
    points, new_tri = module.obcgridding(x, y, triangles, [np.array([0, 1])], 0.1, 1)
    assert points.shape == (5, 2)
    assert new_tri.shape == (3, 3)
    np.testing.assert_array_equal(new_tri[0], triangles[0])
    np.testing.assert_allclose(points[-2:], [[0., -0.1], [1., -0.1]])


def test_circular_obc_geometry_unchanged():
    x = np.array([0., 1., 0.])
    y = np.array([0., 0., 1.])
    triangles = np.array([[0, 1, 2]])
    points, new_tri = module.obcgridding(x, y, triangles, [np.array([0, 1, 2, 0])], 0.1, 1)
    assert points.shape == (6, 2)
    assert new_tri.shape == (7, 3)
    np.testing.assert_array_equal(new_tri[0], triangles[0])
    assert new_tri.min() >= 0
    assert new_tri.max() < len(points)
