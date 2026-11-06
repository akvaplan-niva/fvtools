"""Regression checks for open and circular FVCOM OBC nodestrings."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


obc_mod = _load_module('_test_obc_circular', ROOT / 'fvtools/grid/fvcom/_components/obc.py')
export_mod = _load_module('_test_export_circular', ROOT / 'fvtools/grid/fvcom/_components/export.py')
coast_mod = _load_module('_test_coastline_circular', ROOT / 'fvtools/grid/fvcom/_components/coastline.py')


class Mesh(obc_mod.OBC, export_mod.ExportGrid, coast_mod.CoastLine):
    def __init__(self, tri, node_number, obc_nodes, boundary_edges):
        self.tri = np.asarray(tri, dtype=np.int64)
        self.node_number = node_number
        self.boundary_nodes = np.asarray(boundary_edges, dtype=np.int64)
        self.obc_nodes = list(obc_nodes)
        self.casename = 'test'
        self.x = np.arange(node_number, dtype=float)
        self.y = np.arange(node_number, dtype=float)
        self._full_boundary = np.unique(self.boundary_nodes)

    @property
    def full_model_boundary(self):
        return self._full_boundary


@pytest.fixture
def fan():
    # Four exterior nodes with a centre node (4). All square edges are physical
    # mesh-boundary edges.
    tri = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    boundary = [[0, 1], [1, 2], [2, 3], [3, 0]]
    return tri, boundary


@pytest.fixture
def annulus():
    # 0..3 exterior square, 4..7 inner square. The inner square is a real
    # exposed boundary (hole) unless centre triangles are added.
    tri = []
    for i in range(4):
        a, b = i, (i + 1) % 4
        c, d = 4 + i, 4 + (i + 1) % 4
        tri.extend([[a, b, d], [a, d, c]])
    return tri, [[i, (i + 1) % 4] for i in range(4)] + [
        [4 + i, 4 + (i + 1) % 4] for i in range(4)
    ]


def test_open_chain_order_unchanged(fan):
    tri, boundary = fan
    m = Mesh(tri, 5, [2, 0, 1], boundary)
    assert len(m.nodestrings) == 1
    np.testing.assert_array_equal(m.nodestrings[0], [0, 1, 2])


def test_boundary_loop_is_closed_deterministically(fan):
    tri, boundary = fan
    m = Mesh(tri, 5, [3, 1, 0, 2], boundary)
    np.testing.assert_array_equal(m.nodestrings[0], [0, 1, 2, 3, 0])
    assert m.n_obc == 1
    assert len(m.obc_nodes) == 4


def test_hole_boundary_is_circular(annulus):
    tri, boundary = annulus
    m = Mesh(tri, 8, [6, 4, 7, 5], boundary)
    np.testing.assert_array_equal(m.nodestrings[0], [4, 5, 6, 7, 4])


def test_interior_ring_works_if_unambiguous(annulus):
    tri, boundary = annulus
    tri += [[4+i, 4+(i+1)%4, 8] for i in range(4)]
    # Outer rim only; the OBC ring is now interior mesh edges.
    m = Mesh(tri, 9, [4, 5, 6, 7], boundary[:4])
    np.testing.assert_array_equal(m.nodestrings[0], [4, 5, 6, 7, 4])


def test_ambiguous_interior_loop_requires_explicit_nodestring(annulus):
    tri, boundary = annulus
    tri += [[4, 5, 6], [4, 6, 7]]  # diagonal 4--6 creates shortcuts
    m = Mesh(tri, 8, [4, 5, 6, 7], boundary[:4])
    with pytest.raises(ValueError, match='explicitly ordered'):
        _ = m.nodestrings
    # Existing explicit-nodestring setter can disambiguate safely.
    m.nodestrings = [np.array([4, 5, 6, 7, 4])]
    np.testing.assert_array_equal(m.nodestrings[0], [4, 5, 6, 7, 4])


def test_invalid_nodes_fail(fan):
    tri, boundary = fan
    m = Mesh(tri, 5, [1, 10], boundary)
    with pytest.raises(ValueError, match='Invalid OBC'):
        _ = m.nodestrings


def test_isolated_node_cannot_form_loop(fan):
    tri, boundary = fan
    m = Mesh(tri, 5, [4], boundary)
    with pytest.raises(ValueError, match='explicitly ordered'):
        _ = m.nodestrings


def test_mixed_open_and_closed(fan, annulus):
    # Build separate non-touching components, each with its own path type.
    inner_tri, inner_boundary = annulus
    tri = inner_tri + [[8, 9, 12], [9, 10, 12], [10, 11, 12], [11, 8, 12]]
    boundary = inner_boundary + [[8, 9], [9, 10], [10, 11], [11, 8]]
    m = Mesh(tri, 13, [8, 9, 10, 4, 5, 6, 7], boundary)
    assert [ns.tolist() for ns in m.nodestrings] == [
        [4, 5, 6, 7, 4], [8, 9, 10],
    ]


def test_write_obc_does_not_duplicate_loop_node(fan, tmp_path):
    tri, boundary = fan
    m = Mesh(tri, 5, [0, 1, 2, 3], boundary)
    name = tmp_path / 'OBC.dat'
    m.write_obc(str(name))
    content = name.read_text().splitlines()
    assert content[0] == 'OBC Node Number = 4'
    assert content[1:] == ['1 1 1', '2 2 1', '3 3 1', '4 4 1']


def test_closed_obc_does_not_add_fake_coastline_endpoints(fan):
    tri, boundary = fan
    m = Mesh(tri, 5, [0, 1, 2, 3], boundary)
    assert m.coastline_nodes.size == 0


def test_open_obc_keeps_coastline_endpoints(fan):
    tri, boundary = fan
    m = Mesh(tri, 5, [0, 1, 2], boundary)
    assert set(map(int, m.coastline_nodes)) == {0, 2, 3}


def test_nodestrings_reset_on_obc_assignment(fan):
    tri, boundary = fan
    m = Mesh(tri, 5, [0, 1], boundary)
    assert m.nodestrings[0].tolist() == [0, 1]
    m.obc_nodes = [0, 1, 2, 3]
    assert m.nodestrings[0].tolist() == [0, 1, 2, 3, 0]


def test_2dm_filename_suffix_is_not_duplicated(fan, tmp_path):
    # The writer now belongs to ExportGrid, not fvgrid. Assert the actual
    # output format instead of mocking the removed fvgrid.write_2dm call.
    tri, boundary = fan
    m = Mesh(tri, 5, [0, 1, 2, 3], boundary)
    path = tmp_path / 'B_grid.2dm'
    m.write_2dm(name=str(path))

    assert path.is_file()
    assert not (tmp_path / 'B_grid.2dm.2dm').exists()
    ns_records = [
        line for line in path.read_text(encoding='utf-8').splitlines()
        if line.startswith('NS ')
    ]
    # The final -1 closes the circular nodestring, followed by the
    # boundary number. The writer must preserve that convention.
    assert ns_records == ['NS  1 2 3 4 -1 1']
