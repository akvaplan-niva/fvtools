"""2DM regression tests for the roms-phase6a FVCOM_grid mixins."""

from pathlib import Path

import numpy as np
import pytest

from fvtools.grid.fvcom_grd import FVCOM_grid


SMALL_2DM = Path(__file__).resolve().parent / 'data' / 'mini_nodestrings.2dm'


def test_load_existing_2dm_preserves_boundary_order():
    grid = FVCOM_grid(str(SMALL_2DM))
    np.testing.assert_array_equal(grid.tri, [[0, 1, 2], [0, 2, 3]])
    assert grid.obc_nodes == [0, 1, 2, 3]
    assert [ns.tolist() for ns in grid.nodestrings] == [[0, 1], [2, 3]]


def test_roundtrip_existing_2dm(tmp_path):
    source = FVCOM_grid(str(SMALL_2DM))
    out = tmp_path / 'grid_output.2dm'
    source.write_2dm(str(out))  # Accept a filename WITH its extension.
    assert out.exists()
    assert not out.with_suffix('.2dm.2dm').exists()
    reloaded = FVCOM_grid(str(out))
    np.testing.assert_array_equal(reloaded.tri, source.tri)
    np.testing.assert_allclose(reloaded.x, source.x)
    np.testing.assert_allclose(reloaded.y, source.y)
    assert [ns.tolist() for ns in reloaded.nodestrings] == [[0, 1], [2, 3]]


def test_nonconsecutive_node_ids_can_be_read(tmp_path):
    mesh = tmp_path / 'nonconsecutive.2dm'
    mesh.write_text(
        'MESH2D\n'
        'E3T 1 17 34 8 1\n'
        'E3T 2 17 8 21 1\n'
        'ND 17 500000 7700000 0\n'
        'ND 34 500100 7700000 0\n'
        'ND 8 500100 7700100 0\n'
        'ND 21 500000 7700100 0\n'
        'NS  17 -34 1\n'
        'NS  8 -21 2\n',
        encoding='utf-8',
    )
    grid = FVCOM_grid(str(mesh))
    np.testing.assert_array_equal(grid.tri, [[0, 1, 2], [0, 2, 3]])
    assert [ns.tolist() for ns in grid.nodestrings] == [[0, 1], [2, 3]]


def test_circular_nodestring_and_multiline_ns_roundtrip(tmp_path):
    # 12 perimeter nodes and an interior node, triangulated as a fan.
    angle = 2 * np.pi * np.arange(12) / 12
    x = np.r_[500000.0 + 100.0 * np.cos(angle), 500000.0]
    y = np.r_[7700000.0 + 100.0 * np.sin(angle), 7700000.0]
    tri = np.array([[i, (i + 1) % 12, 12] for i in range(12)], dtype=int)
    grid = FVCOM_grid(x=x, y=y, tri=tri, obc_nodes=list(range(12)))
    grid.nodestrings = [np.r_[np.arange(12), 0]]
    output = tmp_path / 'circle'
    grid.write_2dm(str(output))

    lines = output.with_suffix('.2dm').read_text(encoding='utf-8').splitlines()
    ns_lines = [line for line in lines if line.startswith('NS ')]
    assert len(ns_lines) == 2  # 13 IDs split across 10-node records.
    reloaded = FVCOM_grid(str(output.with_suffix('.2dm')))
    np.testing.assert_array_equal(reloaded.tri, tri)
    assert [row.tolist() for row in reloaded.nodestrings] == [list(range(12)) + [0]]


def test_no_obc_is_not_written_as_an_ns_record(tmp_path):
    grid = FVCOM_grid(str(SMALL_2DM))
    grid.obc_nodes = []
    output = tmp_path / 'no_obc'
    grid.write_2dm(str(output))
    assert not any(line.startswith('NS ') for line in output.with_suffix('.2dm').read_text().splitlines())
    reloaded = FVCOM_grid(str(output.with_suffix('.2dm')))
    assert reloaded.obc_nodes == []
    assert reloaded.nodestrings == []


def test_unknown_node_id_is_rejected(tmp_path):
    path = tmp_path / 'invalid.2dm'
    path.write_text(
        'MESH2D\nE3T 1 1 2 999 1\n'
        'ND 1 500000 7700000 0\nND 2 500100 7700000 0\n'
        'ND 3 500000 7700100 0\n', encoding='utf-8',
    )
    with pytest.raises(ValueError, match='unknown node ID'):
        FVCOM_grid(str(path))
