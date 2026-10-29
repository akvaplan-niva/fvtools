"""Regression coverage when extracting FVCOM TGE and physics mixins.

All analytic tests use the repository's four-node/two-cell M.npy fixture.
The final test also uses the optional 48,803-node reference grid.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.grid.fvcom_grd import (
    AnglesAndPhysics,
    FVCOM_grid,
    PropertiesFromTGE,
)


def test_vector_rotation_roundtrip_and_speed(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    u = np.array([2.0, -3.0, 4.0])
    v = np.array([1.0, 5.0, -2.0])
    angle = np.array([-0.2, 0.0, 0.5])
    projected_u, projected_v = grid.rotate_vector_to_utm(u, v, angle)
    restored_u, restored_v = grid.rotate_vector_from_utm(
        projected_u, projected_v, angle
    )
    np.testing.assert_allclose(restored_u, u, atol=1e-12)
    np.testing.assert_allclose(restored_v, v, atol=1e-12)
    np.testing.assert_allclose(
        np.hypot(projected_u, projected_v), np.hypot(u, v), atol=1e-12
    )
    # Positive quarter-turn is the documented geographic -> UTM convention.
    pu, pv = grid.rotate_vector_to_utm(1.0, 0.0, np.pi / 2)
    np.testing.assert_allclose([pu, pv], [0.0, -1.0], atol=1e-12)


def test_node_and_cell_utm_angles_have_expected_dimensions(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    # The synthetic fixture's lon/lat are illustrative; derive consistent
    # geodetic coordinates before checking the convergence-angle methods.
    grid.lon, grid.lat = grid.Proj(grid.x, grid.y, inverse=True)
    node_angles = grid.node_utm_angle
    cell_angles = grid.cell_utm_angle
    assert node_angles.shape == (grid.node_number,)
    assert cell_angles.shape == (grid.cell_number,)
    assert np.all(np.isfinite(node_angles))
    assert np.all(np.isfinite(cell_angles))


def test_coriolis_is_node_or_cell_based_on_requested_grid(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    assert grid.get_coriolis(cell=False) is None  # historical setter API
    nodal = grid.f.copy()
    assert grid.get_coriolis(cell=True) is None
    elemental = grid.f.copy()

    omega_times_two = 4 * np.pi / (24 * 60 * 60)
    np.testing.assert_allclose(
        nodal, omega_times_two * np.sin(np.deg2rad(grid.lat)), atol=1e-15
    )
    np.testing.assert_allclose(
        elemental, omega_times_two * np.sin(np.deg2rad(grid.latc)), atol=1e-15
    )
    assert nodal.shape == (grid.node_number,)
    assert elemental.shape == (grid.cell_number,)
    assert nodal.shape != elemental.shape


def test_cfl_matches_independent_triangle_edge_calculation(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    assert grid.get_cfl(z=1.5, u=3.0, g=9.81, verbose=False) is None
    vertices = np.stack((grid.x[grid.tri], grid.y[grid.tri]), axis=-1)
    edge_lengths = np.linalg.norm(vertices - np.roll(vertices, -1, axis=1), axis=2)
    max_depth = np.max(grid.h[grid.tri], axis=1)
    expected = np.min(edge_lengths, axis=1) / (
        (np.sqrt((max_depth + 1.5) * 9.81) + 3.0) * np.sqrt(2)
    )
    np.testing.assert_allclose(grid.ts, expected, atol=1e-12)
    assert grid.ts.shape == (grid.cell_number,)


@pytest.fixture
def fake_tge(monkeypatch):
    """Count TGE calls without compiling large TGE/Numba kernels."""
    import fvtools.grid.tge as tge

    calls = {"main": 0, "neighbors": 0, "art1": 0}

    def build_tge(grid, verbose=False):
        calls["main"] += 1

        def compute_art1():
            calls["art1"] += 1
            result.art1 = np.array([1.0, 2.0, 3.0, 4.0])

        result = SimpleNamespace(
            NBSN=np.zeros((grid.node_number, 2), dtype=int),
            NTSN=np.full(grid.node_number, 2, dtype=int),
            NBVE=np.zeros((grid.node_number, 2), dtype=int),
            NTVE=np.ones(grid.node_number, dtype=int),
            NBE=np.full((grid.cell_number, 3), -1, dtype=int),
            ISBCE=np.ones(grid.cell_number, dtype=int),
            ISONB=np.ones(grid.node_number, dtype=int),
            get_art1=compute_art1,
        )
        return result

    def fake_neighbors(nbve, ntve, triangles, n_cells):
        calls["neighbors"] += 1
        assert triangles.shape == (n_cells, 3)
        return np.full((n_cells, 2), -1, dtype=int), np.zeros(n_cells, dtype=int)

    monkeypatch.setattr(tge, "main", build_tge)
    monkeypatch.setattr(tge, "get_NBSE_NESE", fake_neighbors)
    return calls


def test_tge_lazy_initialization_and_cached_node_cell_connectivity(
    small_grid_file, fake_tge
):
    grid = FVCOM_grid(str(small_grid_file))
    assert "_T" not in grid.__dict__
    assert grid.nbsn.shape == (grid.node_number, 2)
    assert grid.ntsn.shape == (grid.node_number,)
    assert grid.nbve.shape == (grid.node_number, 2)
    assert grid.ntve.shape == (grid.node_number,)
    assert grid.nbe.shape == (grid.cell_number, 3)
    assert grid.nbse.shape == (grid.cell_number, 2)
    assert grid.nese.shape == (grid.cell_number,)
    assert grid.ISBCE.shape == (grid.cell_number,)
    assert grid.ISONB.shape == (grid.node_number,)
    np.testing.assert_array_equal(grid.art1, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(grid.art1, [1.0, 2.0, 3.0, 4.0])
    assert fake_tge == {"main": 1, "neighbors": 1, "art1": 1}
    assert grid.T is grid._T


def test_boundary_properties_remain_lazy_without_full_tge(small_grid_file, monkeypatch):
    import fvtools.grid.tge as tge

    calls = []

    def only_boundary(grid, verbose=False):
        calls.append(grid)
        return SimpleNamespace(
            ISBCE=np.array([1, 2]),
            ISONB=np.array([0, 1, 2, 1]),
            NBE=np.full((2, 3), -1, dtype=int),
        )

    monkeypatch.setattr(tge, "calculate_nbe_and_boundaries", only_boundary)
    grid = FVCOM_grid(str(small_grid_file))
    np.testing.assert_array_equal(grid.ISBCE, [1, 2])
    np.testing.assert_array_equal(grid.ISONB, [0, 1, 2, 1])
    np.testing.assert_array_equal(grid.nbe, np.full((2, 3), -1))
    assert len(calls) == 3  # each property independently initializes if missing
    assert "_T" not in grid.__dict__
    _ = grid.ISBCE, grid.ISONB, grid.nbe
    assert len(calls) == 3  # cached after first access
    grid.ISBCE = np.array([0, 0])
    np.testing.assert_array_equal(grid.ISBCE, [0, 0])


def test_reassigning_mesh_geometry_invalidates_tge_cache(small_grid_file, fake_tge):
    grid = FVCOM_grid(str(small_grid_file))
    old = grid.T
    _ = grid.art1
    assert "_art1" in grid.__dict__
    assert fake_tge["main"] == 1

    grid.x = grid.x + 1.0
    assert "_T" not in grid.__dict__
    assert "_nbsn" not in grid.__dict__
    assert "_art1" not in grid.__dict__
    assert grid.T is not old
    assert fake_tge["main"] == 2

    grid.tri = grid.tri.copy()
    assert "_T" not in grid.__dict__
    _ = grid.T
    assert fake_tge["main"] == 3


def test_reference_mesh_coriolis_and_cfl_staggering(reference_paths):
    _, npy_path = reference_paths
    # M.npy contains a pickled object dictionary: only load trusted fixtures.
    grid = FVCOM_grid(str(npy_path))
    grid.get_coriolis(cell=False)
    node_f = grid.f.copy()
    grid.get_coriolis(cell=True)
    cell_f = grid.f.copy()
    assert node_f.shape == (grid.node_number,)
    assert cell_f.shape == (grid.cell_number,)
    assert np.all(np.isfinite(node_f))
    assert np.all(np.isfinite(cell_f))
    grid.get_cfl(verbose=False)
    assert grid.ts.shape == (grid.cell_number,)
    assert np.all(np.isfinite(grid.ts))
    assert np.all(grid.ts > 0)
