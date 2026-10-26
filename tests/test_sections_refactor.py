"""Regressions for SectionMaker after a *mechanical* module extraction.

Uses a synthetic regular FVCOM mesh; the optional large-grid test loads
real-world M.npy plus a user-provided lon/lat section file.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from fvtools.grid.fvcom_grd import FVCOM_grid, SectionMaker
from fvtools.grid._components.sections import SectionMaker as ImplementationSectionMaker


@pytest.fixture
def section_grid():
    """5 x 5 nodes, 4 x 4 squares, 32 CCW triangles, UTM33 metric geometry.

    These test points lie within *both* the node and cell-centre domains.
    All fields are intentionally analytically simple.
    """
    xx, yy = np.meshgrid(
        500000.0 + 100.0 * np.arange(5),
        7700000.0 + 100.0 * np.arange(5),
    )
    x, y = xx.ravel(), yy.ravel()
    triangles = []
    for row in range(4):
        for col in range(4):
            lower = row * 5 + col
            triangles.extend([
                (lower, lower + 1, lower + 6),
                (lower, lower + 6, lower + 5),
            ])

    siglev = np.tile([0.0, -0.5, -1.0], (x.size, 1))
    siglay = np.tile([-0.25, -0.75], (x.size, 1))
    grid = FVCOM_grid(
        x=x, y=y, tri=np.asarray(triangles, dtype=int),
        lon=np.full(x.size, 15.0), lat=np.full(x.size, 69.0),
        h=np.full(x.size, 20.0),
        siglev=siglev, siglay=siglay, reference='epsg:32633',
    )
    grid.zeta = np.full(x.size, 2.0)
    # Assign directly so tests do not need a GUI or write a section file.
    grid.x_sec = np.array([500100.0, 500150.0, 500200.0, 500250.0, 500300.0])
    grid.y_sec = np.full(5, 7700200.0)
    return grid


def _linear(x, y):
    return (np.asarray(x) - 500000.0) / 100.0 + 2.0 * (np.asarray(y) - 7700000.0) / 100.0


def test_public_sectionmaker_identity():
    assert SectionMaker is ImplementationSectionMaker
    assert issubclass(FVCOM_grid, SectionMaker)
    assert hasattr(FVCOM_grid, 'prepare_section_interactive')
    assert hasattr(FVCOM_grid, 'get_section_velocity')


def test_nodal_2d_section_interpolates_linear_field(section_grid):
    grid = section_grid
    result = grid.get_section_data(_linear(grid.x, grid.y))
    assert len(result['transect']) == len(grid.x_sec)
    np.testing.assert_allclose(
        result['transect'], _linear(result['x'], result['y']), atol=1e-9
    )
    np.testing.assert_allclose(result['dst'], [0, 0.05, 0.1, 0.15, 0.2])


def test_cell_2d_section_interpolates_linear_field(section_grid):
    grid = section_grid
    result = grid.get_section_data(_linear(grid.xc, grid.yc))
    assert len(result['transect']) == len(grid.x_sec)
    np.testing.assert_allclose(
        result['transect'], _linear(result['x'], result['y']), atol=1e-9
    )


def test_nodal_3d_section_and_visualization_depth(section_grid):
    grid = section_grid
    base = _linear(grid.x, grid.y)
    levels = np.column_stack((base, base + 7.0))
    result = grid.get_section_data(levels)
    assert result['transect'].shape == (len(grid.x_sec), 2)
    np.testing.assert_allclose(
        result['transect'][:, 0], _linear(result['x'], result['y']), atol=1e-9
    )
    np.testing.assert_allclose(
        result['transect'][:, 1], _linear(result['x'], result['y']) + 7, atol=1e-9
    )
    # dpt_sec deliberately pins the upper midpoint to zeta for plotting.
    np.testing.assert_allclose(result['h'][:, 0], 2.0, atol=1e-9)
    np.testing.assert_allclose(result['h'][:, 1], -15.0, atol=1e-9)


def test_projected_cell_velocities_physical_z_and_cache_reuse(section_grid):
    grid = section_grid
    # Test both accepted input layouts: (sigma,cell) and (cell,sigma).
    u = np.full((2, grid.cell_number), 3.0)
    v = np.full((grid.cell_number, 2), 4.0)
    first = grid.get_section_velocity(u, v, velocity_coordinates='projected')
    assert first['u'].shape == (len(grid.x_sec), 2)
    np.testing.assert_allclose(first['along'], 3.0, atol=1e-9)
    np.testing.assert_allclose(first['normal'], 4.0, atol=1e-9)
    np.testing.assert_allclose(first['speed'], 5.0, atol=1e-9)
    np.testing.assert_allclose(
        first['h'],
        np.tile([-3.5, -14.5], (len(grid.x_sec), 1)),
        atol=1e-9,
    )
    assert first['dst'].shape == first['h'].shape

    triangulation = grid.trs
    weights = grid._section_weights_cache
    basis = grid._section_vector_basis_cache
    second = grid.get_section_velocity(u * 2, v * 2, velocity_coordinates='projected')
    # No new triangles/weights/basis between movie frames.
    assert grid.trs is triangulation
    assert grid._section_weights_cache is weights
    assert grid._section_vector_basis_cache is basis
    np.testing.assert_allclose(second['speed'], 10.0, atol=1e-9)


def test_section_weights_and_vector_basis_refresh_after_point_change(section_grid):
    grid = section_grid
    u = np.full((2, grid.cell_number), 3.0)
    v = np.full((2, grid.cell_number), 4.0)
    grid.get_section_velocity(u, v, velocity_coordinates='projected')
    previous_weights = grid._section_weights_cache
    previous_basis = grid._section_vector_basis_cache
    grid.x_sec = grid.x_sec + 5.0
    result = grid.get_section_velocity(u, v, velocity_coordinates='projected')
    assert grid._section_weights_cache is not previous_weights
    assert grid._section_vector_basis_cache is not previous_basis
    np.testing.assert_allclose(result['x'], grid.x_sec)


def test_invalid_velocity_does_not_change_section_geometry(section_grid):
    grid = section_grid
    u = np.full((2, grid.cell_number), 3.0)
    v = np.full((2, grid.cell_number), 4.0)
    good = grid.get_section_velocity(u, v, velocity_coordinates='projected')
    u[:] = np.nan
    bad = grid.get_section_velocity(u, v, velocity_coordinates='projected')
    np.testing.assert_array_equal(bad['x'], good['x'])
    np.testing.assert_array_equal(bad['y'], good['y'])
    np.testing.assert_allclose(bad['h'], good['h'])
    assert np.isnan(bad['normal']).all()
    assert np.isnan(bad['along']).all()
    assert np.isnan(bad['speed']).all()


def test_optional_real_mesh_section(reference_paths):
    """Set FVTOOLS_TEST_SECTION to a two-column lon/lat text section file.

    This is optional: the user's real section can be supplied without adding
    potentially sensitive or bulky field data to the repository.
    """
    section = os.environ.get('FVTOOLS_TEST_SECTION')
    if not section:
        pytest.skip('Set FVTOOLS_TEST_SECTION to a two-column lon/lat section file')
    path = Path(section)
    if not path.is_file():
        pytest.skip(f'Optional real section file not found: {path}')
    _, npy = reference_paths
    grid = FVCOM_grid(str(npy))
    grid.prepare_section(section_file=str(path))
    assert len(grid.x_sec) >= 2
    result = grid.get_section_data(grid.h)
    assert len(result['transect']) > 0
    assert np.isfinite(result['transect']).all()
