"""Regression tests for the extraction of InterpolateToZ and ExportGrid.

All export files go into pytest temporary directories, never model input paths.
"""
import numpy as np
import pytest

from fvtools.grid.fvcom_grd import FVCOM_grid, InterpolateToZ, ExportGrid


def test_linear_sigma_to_z_interpolation_and_exact_levels(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    depths = np.broadcast_to(np.array([1., 4., 7.])[:, None], (3, 4))
    field = 3. * depths + np.arange(4)[None, :]
    np.testing.assert_allclose(grid.interpolate_to_z(field, 2.5, depths=depths),
                               7.5 + np.arange(4))
    np.testing.assert_allclose(grid.interpolate_to_z(field, 4.0, depths=depths),
                               12. + np.arange(4))


def test_sigma_to_z_shallow_above_and_below_bottom(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    depths = np.broadcast_to(np.array([1., 4., 7.])[:, None], (3, 4))
    field = 2. * depths
    np.testing.assert_allclose(grid.interpolate_to_z(field, 0.25, depths=depths),
                               np.full(4, 2.))
    assert np.isnan(grid.interpolate_to_z(field, 8., depths=depths)).all()


def test_sigma_to_z_skips_invalid_levels(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    depths = np.broadcast_to(np.array([1., 4., 7.])[:, None], (3, 4))
    field = (2. * depths).copy()
    field[:, 0] = np.nan
    field[1, 1] = np.nan
    out = grid.interpolate_to_z(field, 4., depths=depths)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [8., 8., 8.])


@pytest.mark.parametrize('z', [-1., np.nan, np.inf, [2.]])
def test_invalid_interpolation_depth_is_rejected(small_grid_file, z):
    grid = FVCOM_grid(str(small_grid_file))
    with pytest.raises(ValueError, match='nonnegative, finite depth'):
        grid.interpolate_to_z(np.ones((2, 4)), z, depths=np.ones((2, 4)))


def test_interpolation_shape_validation(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    with pytest.raises(ValueError, match='f and depths must have shape'):
        grid.interpolate_to_z(np.ones((2, 4)), 1., depths=np.ones((3, 4)))
    with pytest.raises(ValueError, match='Field shape does not match sigma-depth'):
        grid.interpolate_to_z(np.ones((7, 8)), 1.)


def test_automatic_sigma_depths_preserve_nodal_and_elemental_staggering(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    nlayer = grid.siglay.shape[1]
    node_field = np.tile(np.arange(grid.node_number, dtype=float), (nlayer, 1))
    cell_field = np.tile(np.arange(grid.cell_number, dtype=float), (nlayer, 1))
    np.testing.assert_allclose(
        grid.interpolate_to_z(node_field, .1),
        grid.interpolate_to_z(node_field, .1, depths=-grid.siglayz.T))
    np.testing.assert_allclose(
        grid.interpolate_to_z(cell_field, .1),
        grid.interpolate_to_z(cell_field, .1, depths=-grid.siglayz_uv.T))
    assert grid.interpolate_to_z(node_field, .1).shape == (grid.node_number,)
    assert grid.interpolate_to_z(cell_field, .1).shape == (grid.cell_number,)


def test_interpolation_matrix_delegates_to_correct_node_and_element_depth(monkeypatch, small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    seen = []
    def stub(levels, depth):
        seen.append((levels.copy(), depth))
        return np.full(levels.shape, depth, dtype=float)
    monkeypatch.setattr('fvtools.grid.tools.make_interpolation_matrices', stub)
    grid.make_interpolation_matrices_TS([-5])
    grid.make_interpolation_matrices_uv([-3])
    np.testing.assert_array_equal(seen[0][0], grid.siglayz)
    np.testing.assert_array_equal(seen[1][0], grid.siglayz_uv)
    assert [depth for _, depth in seen] == [-5, -3]
    np.testing.assert_array_equal(grid.interpolation_matrix_TS_5_m, np.full(grid.siglayz.shape, -5.))
    np.testing.assert_array_equal(grid.interpolation_matrix_uv_3_m, np.full(grid.siglayz_uv.shape, -3.))


def test_get_xy_respects_latlon_option(small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    assert grid.get_xy(False)[0] is grid.x
    assert grid.get_xy(False)[1] is grid.y
    assert grid.get_xy(True)[0] is grid.lon
    assert grid.get_xy(True)[1] is grid.lat


def test_write_bath_uses_requested_coordinate_system(tmp_path, small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    output = tmp_path / 'depth.dat'
    assert grid.write_bath(str(output), latlon=False) is None
    assert output.read_text().splitlines()[0] == 'Node Number = 4'
    np.testing.assert_allclose(np.loadtxt(output, skiprows=1),
                               np.column_stack((grid.x, grid.y, grid.h)), atol=1e-6)
    geo = tmp_path / 'depth_ll.dat'
    grid.write_bath(str(geo), latlon=True)
    np.testing.assert_allclose(np.loadtxt(geo, skiprows=1),
                               np.column_stack((grid.lon, grid.lat, grid.h)), atol=1e-6)


@pytest.mark.parametrize('store_z', [False, True])
def test_write_grd_uses_fortran_indices_and_optional_bathymetry(tmp_path, small_grid_file, store_z):
    grid = FVCOM_grid(str(small_grid_file))
    output = tmp_path / 'grid.dat'
    grid.write_grd(str(output), store_z=store_z)
    lines = output.read_text().splitlines()
    assert lines[:2] == ['Node Number = 4', 'Cell Number = 2']
    assert len(lines) == 2 + grid.cell_number + grid.node_number
    for i, line in enumerate(lines[2:4]):
        assert [int(v) for v in line.split()] == [i+1, *(grid.tri[i] + 1), i]
    saved_nodes = np.loadtxt(output, skiprows=2+grid.cell_number)
    np.testing.assert_array_equal(saved_nodes[:, 0], np.arange(1, 5))
    np.testing.assert_allclose(saved_nodes[:, 1:3], np.column_stack((grid.x, grid.y)))
    np.testing.assert_allclose(saved_nodes[:, 3], grid.h if store_z else np.zeros(4))


def test_write_obc_keeps_nodestring_sequence_and_one_based_nodes(tmp_path, small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    path = tmp_path / 'obc.dat'
    grid.write_obc(str(path))
    assert path.read_text().splitlines() == [
        'OBC Node Number = 4', '1 1 1', '2 2 1', '3 3 1', '4 4 1'
    ]


def test_write_cor_writes_latitude_and_correct_horizontal_coordinates(tmp_path, small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    output = tmp_path / 'cor.dat'
    grid.write_cor(str(output))
    np.testing.assert_allclose(np.loadtxt(output, skiprows=1),
                               np.column_stack((grid.x, grid.y, grid.lat)), atol=1e-6)
    output = tmp_path / 'cor_ll.dat'
    grid.write_cor(str(output), latlon=True)
    np.testing.assert_allclose(np.loadtxt(output, skiprows=1),
                               np.column_stack((grid.lon, grid.lat, grid.lat)), atol=1e-6)


def test_write_sponge_one_based_node_indices(tmp_path, small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    grid.sponge_nodes = [1, 3]
    grid.sponge_radius = 100.0
    grid.sponge_factor = 0.12
    output = tmp_path / 'spg.dat'
    grid.write_sponge(str(output))
    assert output.read_text().splitlines() == [
        'Sponge Node Number = 2', '2 100.000000 0.120000', '4 100.000000 0.120000'
    ]


def test_write_2dm_roundtrip_preserves_connectivity_and_nodestrings(tmp_path, monkeypatch, small_grid_file):
    from fvtools.grid.fvgrid import read_sms_mesh
    grid = FVCOM_grid(str(small_grid_file))
    monkeypatch.chdir(tmp_path)
    grid.write_2dm(name='exported')
    exported = tmp_path / 'exported.2dm'
    assert exported.exists()
    tri, _, x, y, _, _, strings = read_sms_mesh(str(exported), nodestrings=True)
    np.testing.assert_array_equal(tri, grid.tri)
    np.testing.assert_allclose(x, grid.x)
    np.testing.assert_allclose(y, grid.y)
    assert [list(nodes) for nodes in strings] == [list(nodes) for nodes in grid.nodestrings]


def test_write_section_exports_geographic_points(tmp_path, monkeypatch, small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    grid.x_sec = np.array([grid.x[0], grid.x[2]])
    grid.y_sec = np.array([grid.y[0], grid.y[2]])
    monkeypatch.chdir(tmp_path)
    grid.write_section()
    output = tmp_path / 'mini_nodestrings_section.txt'
    assert output.is_file()
    lon, lat = grid.Proj(grid.x_sec, grid.y_sec, inverse=True)
    np.testing.assert_allclose(np.loadtxt(output), np.column_stack((lon, lat)), atol=1e-7)


def test_to_npy_preserves_mesh_sigma_and_projection(tmp_path, small_grid_file):
    grid = FVCOM_grid(str(small_grid_file))
    output = tmp_path / 'export.npy'
    grid.to_npy(str(output))
    # Our own trusted fixture; this file contains a pickled dictionary.
    saved = np.load(output, allow_pickle=True).item()
    np.testing.assert_array_equal(saved['nv'], grid.tri)
    np.testing.assert_array_equal(saved['h'], grid.h)
    np.testing.assert_array_equal(saved['siglev'], grid.siglev)
    np.testing.assert_array_equal(saved['siglay'], grid.siglay)
    assert saved['reference'] == grid.reference


def test_reference_mesh_interpolation_does_not_mutate_input(reference_paths):
    _, filename = reference_paths
    grid = FVCOM_grid(str(filename))
    field = grid.siglayz.T.copy()
    original = field.copy()
    result = grid.interpolate_to_z(field, 0.25)
    assert result.shape == (grid.node_number,)
    # Very shallow nodes may lie below the deepest layer at z=0.25 m.
    assert np.isfinite(result).any()
    np.testing.assert_array_equal(field, original)
