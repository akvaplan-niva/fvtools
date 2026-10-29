'''
read_metCoop: MEPS x_wind_10m/y_wind_10m are along MEPS's Lambert grid axes, not east/north.
These tests build a synthetic MEPS grid (MEPS's own Lambert parameters) around a UTM 31 "mesh",
and check that the winds come out along the model's x/y axes (latlon=False) or east/north (latlon=True).

    python -m pytest tests
'''
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
from pyproj import Proj, Transformer

from fvtools.grid.arome_grid import NewAromeGrid, OldAromeGrid, grid_x_axis_angles
from fvtools.interpolators.horizontal_interpolator import N4Coefficients
from fvtools.atm.read_metCoop import AROMEDownloader, AROMETimestep, check_latlon_flag, create_nc_forcing_file

# projection_lambert of meps25epsarchive (2025)
MEPS_LCC = '+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +R=6371000 +units=m +no_defs'
N_LCC = np.sin(np.radians(63.3))  # cone constant
MODEL = 'epsg:32631'
FIELDS = ['caf', 'Pair', 'relative_humidity', 'spechum', 'Tair', 'evap', 'precip', 'andLw', 'SwD']


def meps_lonlat(nx=160, ny=120, dx=2500.0, lon0=4.5, lat0=61.0):
    '''lon/lat of a MEPS-like 2.5 km Lambert grid centred on lon0, lat0, shaped (y, x)'''
    to_lcc = Transformer.from_crs('EPSG:4326', MEPS_LCC, always_xy=True)
    X0, Y0 = to_lcc.transform(lon0, lat0)
    X, Y = np.meshgrid(X0 + dx * (np.arange(nx) - nx // 2), Y0 + dx * (np.arange(ny) - ny // 2))
    return to_lcc.transform(X, Y, direction='INVERSE')


def write_grid_file(path, lon, lat):
    with netCDF4.Dataset(path, 'w') as nc:
        nc.createDimension('y', lon.shape[0])
        nc.createDimension('x', lon.shape[1])
        nc.createDimension('one', 1)
        nc.createVariable('longitude', 'f8', ('y', 'x'))[:] = lon
        nc.createVariable('latitude', 'f8', ('y', 'x'))[:] = lat
        nc.createVariable('land_area_fraction', 'f4', ('one', 'one', 'y', 'x'))[:] = 0.0  # all ocean
    return str(path)


def model_cells(n=400, seed=1):
    '''Random nodes and cells of a "mesh" in UTM 31, inside the synthetic MEPS grid'''
    rng = np.random.default_rng(seed)
    proj = Proj(MODEL)
    lon, lat = rng.uniform(2.0, 7.0, n), rng.uniform(60.2, 61.8, n)
    lonc, latc = rng.uniform(2.0, 7.0, n), rng.uniform(60.2, 61.8, n)
    x, y = proj(lon, lat)
    xc, yc = proj(lonc, latc)
    return SimpleNamespace(x=x, y=y, xc=xc, yc=yc, lon=lon, lat=lat, lonc=lonc, latc=latc,
                           cell_utm_angle=np.zeros(n), reference=MODEL)


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    lon, lat = meps_lonlat()
    meps = NewAromeGrid(Proj(MODEL))
    meps._basepath = write_grid_file(tmp_path_factory.mktemp('meps') / 'meps_det_grid.nc', lon, lat)
    meps.prepare_grid()
    M = model_cells()
    N4 = N4Coefficients(M, meps)
    N4.compute_nearest4()
    return SimpleNamespace(meps=meps, M=M, N4=N4)


def run_pipeline(setup, latlon, x_wind, y_wind):
    '''Cell coefficients and angle for latlon, then read_metCoop's interpolation and rotation of one time step'''
    setup.N4.compute_cell_arome_angle(latlon)
    D = AROMEDownloader(setup.N4, None, [], [], [], [], [], 'unused.nc', latlon)
    assert D.N4new is not None, 'N4Coefficients.dump() failed'
    assert np.array_equal(D.N4new.cell_arome_angle, setup.N4.cell_arome_angle)
    mask = D.N4new.fv_domain_mask
    ts = AROMETimestep()
    for f in FIELDS:
        setattr(ts, f, np.ones(mask.sum()))
    ts.Uwind, ts.Vwind = np.asarray(x_wind)[mask], np.asarray(y_wind)[mask]
    ts = D.interpolate_arome_data(ts, D.N4new)
    ts = D.rotate_arome_vectors(ts, D.N4new)
    return ts.Uwind, ts.Vwind, D.N4new.cell_arome_angle


def wrap_deg(a):
    return (np.degrees(a) + 180) % 360 - 180


def lcc_x_axis_in_model(lon, lat, step=50.0):
    '''Angle from the model's x-axis to MEPS's x-axis, straight from pyproj (as FLOWE's rotate_winds.py)'''
    to_lcc = Transformer.from_crs('EPSG:4326', MEPS_LCC, always_xy=True)
    lcc_to_model = Transformer.from_crs(MEPS_LCC, MODEL, always_xy=True)
    X, Y = to_lcc.transform(lon, lat)
    xa, ya = lcc_to_model.transform(X - step, Y)
    xb, yb = lcc_to_model.transform(X + step, Y)
    return np.arctan2(yb - ya, xb - xa)


def east_in_model(lon, lat):
    '''Angle of true east in the model's x/y'''
    to_model = Transformer.from_crs('EPSG:4326', MODEL, always_xy=True)
    xa, ya = to_model.transform(lon - 0.001, lat)
    xb, yb = to_model.transform(lon + 0.001, lat)
    return np.arctan2(yb - ya, xb - xa)


@pytest.mark.parametrize('latlon', [False, True])
def test_unit_x_wind_points_at_cell_angle(setup, latlon):
    '''x_wind_10m = 1, y_wind_10m = 0 must come out pointing at each cell's cell_arome_angle'''
    shape = setup.meps.lon.shape
    u, v, angle = run_pipeline(setup, latlon, np.ones(shape), np.zeros(shape))
    assert np.allclose(np.hypot(u, v), 1.0, atol=1e-12)
    assert np.allclose(wrap_deg(np.arctan2(v, u) - angle), 0.0, atol=1e-9)
    # and the angle is not zero: the winds are turned (6.5-12.3 deg over this area)
    assert np.all(np.abs(np.degrees(angle)) > 5)


def test_cell_angle_matches_projection(setup):
    '''cell_arome_angle against independent references: pyproj (latlon=False) and the Lambert cone (latlon=True)'''
    M = setup.M
    setup.N4.compute_cell_arome_angle(latlon=False)
    d = wrap_deg(setup.N4.cell_arome_angle - lcc_x_axis_in_model(M.lonc, M.latc))
    assert np.abs(d).max() < 0.05, f'model x/y: {np.abs(d).max():.4f} deg from pyproj'

    setup.N4.compute_cell_arome_angle(latlon=True)
    d = wrap_deg(setup.N4.cell_arome_angle - np.radians(N_LCC * (15.0 - M.lonc)))
    assert np.abs(d).max() < 0.005, f'east: {np.abs(d).max():.4f} deg from n (15 - lon)'


@pytest.mark.parametrize('latlon', [False, True])
def test_east_wind_end_to_end(setup, latlon):
    '''A 10 m/s eastward wind, given as MEPS grid components, must come out eastward'''
    lon, lat = setup.meps.lon, setup.meps.lat
    a_grid = np.radians(N_LCC * (15.0 - lon))  # MEPS x-axis from east, counter-clockwise
    x_wind, y_wind = 10 * np.cos(a_grid), -10 * np.sin(a_grid)
    u, v, _ = run_pipeline(setup, latlon, x_wind, y_wind)

    M = setup.M
    expected = np.zeros(len(M.xc)) if latlon else east_in_model(M.lonc, M.latc)
    d = wrap_deg(np.arctan2(v, u) - expected)
    assert np.abs(d).max() < (0.01 if latlon else 0.05), f'{np.abs(d).max():.4f} deg off'
    assert np.allclose(np.hypot(u, v), 10.0, rtol=1e-6)

    # Before the fix, latlon=True wrote the interpolated grid components as they were (6-12 deg off here)
    if latlon:
        D = AROMEDownloader(setup.N4, None, [], [], [], [], [], 'unused.nc', latlon)
        raw_u = np.sum(x_wind[D.N4new.fv_domain_mask][D.N4new.cindex] * D.N4new.ccoef, axis=1)
        raw_v = np.sum(y_wind[D.N4new.fv_domain_mask][D.N4new.cindex] * D.N4new.ccoef, axis=1)
        assert np.abs(wrap_deg(np.arctan2(raw_v, raw_u))).min() > 5


def test_grid_angles_on_old_grid(tmp_path):
    '''OldAromeGrid gets the same angles, from its pp grid'''
    lon, lat = meps_lonlat(nx=40, ny=30)
    old = OldAromeGrid(Proj(MODEL))
    old.pathToPP = write_grid_file(tmp_path / 'pp.nc', lon, lat)
    old.pathToEX = write_grid_file(tmp_path / 'ex.nc', lon, lat)
    old.prepare_grid()
    a_xy, a_east = grid_x_axis_angles(lon, lat, *Proj(MODEL)(lon, lat))
    assert old.x_axis_angle.shape == lon.shape
    assert np.array_equal(old.x_axis_angle, a_xy) and np.array_equal(old.x_axis_angle_east, a_east)


def test_grid_x_axis_angles_across_dateline():
    '''Longitudes that jump from 180 to -180 must not break the east angle'''
    lon = np.array([[178.0, 179.0, 180.0, -179.0, -178.0]] * 2)
    lat = np.array([[60.0] * 5, [60.1] * 5])
    _, a_east = grid_x_axis_angles(lon, lat, lon, lat)
    assert np.allclose(a_east, 0.0, atol=1e-12)


@pytest.mark.parametrize('latlon, u_name, components', [
    (False, 'Wind Speed along epsg:32631 x', 'along epsg:32631 x/y'),
    (True, 'Eastward Wind Speed', 'eastward/northward'),
])
def test_forcing_file_labels(tmp_path, latlon, u_name, components):
    M = model_cells(n=3)
    M.tri = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    out = tmp_path / 'atm.nc'
    create_nc_forcing_file(str(out), M, np.array([60000.0]), latlon, MODEL)
    with netCDF4.Dataset(out) as nc:
        assert nc.wind_components.startswith(components)
        assert nc['uwind_speed'].long_name == nc['U10'].long_name == u_name
        assert nc['vwind_speed'].long_name == nc['V10'].long_name == u_name.replace(' x', ' y').replace('Eastward', 'Northward')


@pytest.mark.parametrize('reference, latlon, warns', [
    ('epsg:32631', True, True), ('epsg:32631', False, False),
    ('epsg:4326', False, True), ('epsg:4326', True, False),
])
def test_latlon_flag_warning(capsys, reference, latlon, warns):
    check_latlon_flag(SimpleNamespace(reference=reference), latlon)
    assert ('WARNING' in capsys.readouterr().out) == warns
