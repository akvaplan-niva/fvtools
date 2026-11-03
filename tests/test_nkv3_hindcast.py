'''
METNorKystV3Hindcast ('NKv3-hindcast'): the NorKyst v3 hindcast (thredds romshindcast/norkyst_v3/sdepth) is
post-processed: every field is at the rho points, the currents are already east/north, and the names differ
from ROMS output (temperature, salinity, u_eastward, ..., time in seconds since 1970, sea_mask).

These tests write a small synthetic day file in that layout (NorKyst's polar stereographic 800 m grid near
Tampen, int16-packed like MET's files, with land in one corner) and check the reader, the nearest-4 stencils,
the downloader's names, the rotation of the currents and the file list. They also check that the staggered
(operational) readers behave as before.

    python -m pytest tests
'''
from datetime import datetime
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
from pyproj import Proj, Transformer
from scipy.ndimage import binary_dilation
from scipy.spatial import Delaunay

from fvtools.grid.roms_grid import (METNorKystV3, METNorKystV3Hindcast, ROMSbase, RomsDownloader, get_roms_grid)
from fvtools.interpolators.roms_interpolators import LinearInterpolation, N4ROMSNESTING
import fvtools.nesting.roms_nesting_fg as rn

NORKYST = '+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70'
MODEL = 'epsg:32631'
NY, NX, NZ, DX = 40, 50, 40, 800.0
U0, V0 = 0.3, 0.1          # eastward and northward current at sea


def hindcast_grid(lon0=4.6, lat0=61.0):
    '''lon/lat of a NorKyst-like 800 m polar stereographic grid, (Y, X)'''
    to_st = Transformer.from_crs('EPSG:4326', NORKYST, always_xy=True)
    X0, Y0 = to_st.transform(lon0, lat0)
    X, Y = np.meshgrid(X0 + DX * (np.arange(NX) - NX // 2), Y0 + DX * (np.arange(NY) - NY // 2))
    lon, lat = to_st.transform(X, Y, direction='INVERSE')
    return X, Y, lon, lat


def packed(nc, name, dims, scale, offset=0.0):
    v = nc.createVariable(name, 'i2', dims, fill_value=np.int16(-32767))
    v.scale_factor, v.add_offset = scale, offset
    return v


def temperature_field(xu, yu, x0, y0):
    '''linear in UTM x/y, so bilinear interpolation on the (nearly rectangular) boxes is close to exact'''
    return 8 + 2e-5 * (xu - x0) - 1e-5 * (yu - y0)


def write_day(path, day=datetime(2024, 1, 1)):
    X, Y, lon, lat = hindcast_grid()
    xu, yu = Proj(MODEL)(lon, lat)
    land = (np.arange(NY)[:, None] >= 34) & (np.arange(NX)[None, :] >= 44)
    coast = binary_dilation(land) & ~land            # sea cells next to land: the hindcast masks ubar/vbar there
    h = 50 + 2.0 * np.arange(NX)[None, :] + 1.0 * np.arange(NY)[:, None]
    h[land] = 10.0
    s_w = np.linspace(-1, 0, NZ + 1)
    s_rho = (s_w[1:] + s_w[:-1]) / 2
    t0 = (day - datetime(1970, 1, 1)).total_seconds()
    x0, y0 = xu.mean(), yu.mean()
    temp = temperature_field(xu, yu, x0, y0)
    with netCDF4.Dataset(path, 'w') as nc:
        for name, n in (('time', None), ('s_rho', NZ), ('s_w', NZ + 1), ('Y', NY), ('X', NX)):
            nc.createDimension(name, n)
        t = nc.createVariable('time', 'f8', ('time',))
        t.units = 'seconds since 1970-01-01 00:00:00'
        t[:] = t0 + 3600.0 * np.arange(24)
        nc.createVariable('X', 'f8', ('X',))[:] = X[0]
        nc.createVariable('Y', 'f8', ('Y',))[:] = Y[:, 0]
        nc.createVariable('lon', 'f8', ('Y', 'X'))[:] = lon
        nc.createVariable('lat', 'f8', ('Y', 'X'))[:] = lat
        nc.createVariable('h', 'f4', ('Y', 'X'))[:] = h
        nc.createVariable('sea_mask', 'i1', ('Y', 'X'), fill_value=np.int8(-127))[:] = (~land).astype(np.int8)
        sr = nc.createVariable('s_rho', 'f8', ('s_rho',))
        sr.standard_name = 'ocean_s_coordinate_g2'
        sr[:] = s_rho
        nc.createVariable('Cs_r', 'f8', ('s_rho',))[:] = s_rho
        nc.createVariable('s_w', 'f8', ('s_w',))[:] = s_w
        nc.createVariable('Cs_w', 'f8', ('s_w',))[:] = s_w
        nc.createVariable('hc', 'f8', ())[...] = 100.0
        packed(nc, 'zeta', ('time', 'Y', 'X'), 0.01)[:] = np.ma.masked_array(np.full((24, NY, NX), 0.5), np.broadcast_to(land, (24, NY, NX)))
        T = packed(nc, 'temperature', ('time', 's_rho', 'Y', 'X'), 0.001, 10.0)
        S = packed(nc, 'salinity', ('time', 's_rho', 'Y', 'X'), 0.001, 30.0)
        u = packed(nc, 'u_eastward', ('time', 's_rho', 'Y', 'X'), 0.001)
        v = packed(nc, 'v_northward', ('time', 's_rho', 'Y', 'X'), 0.001)
        ub = packed(nc, 'ubar_eastward', ('time', 'Y', 'X'), 0.001)
        vb = packed(nc, 'vbar_northward', ('time', 'Y', 'X'), 0.001)
        for k in range(24):
            T[k] = temp[None] + 0.01 * np.arange(NZ)[:, None, None] + 0.001 * k
            S[k] = np.full((NZ, NY, NX), 35.0)
            u[k] = np.where(land, 0.0, U0)[None].repeat(NZ, 0)          # not masked on land, as in the hindcast
            v[k] = np.where(land, 0.0, V0)[None].repeat(NZ, 0)
            ub[k] = np.ma.masked_array(np.full((NY, NX), U0), land | coast)     # masked on land and next to it
            vb[k] = np.ma.masked_array(np.full((NY, NX), V0), land | coast)
    return dict(lon=lon, lat=lat, xu=xu, yu=yu, land=land, coast=coast, h=h, x0=x0, y0=y0)


@pytest.fixture(scope='module')
def day(tmp_path_factory):
    path = str(tmp_path_factory.mktemp('hindcast') / 'norkyst800-20240101.nc')
    return path, write_day(path)


def reader(path):
    R = METNorKystV3Hindcast()
    R.get_norkyst_url = lambda date: path
    R.Proj = Proj(MODEL)
    return R


def mesh_in_sea(G, n=120, seed=2):
    '''random nodes in the sea part of the grid, away from the land corner and the edges, and their triangles'''
    rng = np.random.default_rng(seed)
    j = rng.uniform(4, 28, n)
    i = rng.uniform(4, 38, n)
    x = np.interp(i, np.arange(NX), G['xu'][NY // 2]) + 0 * j
    y = np.interp(j, np.arange(NY), G['yu'][:, NX // 2]) + 0 * i
    tri = Delaunay(np.c_[x, y]).simplices
    return x, y, tri


def test_mother_name_and_url():
    R = get_roms_grid('NKv3-hindcast')
    assert isinstance(R, METNorKystV3Hindcast)
    assert R.get_norkyst_url(datetime(2024, 1, 1)) == \
        'https://thredds.met.no/thredds/dodsC/romshindcast/norkyst_v3/sdepth/2024/01/norkyst800-20240101.nc'
    assert R.time_name == 'time' and R.staggered is False
    assert R.varnames == {'salt': 'salinity', 'temp': 'temperature', 'zeta': 'zeta', 'u': 'u_eastward',
                          'v': 'v_northward', 'ua': 'ubar_eastward', 'va': 'vbar_northward'}
    # the ROMS-output readers keep their names and staggering
    for mother in ('NKv2', 'NKv3', 'H-NS', 'IMR-NF'):
        O = get_roms_grid(mother)
        assert O.time_name == 'ocean_time' and O.staggered is True and O.varnames == ROMSbase.varnames


def test_grid_on_rho_points(day):
    path, G = day
    R = reader(path)
    x, y, _ = mesh_in_sea(G)
    R.load_grid(x, y, offset=3000)                      # no date: reads the grid from grid_date's file
    # scalars: land from sea_mask; currents: also the sea cells where ubar/vbar are masked
    assert np.array_equal(R.rho_mask, G['land']) and G['coast'].any()
    assert np.array_equal(R.u_mask, G['land'] | G['coast']) and np.array_equal(R.v_mask, R.u_mask)
    assert not R.angle.any() and R.angle.shape == (NY, NX)
    assert np.array_equal(R.x_u, R.x_rho) and np.array_equal(R.y_v, R.y_rho)
    assert np.allclose(R.x_rho, G['xu']) and np.allclose(R.h_rho, G['h'])
    # depths at u and v points are the rho depths, with the rho shapes
    assert R.h_u is R.h_rho or np.array_equal(R.h_u, R.h_rho)
    assert R.z_u.shape == R.z_rho.shape == (NY, NX, NZ) and R.zw_v.shape == R.zw_rho.shape == (NZ + 1, NY, NX)
    assert R.dsigma_u.shape == (NZ, NY, NX) and np.allclose(R.dsigma_u.sum(0), 1)
    assert np.array_equal(R.fv_u_mask, R.fv_rho_mask) and R.Land_u.shape == R.Land_rho.shape


def test_staggered_readers_unchanged():
    R = METNorKystV3()
    R.lon_rho = np.zeros((5, 7))
    R.h_rho = 100 + np.arange(35.0).reshape(5, 7)
    R.s_rho = R.Cs_r = np.linspace(-0.95, -0.05, 10)
    R.s_w = R.Cs_w = np.linspace(-1, 0, 11)
    R.hc = 100.0
    assert R.h_u.shape == (5, 6) and np.allclose(R.h_u, (R.h_rho[:, 1:] + R.h_rho[:, :-1]) / 2)
    assert R.h_v.shape == (4, 7) and np.allclose(R.h_v, (R.h_rho[1:] + R.h_rho[:-1]) / 2)
    assert R.z_u.shape == (5, 6, 10) and R.zw_v.shape == (11, 4, 7)


def test_uv_stencils_are_rho_stencils_at_cells(day):
    path, G = day
    R = reader(path)
    x, y, tri = mesh_in_sea(G)
    R.load_grid(x, y, offset=3000)
    N4 = N4ROMSNESTING(R, x=x, y=y, tri=tri, uv=True)
    N4.nearest4()                                       # land check on: the mesh is away from the land corner
    # u and v use the rho points around each cell centre
    ref_i, ref_c = N4.get_interpolation_matrices(xy_source=N4.rho_points, xy_source_center=N4.psi_points,
                                                 xy_fvcom=N4.fv_cells)
    assert np.array_equal(N4.u_index, ref_i) and np.array_equal(N4.v_index, ref_i)
    assert np.allclose(N4.u_coef, ref_c) and np.allclose(N4.v_coef, ref_c)
    # each stencil is the box around the point: weights in [0, 1] summing to 1, and x/y come back
    for idx, coef, px, py in ((N4.rho_index, N4.rho_coef, x, y), (N4.u_index, N4.u_coef, N4.xc, N4.yc)):
        assert np.allclose(coef.sum(1), 1) and coef.min() > -1e-9 and coef.max() < 1 + 1e-9
        assert np.abs(np.sum(R.cropped_x_rho[idx] * coef, 1) - px).max() < 5
        assert np.abs(np.sum(R.cropped_y_rho[idx] * coef, 1) - py).max() < 5
    # the grid angle is zero everywhere, so no rotation from grid axes to east/north
    assert not N4.angle.any()


def test_land_in_stencil_is_caught(day):
    path, G = day
    R = reader(path)
    j, i = np.where(G['land'])
    x = np.r_[G['xu'][j.min() - 1, i.min() - 1], G['xu'][j.min() + 1, i.min() + 1], G['xu'][j.min() - 3, i.min() + 2]]
    y = np.r_[G['yu'][j.min() - 1, i.min() - 1], G['yu'][j.min() + 1, i.min() + 1], G['yu'][j.min() - 3, i.min() + 2]]
    R.load_grid(x, y, offset=3000)
    N4 = N4ROMSNESTING(R, x=x, y=y, tri=np.array([[0, 1, 2]]), uv=True)
    import matplotlib
    matplotlib.use('Agg')
    from fvtools.interpolators.roms_interpolators import LandError
    with pytest.raises(LandError):
        N4.nearest4()


class Downloader(RomsDownloader, LinearInterpolation):
    pass


def test_downloader_names_and_rotation(day):
    path, G = day
    R = reader(path)
    x, y, tri = mesh_in_sea(G)
    R.load_grid(x, y, offset=3000)
    N4 = N4ROMSNESTING(R, x=x, y=y, tri=tri, uv=True)
    N4.nearest4()
    utm_angle = np.linspace(-0.2, 0.3, len(tri))       # whatever the grid's turn from east/north is
    N4.FV = SimpleNamespace(cell_utm_angle=utm_angle)
    D = Downloader()
    D.N4, D.latlon = N4.dump(), False
    assert D.N4.varnames == METNorKystV3Hindcast.varnames

    k = 5
    ts = D.read_timestep(k, path)
    with netCDF4.Dataset(path) as nc:
        box = (slice(D.N4.m_ri, D.N4.x_ri + 1), slice(D.N4.m_rj, D.N4.x_rj + 1))
        assert np.array_equal(ts.temp, nc['temperature'][(k, slice(None)) + box])
        assert np.array_equal(ts.u, nc['u_eastward'][(k, slice(None)) + box])
        assert np.array_equal(ts.ua, nc['ubar_eastward'][(k,) + box])
    ts = D.crop_and_transpose(ts)
    ts = D.horizontal_interpolation(ts)
    expect_T = temperature_field(x, y, G['x0'], G['y0'])[:, None] + 0.01 * np.arange(NZ)[None] + 0.001 * k
    assert np.abs(ts.temp - expect_T).max() < 2e-3     # packing is 0.001 °C
    assert np.allclose(ts.zeta, 0.5) and np.allclose(ts.salt, 35.0)

    # the vertical interpolation leaves (layers, cells); currents then turn from east/north into UTM only
    ts.u, ts.v = ts.u.T, ts.v.T
    ts = D.adjust_uv(ts)
    c, s = np.cos(utm_angle), np.sin(utm_angle)
    assert np.allclose(ts.u, U0 * c + V0 * s, atol=2e-3) and np.allclose(ts.v, -U0 * s + V0 * c, atol=2e-3)
    assert np.allclose(ts.ua, U0 * c + V0 * s, atol=2e-3) and np.allclose(ts.va, -U0 * s + V0 * c, atol=2e-3)


def test_file_list_reads_time_in_seconds(day):
    path, _ = day
    R = reader(path)
    time, paths, index = rn.make_fileList(datetime(2024, 1, 1), datetime(2024, 1, 1), R)
    assert time[0] == 60310.0                           # 2024-01-01 00:00 as MJD
    assert np.allclose(np.diff(time), 1 / 24) and index == list(range(24)) and set(paths) == {path}
