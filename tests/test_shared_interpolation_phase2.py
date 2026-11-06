"""Offline regression tests for the shared ROMS/AROME/GLORYS/ERA5 interpolation.

All source/target grids are synthetic. No THREDDS calls or output files.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fvtools.interpolators.structured_bilinear import structured_bilinear_weights


def grid(ny=8, nx=12, *, angle=0.0, skew=False):
    j, i = np.mgrid[:ny, :nx]
    a = 1.7 * i + .08 * i**2
    b = 2.2 * j + .05 * j**2
    if skew:
        a = a + .15 * j * i / nx
        b = b + .13 * j * i / ny
    ca, sa = np.cos(angle), np.sin(angle)
    return np.stack((543000 + a*ca - b*sa, 7040000 + a*sa + b*ca), axis=-1)


def in_cell(g, j, i, s, t):
    p00,p10,p11,p01 = g[j,i],g[j,i+1],g[j+1,i+1],g[j+1,i]
    return (1-s)*(1-t)*p00+s*(1-t)*p10+s*t*p11+(1-s)*t*p01


@pytest.mark.parametrize('angle,skew', [(0,False),(.75,False),(-.8,True)])
def test_reproduces_coordinates_and_linear_fields(angle,skew):
    g = grid(angle=angle, skew=skew)
    rng = np.random.default_rng(17)
    q = np.stack([in_cell(g,int(j),int(i),s,t) for j,i,s,t in zip(
        rng.integers(0,7,150), rng.integers(0,11,150),rng.random(150),rng.random(150))])
    mask = np.ones(g.shape[:2],bool)
    ids,w = structured_bilinear_weights(g[mask], mask, q, batch_size=19)
    assert ids.shape == w.shape == (150,4)
    np.testing.assert_allclose(w.sum(axis=1),1.,atol=1e-12)
    assert np.all(w >= 0) and np.all(w <= 1)
    np.testing.assert_allclose(np.sum(g[mask][ids]*w[:,:,None],axis=1),q,atol=1e-8)
    z = .25*g[mask,0] - .75*g[mask,1] + 9
    np.testing.assert_allclose(np.sum(z[ids]*w,axis=1), .25*q[:,0]-.75*q[:,1]+9,atol=1e-8)


def test_exact_bilinear_field_on_skew_cell():
    g=grid(skew=True)
    mask=np.ones(g.shape[:2],bool)
    j,i,s,t=3,5,.2,.7
    q=in_cell(g,j,i,s,t)[None,:]
    ids,w=structured_bilinear_weights(g[mask],mask,q)
    idx=np.array([j*12+i,j*12+i+1,(j+1)*12+i+1,(j+1)*12+i])
    np.testing.assert_array_equal(ids[0],idx)
    np.testing.assert_allclose(w[0],[(1-s)*(1-t),s*(1-t),s*t,(1-s)*t],atol=1e-8)


def test_irregular_crop_indices_match_masked_flattening():
    g=grid(ny=6,nx=7,angle=.2,skew=True)
    mask=np.ones(g.shape[:2],bool)
    mask[0,:2]=False
    mask[5,5:]=False
    q=in_cell(g,2,3,.25,.65)[None,:]
    ids,w=structured_bilinear_weights(g[mask],mask,q)
    lookup=np.full(mask.shape,-1,dtype=int)
    lookup[mask]=np.arange(mask.sum())
    np.testing.assert_array_equal(ids[0],lookup[[2,2,3,3],[3,4,4,3]])
    np.testing.assert_allclose(np.sum(g[mask][ids]*w[:,:,None],axis=1),q,atol=1e-8)


def test_missing_corner_and_outside_domain_raise():
    g=grid(4,4)
    mask=np.ones((4,4),bool)
    mask[1,1]=False
    with pytest.raises(ValueError,match='outside complete structured-grid'):
        structured_bilinear_weights(g[mask],mask,in_cell(g,0,0,.5,.5)[None,:])
    with pytest.raises(ValueError,match='outside complete structured-grid'):
        structured_bilinear_weights(g.reshape(-1,2),np.ones((4,4),bool),g[0,0:1]-[100,100])


def test_single_cell_edges_and_empty_targets():
    g=grid(2,2)
    mask=np.ones((2,2),bool)
    q=np.stack([g[0,0],g[0,1],g[1,0],g[1,1],in_cell(g,0,0,.0,.4)])
    ids,w=structured_bilinear_weights(g[mask],mask,q)
    np.testing.assert_allclose(np.sum(g[mask][ids]*w[:,:,None],axis=1),q,atol=1e-8)
    a,b=structured_bilinear_weights(g[mask],mask,np.empty((0,2)))
    assert a.shape==b.shape==(0,4)


def test_invalid_shapes_and_nonfinite_input():
    g=grid(3,4)
    mask=np.ones(g.shape[:2],bool)
    with pytest.raises(ValueError,match='Source coordinates'):
        structured_bilinear_weights(g[mask][:-1],mask,np.array([[1.,2.]]))
    with pytest.raises(ValueError,match='finite'):
        structured_bilinear_weights(g[mask],mask,np.array([[np.nan,2.]]))
    with pytest.raises(ValueError,match='at least 2x2'):
        structured_bilinear_weights(np.zeros((3,2)),np.ones((1,3),bool),np.array([[1.,2.]]))


def test_all_three_roms_staggers_share_same_core():
    g=grid(ny=7,nx=9,skew=True)
    staggers={'rho':g, 'u':.5*(g[:,:-1]+g[:,1:]), 'v':.5*(g[:-1]+g[1:])}
    for gsub in staggers.values():
        mask=np.ones(gsub.shape[:2],bool)
        q=in_cell(gsub,2,3,.25,.6)[None,:]
        inds,w=structured_bilinear_weights(gsub[mask],mask,q)
        np.testing.assert_allclose(np.sum(gsub[mask][inds]*w[:,:,None],axis=1),q,atol=1e-8)


def test_complete_fallback_beyond_128_nearest_centers():
    # Construct many very small cells packed next to one very wide cell. The
    # containing wide cell's center is far outside the k=128 nearest centers.
    # Guaranteed radius fallback must still select the containing wide cell.
    ny, nx = 3, 401
    xrow=np.r_[0.,1000.,1000.+np.arange(1,nx-1)*.002]
    yrow=np.arange(ny)*10.
    xx,yy=np.meshgrid(xrow,yrow)
    xy=np.stack((xx,yy),axis=-1)
    mask=np.ones((ny,nx),bool)
    q=np.array([[999.99,5.]])
    ids,w=structured_bilinear_weights(xy[mask],mask,q)
    np.testing.assert_allclose(np.sum(xy[mask][ids]*w[:,:,None],axis=1),q,atol=1e-8)
    assert 0 in ids[0] or nx in ids[0]


def test_public_n4_entrypoint_uses_shared_kernel():
    from fvtools.interpolators.nearest4 import N4
    g=grid(4,6,skew=True)
    mask=np.ones(g.shape[:2],bool)
    q=in_cell(g,1,2,.3,.7)[None,:]
    idx,w=N4().get_interpolation_matrices(xy_source=g[mask],source_mask=mask,
                                             xy_fvcom=q,widget_title='test')
    np.testing.assert_allclose(np.sum(g[mask][idx]*w[:,:,None],axis=1),q,atol=1e-8)


def test_atmospheric_adapter_indices_match_cropped_source_and_fields():
    from fvtools.interpolators.horizontal_interpolator import N4Coefficients
    g=grid(8,10,skew=True)
    mask=np.ones(g.shape[:2],bool)
    mask[:1,:2]=False
    mask[-1,-2:]=False
    q=np.stack([in_cell(g,2,3,.2,.4),in_cell(g,3,4,.3,.6),in_cell(g,3,5,.3,.1)])
    coarse=SimpleNamespace(x=g[:,:,0], y=g[:,:,1],
        x_center=(g[:-1,:-1,0]+g[:-1,1:,0]+g[1:,:-1,0]+g[1:,1:,0])/4,
        y_center=(g[:-1,:-1,1]+g[:-1,1:,1]+g[1:,:-1,1]+g[1:,1:,1])/4,
        land_mask=np.ones(mask.shape,bool),
        crop_grid=lambda xlim,ylim:mask)
    fv=SimpleNamespace(x=q[:,0],y=q[:,1],xc=q[:1,0],yc=q[:1,1],cell_utm_angle=np.zeros(1))
    n=N4Coefficients(fv,coarse)
    n.compute_nearest4()
    np.testing.assert_allclose(np.sum(g[mask][n.nindex]*n.ncoef[:,:,None],axis=1),q,atol=1e-8)
    assert n.nindex.max()<mask.sum()
    assert n.cindex.shape==(1,4)
    # With all-ocean land mask, radiation weights match raw interpolation.
    np.testing.assert_allclose(n.ncoef_ocean,n.ncoef,atol=1e-10)


@pytest.mark.parametrize('uv', [False, True])
def test_glorys_adapter_matches_rectilinear_source(uv):
    from fvtools.interpolators.glorys_interpolators import N4GLORYS
    lon,lat=np.meshgrid(np.linspace(4,6,9),np.linspace(60,61,7))
    g=np.stack((lon,lat),axis=-1)
    query=np.stack([in_cell(g,2,3,.2,.4),in_cell(g,3,4,.3,.6),in_cell(g,3,5,.3,.1)])
    # Lightweight identity projection, enough to verify indexing/staggering.
    G=SimpleNamespace(lon=lon,lat=lat,lonlat_data=g.reshape(-1,2),
        lonlat_center=np.stack(((lon[:-1,:-1]+lon[1:,1:])/2,(lat[:-1,:-1]+lat[1:,1:])/2),axis=-1).reshape(-1,2),
        land_mask=np.zeros(lon.shape,bool))
    interp=N4GLORYS(G,x=query[:,0],y=query[:,1],tri=np.array([[0,1,2]]),
                    uv=uv,land_check=True,proj=lambda x,y:(x,y))
    interp.nearest4()
    np.testing.assert_allclose(np.sum(g.reshape(-1,2)[interp.rho_index]*interp.rho_coef[:,:,None],axis=1),query,atol=1e-12)
    if uv:
        assert interp.uv_index.shape==(1,4)


def test_roms_nesting_adapter_uses_topological_rho_u_v():
    from fvtools.interpolators.roms_interpolators import N4ROMSNESTING
    rho=grid(7,9,angle=.3,skew=True)
    u=(rho[:,:-1]+rho[:,1:])/2
    v=(rho[:-1,:]+rho[1:,:])/2
    xy=np.stack([in_cell(rho,2,3,.25,.3),in_cell(rho,2,4,.3,.4),in_cell(rho,3,4,.4,.4)])
    class R:pass
    r=R()
    for name,g in (('rho',rho),('u',u),('v',v)):
        setattr(r,f'cropped_x_{name}',g[:,:,0].ravel())
        setattr(r,f'cropped_y_{name}',g[:,:,1].ravel())
        setattr(r,f'fv_{name}_mask',np.ones(g.shape[:2],bool))
        setattr(r,f'Land_{name}',np.zeros(g.size//2,bool))
    n=N4ROMSNESTING(r,x=xy[:,0],y=xy[:,1],tri=np.array([[0,1,2]]),uv=True,land_check=False)
    n.nearest4()
    cell=xy.mean(axis=0)[None,:]
    for name,g,targets in (('rho',rho,xy),('u',u,cell),('v',v,cell)):
        ids=getattr(n,f'{name}_index');w=getattr(n,f'{name}_coef')
        np.testing.assert_allclose(np.sum(g.reshape(-1,2)[ids]*w[:,:,None],axis=1),targets,atol=1e-8)


def test_era5_style_rectilinear_grid_uses_atmospheric_adapter():
    from fvtools.interpolators.horizontal_interpolator import N4Coefficients
    lon,lat=np.meshgrid(np.linspace(0.,500.,9),np.linspace(0.,300.,7))
    g=np.stack((lon,lat),axis=-1)
    q=np.stack([in_cell(g,2,3,.2,.4),in_cell(g,3,4,.3,.6),in_cell(g,3,5,.3,.1)])
    mask=np.ones(lon.shape,bool)
    coarse=SimpleNamespace(x=lon,y=lat,
        x_center=.25*(lon[:-1,:-1]+lon[1:,1:]+lon[:-1,1:]+lon[1:,:-1]),
        y_center=.25*(lat[:-1,:-1]+lat[1:,1:]+lat[:-1,1:]+lat[1:,:-1]),
        land_mask=np.ones(lon.shape,bool), crop_grid=lambda xlim,ylim:mask)
    fv=SimpleNamespace(x=q[:,0],y=q[:,1],xc=q[:1,0],yc=q[:1,1],cell_utm_angle=np.zeros(1))
    interp=N4Coefficients(fv,coarse)
    interp.compute_nearest4()
    np.testing.assert_allclose(np.sum(g.reshape(-1,2)[interp.nindex]*interp.ncoef[:,:,None],axis=1),q,atol=1e-10)
