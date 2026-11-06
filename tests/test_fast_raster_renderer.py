"""Small deterministic tests; no FVCOM output files needed."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pytest

from fvtools.plot._fast_raster import FastRasterRenderer


def _mesh():
    return mtri.Triangulation([0., 1., 1., 0.], [0., 0., 1., 1.],
                              triangles=np.array([[0, 1, 2], [0, 2, 3]]))


def _renderer(ax, mesh):
    return FastRasterRenderer(ax, mesh, cmap=plt.get_cmap('viridis'),
                              levels=np.linspace(0, 5, 6),
                              bounds=(0., 1., 0., 1.), resolution=32)


def test_linear_field_interpolates_barycentrically():
    fig, ax = plt.subplots()
    try:
        ren = _renderer(ax, _mesh())
        # Values f(x,y)=x+2y interpolate exactly throughout both triangles.
        artist = ren.update(np.array([0., 1., 3., 2.]))
        assert artist is ren.artist
        image = artist.get_array()
        xv = (np.arange(ren.width) + 0.5) / ren.width
        yv = (np.arange(ren.height) + 0.5) / ren.height
        expected = xv[None,:] + 2*yv[:,None]
        np.testing.assert_allclose(image, expected, rtol=0, atol=1e-6)
        old_artist = ren.artist
        ren.update(np.full(4, 1.))
        assert ren.artist is old_artist
        assert len(ax.images) == 1
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_masking_preserves_holes_and_all_invalid():
    fig, ax = plt.subplots()
    try:
        ren = _renderer(ax, _mesh())
        ren.update(np.ma.array([0., 1., 3., 2.], mask=[0, 1, 0, 0]))
        img = ren.artist.get_array()
        assert 0 < np.ma.count(img) < img.size
        assert ren.update([np.nan]*4) is None
        assert np.ma.count(ren.artist.get_array()) == 0
    finally:
        plt.close(fig)


def test_reject_invalid_sizes():
    fig, ax = plt.subplots()
    try:
        ren = _renderer(ax, _mesh())
        with pytest.raises(ValueError):
            ren.update([1,2])
    finally:
        plt.close(fig)


def test_geometry_can_be_shared_between_fields():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    try:
        old = _renderer(ax1, _mesh())
        new = FastRasterRenderer(
            ax2, _mesh(), cmap=plt.get_cmap('magma'),
            levels=np.linspace(0, 5, 6), bounds=(0., 1., 0., 1.),
            resolution=32, geometry=old.geometry(),
        )
        assert new.nodes is old.nodes
        assert new.w0 is old.w0
        assert new.update(np.array([0., 1., 3., 2.])) is new.artist
    finally:
        plt.close(fig1)
        plt.close(fig2)
