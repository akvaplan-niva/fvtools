"""Regression checks for every @njit/@jit production kernel in fvtools.

All sample inputs are small, deterministic and offline. Each test calls the
compiled dispatcher and the undecorated Python implementation on independent
copies, compares outputs and input mutations, and checks successful nopython
specialization. Independent geometry/physics assertions catch mistakes shared
by both implementations. No THREDDS/network requests or benchmark timings.

Update KERNELS whenever a new Numba decorator is introduced: the inventory
check fails when a kernel lacks a test case.
"""
from __future__ import annotations

import ast
import copy
import importlib
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Module path (relative to fvtools/) -> qualified names of *all* active JIT kernels.
KERNELS = {
    "grid/fvcom/tge.py": (
        "get_NBE", "get_BOUNDARY", "get_MAXNBR", "get_NBVE_NBVT",
        "get_NTSN_NBSN", "get_TRI_EDGE_PARAM", "set_boundary", "get_CV",
        "get_NBSE_NESE", "shape_coefficients", "get_art", "get_art1",
        "vorticity_node", "node_gradient", "cell_gradient",
        "fvcom_cell_gradients_speed", "fvcom_cell_gradients_scalar", "streamfunction",
    ),
    "interpolators/nearest4.py": (
        "compute_interpolation_coefficients", "_bilinear_coefficients", "_rotate_subset",
    ),
    "plot/fvcom_streamlines.py": (
        "remove_land", "is_inside_tri", "get_weights", "compute_streamline_linear",
    ),
    "prepro/gridding/estimator.py": ("DistanceFunction._get_node_estimate_function",),
    "prepro/_buildcase/depth.py": ("DepthHandler.laplacian_filter", "DepthHandler.mellor_ezer_oey"),
    "grid/fvcom/_components/control_volumes.py": ("_find_cv_walls", "_connect_cv_walls"),
    "grid/fvcom_grd.py": ("FVCOM_grid._smooth",),
    "prepro/nesting/vertical_interpolation.py": (
        "calc_interp_matrices", "numba_absdiff", "find_inds_and_weights",
    ),
    "interpolators/structured_bilinear.py": ("_solve_batch",),
}
ALL_KERNELS = [(path, name) for path, names in KERNELS.items() for name in names]


def _decorated_names(path):
    """Discover JIT decorators, including methods beneath @staticmethod."""
    source = (ROOT / "fvtools" / path).read_text(encoding="utf-8")
    # Skip unrelated legacy Python 2 scripts; some are not parsable by Python 3.
    # Never skip a module that explicitly imports Numba: parsing failures in
    # candidate modules should still fail the test instead of hiding a kernel.
    if not re.search(r"(?m)^[ \t]*(?:from[ \t]+numba[ \t]+import|import[ \t]+numba\b)", source):
        return []
    tree = ast.parse(source, filename=path)
    found = []

    def visit(node, parents):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(
                    isinstance(d, ast.Name) and d.id in ("njit", "jit", "vectorize", "guvectorize")
                    or isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                    and d.func.id in ("njit", "jit", "vectorize", "guvectorize")
                    for d in child.decorator_list
                ):
                    found.append(".".join((*parents, child.name)))
            elif isinstance(child, ast.ClassDef):
                visit(child, (*parents, child.name))
    visit(tree, ())
    return found


def test_all_numba_decorated_functions_are_accounted_for():
    discovered = {}
    for source in (ROOT / "fvtools").rglob("*.py"):
        path = source.relative_to(ROOT / "fvtools").as_posix()
        decorated = _decorated_names(path)
        if decorated:
            discovered[path] = tuple(decorated)
    assert {p: set(v) for p, v in discovered.items()} == {
        p: set(v) for p, v in KERNELS.items()
    }, "A Numba kernel was added, removed or moved: update the test fixtures."


def _module(path):
    name = "fvtools." + path[:-3].replace("/", ".")
    return importlib.import_module(name)


def _dispatcher(path, qualified):
    obj = _module(path)
    for part in qualified.split("."):
        obj = getattr(obj, part)
    assert hasattr(obj, "py_func"), f"Not a Numba dispatcher: {path}::{qualified}"
    return obj


def _same(a, b):
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b)
        for lhs, rhs in zip(a, b):
            _same(lhs, rhs)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
        assert a.shape == b.shape
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6, equal_nan=True)
        else:
            np.testing.assert_array_equal(a, b)
    elif isinstance(a, (np.number, int, float, bool)):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6, equal_nan=True)
    else:
        assert a == b


def _fan():
    """A 4-triangle clockwise fan with one interior node and 4 boundary nodes."""
    x = np.array([0, 1, 1, 0, .5], dtype=np.float64)
    y = np.array([0, 0, 1, 1, .5], dtype=np.float64)
    tri = np.array([[0, 4, 1], [1, 4, 2], [2, 4, 3], [3, 4, 0]], dtype=np.int64)
    cx, cy = x[tri].mean(axis=1), y[tri].mean(axis=1)
    return x, y, tri, cx, cy


def _topology():
    """Use production connectivity algorithms on the synthetic mesh."""
    from fvtools.grid.fvcom import tge
    x, y, tri, cx, cy = _fan()
    nt, mt = len(tri), len(x)
    nbe = tge.get_NBE(nt, mt, tri)
    isbce, isonb = tge.get_BOUNDARY(nt, mt, nbe, tri)
    ntve_count = tge.get_MAXNBR(mt, nt, tri)
    nbve, nbvt, ntve = tge.get_NBVE_NBVT(mt, nt, tri, int(ntve_count.max()))
    ntsn, nbsn, nbve, nbvt, invalid = tge.get_NTSN_NBSN(
        nbve, ntve, nbvt, nbe, tri, isonb, int(ntve_count.max()), mt
    )
    return locals()


def _args(path, name):
    """A valid deterministic calling convention for each of the 35 kernels."""
    x, y, tri, cx, cy = _fan()
    nt, mt = len(tri), len(x)
    k = name.split(".")[-1]

    if path == "grid/fvcom/tge.py":
        from fvtools.grid.fvcom import tge
        nbe = tge.get_NBE(nt, mt, tri)
        if k == "get_NBE": return (nt, mt, tri)
        if k == "get_BOUNDARY": return (nt, mt, nbe, tri)
        if k == "get_MAXNBR": return (mt, nt, tri)
        if k == "get_NBVE_NBVT": return (mt, nt, tri, 4)
        top = _topology()
        nbve, nbvt, ntve = top["nbve"], top["nbvt"], top["ntve"]
        ntsn, nbsn = top["ntsn"], top["nbsn"]
        isbce, isonb = top["isbce"], top["isonb"]
        if k == "get_NTSN_NBSN":
            v, vt, counts = tge.get_NBVE_NBVT(mt, nt, tri, 4)
            return (v, counts, vt, nbe, tri, isonb, 4, mt)
        if k == "get_TRI_EDGE_PARAM": return (nt, nbe, tri)
        if k == "set_boundary": return (isonb, isbce, tri, nbe, np.array([0, 1], dtype=np.int64), nt)
        if k == "get_NBSE_NESE": return (nbve, ntve, tri, nt)
        if k == "shape_coefficients": return (nt, cx, cy, nbe, isbce)
        if k == "get_art": return (x, y, tri, nt)
        if k == "get_art1": return (nt, mt, x, y, cx, cy, tri, isonb, ntve, nbve, nbvt, nbsn)
        ne, iec, ienode, isbc = tge.get_TRI_EDGE_PARAM(nt, nbe, tri)
        xx = (x[ienode[:, 0]] + x[ienode[:, 1]]) / 2
        yy = (y[ienode[:, 0]] + y[ienode[:, 1]]) / 2
        if k == "get_CV": return (xx, yy, cx, cy, isbc, iec, ienode, ne, nt, mt)
        if k == "vorticity_node":
            # Zero current guarantees zero curl regardless of edge orientation.
            e = tge.get_CV(xx, yy, cx, cy, isbc, iec, ienode, ne, nt, mt)
            _, _, ntrg, niec, dltxe, dltye, _, _, ncv = e
            art1 = tge.get_art1(nt, mt, x, y, cx, cy, tri, isonb, ntve, nbve, nbvt, nbsn)
            return (np.zeros(nt), np.zeros(nt), ncv, mt, ntrg, niec,
                    dltxe, dltye, art1, isonb, ntsn, nbsn)
        if k == "node_gradient": return (2*x + 3*y, x, y, mt, nbsn, ntsn, isonb)
        if k == "cell_gradient": return (2*cx + 3*cy, cx, cy, nt, nbe, isbce, np.arange(nt))
        if k in ("fvcom_cell_gradients_speed", "fvcom_cell_gradients_scalar"):
            a1, a2 = tge.shape_coefficients(nt, cx, cy, nbe, isbce)
            if k.endswith("speed"):
                return (nt, nbe, a1, a2, 2*cx + cy, cx - cy, np.arange(nt))
            return (nt, nbe, a1, a2, 2*cx + cy, np.arange(nt))
        if k == "streamfunction":
            return (mt, x, y, nbsn, ntve, ntsn, nbve, nbvt, isonb,
                    np.zeros(nt), np.zeros(nt), 1.0, False)
    elif path == "interpolators/nearest4.py":
        pts = np.array([[0., 0.], [1., .1], [.9, 1.1], [-.1, 1.]], dtype=np.float64)
        if k == "_rotate_subset" or k == "_bilinear_coefficients":
            return (pts[:, 0], pts[:, 1], .45, .55)
        return (np.array([[0, 1, 2, 3]], dtype=np.int64), pts,
                np.array([[.45, .55]]), 1)
    elif path == "plot/fvcom_streamlines.py":
        from fvtools.grid.fvcom import tge
        nbve, _, ntve = tge.get_NBVE_NBVT(mt, nt, tri, 4)
        nbse, nese = tge.get_NBSE_NESE(nbve, ntve, tri, nt)
        initial = np.array([[.5, .10]], dtype=np.float64)
        cell = np.array([0], dtype=np.int64)
        if k == "is_inside_tri": return (x[tri[0]], y[tri[0]], .5, .10)
        if k == "get_weights": return (x[tri[0]], y[tri[0]], .5, .10)
        if k == "remove_land": return (x, y, tri, nbse, nese, initial, cell)
        if k == "compute_streamline_linear":
            return (x, y, tri, nbse, nese, np.ones(nt, dtype=np.float64),
                    np.ones(nt, dtype=np.int32), initial, 4,
                    np.zeros(mt), np.zeros(mt), cell)
    elif path == "prepro/gridding/estimator.py":
        return (1.0, 5.0, 10.0, 2.0, 3.0, 5.0,
                np.array([[1., 2.], [3., 4.]]),
                np.array([[0., 0.], [10., 0.]]), np.array([2., 3.]), 1.0)
    elif path == "prepro/_buildcase/depth.py":
        neigh = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        counts = np.array([2, 2, 2], dtype=np.int64)
        z = np.array([20., 2., 5.])
        if k == "laplacian_filter": return (np.arange(3.), np.arange(3.), z, neigh, counts, .25)
        return (z, counts, neigh, .2, np.arange(3))
    elif path == "grid/fvcom/_components/control_volumes.py":
        # One boundary node, two adjacent elements, 3 ordered neighbours.
        nodes = np.array([1, 2, 3], dtype=np.int64)
        elements = np.array([0, 1], dtype=np.int64)
        xx = np.array([0., 1., 0., -1.])
        yy = np.array([0., 0., 1., 0.])
        xc = np.array([.5, -.5]); yc = np.array([.5, .5])
        if k == "_find_cv_walls": return (0, nodes, elements, xx, yy, xc, yc)
        return ((xx[0]+xx[nodes])/2, (yy[0]+yy[nodes])/2,
                xc[elements], yc[elements], xx, yy, nodes, elements, 0)
    elif path == "grid/fvcom_grd.py":
        return (np.array([3., 6., 9.]), .25, 2,
                np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                np.array([2, 2, 2], dtype=np.int64))
    elif path == "prepro/nesting/vertical_interpolation.py":
        source = np.array([[0., 0.], [10., 12.], [20., 24.]])
        target = np.array([[0., 3.], [5., 12.], [20., 30.]])
        if k == "calc_interp_matrices": return (source, target)
        if k == "numba_absdiff": return (target[:, 0], source[:, 0])
        if k == "find_inds_and_weights":
            return (np.array([0, 1, 2], dtype=np.int64), source[:, 0],
                    np.array([0., 7., 20.]))
    elif path == "interpolators/structured_bilinear.py":
        pts = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        return (np.array([[.25, .75], [2., 2.]]),
                np.array([[0], [0]], dtype=np.int64),
                np.array([[0, 1, 2, 3]], dtype=np.int64), pts)
    raise AssertionError(f"Missing sample inputs: {path}::{name}")


def _same_structured_bilinear_batch(actual, expected):
    """Ignore intentionally uninitialized indices for rejected source cells.

    _solve_batch initializes indices with np.empty(), but sets found=False for
    targets outside all source quadrilaterals. Only indices where found=True
    are meaningful; weights and found must agree for all targets.
    """
    actual_indices, actual_weights, actual_found = actual
    expected_indices, expected_weights, expected_found = expected
    np.testing.assert_array_equal(actual_found, expected_found)
    assert actual_found.dtype.kind == "b"
    assert actual_indices.shape == expected_indices.shape
    _same(actual_indices[actual_found], expected_indices[expected_found])
    _same(actual_weights, expected_weights)


@pytest.mark.parametrize("path,name", ALL_KERNELS, ids=[f"{p}::{n}" for p, n in ALL_KERNELS])
def test_numba_compiled_matches_python_and_compiles_in_nopython(path, name):
    dispatcher = _dispatcher(path, name)
    arguments = _args(path, name)
    compiled_inputs = copy.deepcopy(arguments)
    python_inputs = copy.deepcopy(arguments)
    expected = dispatcher.py_func(*python_inputs)
    result = dispatcher(*compiled_inputs)
    if (path, name) == ("interpolators/structured_bilinear.py", "_solve_batch"):
        _same_structured_bilinear_batch(result, expected)
    else:
        _same(result, expected)
    for a, b in zip(compiled_inputs, python_inputs):
        _same(a, b)  # mutation equivalence is important for TGE connectivities
    assert dispatcher.nopython_signatures, f"{name} did not compile in nopython mode"


def test_topology_independent_oracle():
    from fvtools.grid.fvcom import tge
    x, y, tri, _, _ = _fan()
    neighbors = tge.get_NBE(4, 5, tri)
    assert np.count_nonzero(neighbors == -1) == 4
    for i in range(4):
        assert sorted(neighbors[i][neighbors[i] >= 0].tolist()) == sorted(((i-1) % 4, (i+1) % 4))
    np.testing.assert_array_equal(tge.get_MAXNBR(5, 4, tri), [2, 2, 2, 2, 4])
    np.testing.assert_allclose(tge.get_art(x, y, tri, 4), np.full(4, .25))
    isbce, isonb = tge.get_BOUNDARY(4, 5, neighbors, tri)
    np.testing.assert_array_equal(isbce, np.ones(4))
    np.testing.assert_array_equal(isonb, [1, 1, 1, 1, 0])


def test_vertical_interpolation_weights_reconstruct_depth():
    from fvtools.prepro.nesting.vertical_interpolation import calc_interp_matrices
    source = np.array([[0., 0.], [10., 12.], [20., 24.]])
    target = np.array([[3., 6.], [17., 18.]])
    i, j, w, v = calc_interp_matrices(source, target)
    assert np.all(w >= 0) and np.all(v >= 0)
    np.testing.assert_allclose(w + v, 1, atol=1e-13)
    np.testing.assert_allclose(
        np.take_along_axis(source, i, axis=0) * w +
        np.take_along_axis(source, j, axis=0) * v, target, atol=1e-12)


def test_bilinear_weights_and_outside_mask():
    from fvtools.interpolators.structured_bilinear import _solve_batch
    args = _args("interpolators/structured_bilinear.py", "_solve_batch")
    indices, weights, valid = _solve_batch(*args)
    assert bool(valid[0]) and not bool(valid[1])
    np.testing.assert_array_equal(indices[0], [0, 1, 2, 3])
    np.testing.assert_allclose(weights[0], [.1875, .0625, .1875, .5625], atol=1e-12)
    np.testing.assert_allclose(weights[0].sum(), 1, atol=1e-12)


def test_triangle_weights_reconstruct_point():
    from fvtools.plot.fvcom_streamlines import get_weights, is_inside_tri
    xv = np.array([0., 1., 0.]); yv = np.array([0., 0., 1.])
    w = get_weights(xv, yv, .2, .3)
    np.testing.assert_allclose(w.sum(), 1, atol=1e-6)
    np.testing.assert_allclose(w @ xv, .2, atol=1e-6)
    np.testing.assert_allclose(w @ yv, .3, atol=1e-6)
    assert is_inside_tri(xv, yv, .2, .3)
    assert not is_inside_tri(xv, yv, 1.2, 1.3)


def test_depth_smoothing_conservation():
    from fvtools.prepro._buildcase.depth import DepthHandler
    args = _args("prepro/_buildcase/depth.py", "DepthHandler.mellor_ezer_oey")
    bath, rmax, fixed = DepthHandler.mellor_ezer_oey(*args)
    assert fixed >= 1 and rmax > 0.2
    np.testing.assert_allclose(bath.sum(), args[0].sum(), atol=1e-12)
    assert (bath > 0).all()


def test_streamfunction_zero_velocity():
    from fvtools.grid.fvcom import tge
    args = _args("grid/fvcom/tge.py", "streamfunction")
    psi = tge.streamfunction(*args)
    np.testing.assert_allclose(psi, 0, atol=1e-7)

@pytest.mark.parametrize("bad_mesh", [
    np.array([[0, 1, 1]], dtype=np.int64),  # degenerate triangle
    np.array([[0, 1, 6]], dtype=np.int64),  # index outside node count
])
def test_tge_rejects_invalid_connectivity(bad_mesh):
    from fvtools.grid.fvcom import tge
    with pytest.raises(ValueError):
        tge.get_NBE(1, 5, bad_mesh)


def test_nearest4_center_weights_are_finite_and_normalized():
    from fvtools.interpolators.nearest4 import _bilinear_coefficients
    pts = np.array([[0., 0.], [1., .1], [.9, 1.1], [-.1, 1.]])
    weights = _bilinear_coefficients(pts[:, 0], pts[:, 1], .45, .55)
    assert np.isfinite(weights).all()
    np.testing.assert_allclose(weights.sum(), 1., rtol=0, atol=1e-12)


def test_numba_smoother_preserves_constant_field():
    from fvtools.grid.fvcom_grd import FVCOM_grid
    neighbours = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    levels = np.full(3, 2, dtype=np.int64)
    constant = np.full(3, 5., dtype=float)
    np.testing.assert_allclose(
        FVCOM_grid._smooth(constant, .25, 3, neighbours, levels), constant)
