"""Topology-aware bilinear interpolation for structured 2D model grids.

Source coordinates are flattened with their corresponding 2D crop mask.
Build quadrilaterals from actual (i, j) adjacency, not by guessing four
corners from an arbitrary kNN search radius. Coordinates use a common projected CRS.
"""

import numpy as np
from numba import njit
from scipy.spatial import cKDTree


@njit(cache=True)
def _solve_batch(targets, nearest_cells, cells, points):
    """Inverse bilinear transform; reject candidates outside their quadrilateral."""
    count = len(targets)
    indices = np.empty((count, 4), dtype=np.int64)
    weights = np.zeros((count, 4), dtype=np.float64)
    found = np.zeros(count, dtype=np.bool_)

    for i in range(count):
        x, y = targets[i, 0], targets[i, 1]
        for cell_id in nearest_cells[i]:
            corners = cells[cell_id]
            x00, y00 = points[corners[0], 0], points[corners[0], 1]
            ax = points[corners[1], 0] - x00
            ay = points[corners[1], 1] - y00
            bx = points[corners[3], 0] - x00
            by = points[corners[3], 1] - y00
            cx = points[corners[2], 0] - points[corners[1], 0] - points[corners[3], 0] + x00
            cy = points[corners[2], 1] - points[corners[1], 1] - points[corners[3], 1] + y00
            px, py = x-x00, y-y00
            scale = max(np.hypot(ax, ay), np.hypot(bx, by), np.hypot(ax+cx, ay+cy), np.hypot(bx+cx, by+cy))
            if scale <= 0 or not np.isfinite(scale):
                continue

            # Affine approximation gives an excellent initial guess for ROMS quads.
            det0 = ax*by - ay*bx
            if abs(det0) <= 1e-13*scale*scale:
                continue
            s = (px*by-py*bx)/det0
            t = (ax*py-ay*px)/det0
            if not np.isfinite(s) or not np.isfinite(t):
                continue

            valid = True
            for _ in range(20):
                fx = s*ax+t*bx+s*t*cx-px
                fy = s*ay+t*by+s*t*cy-py
                if np.hypot(fx, fy) <= 1e-11*scale:
                    break
                jsx, jsy = ax+t*cx, ay+t*cy
                jtx, jty = bx+s*cx, by+s*cy
                det = jsx*jty-jsy*jtx
                if abs(det) <= 1e-13*scale*scale:
                    valid = False
                    break
                ds = (fx*jty-fy*jtx)/det
                dt = (jsx*fy-jsy*fx)/det
                s -= ds
                t -= dt
                if not np.isfinite(s) or not np.isfinite(t):
                    valid = False
                    break

            if not valid:
                continue
            residual = np.hypot(s*ax+t*bx+s*t*cx-px,
                                s*ay+t*by+s*t*cy-py)
            if residual > 1e-9*scale or s < -1e-9 or s > 1+1e-9 or t < -1e-9 or t > 1+1e-9:
                continue

            s = min(max(s, 0.), 1.)
            t = min(max(t, 0.), 1.)
            indices[i] = corners
            weights[i, 0] = (1-s)*(1-t)
            weights[i, 1] = s*(1-t)
            weights[i, 2] = s*t
            weights[i, 3] = (1-s)*t
            found[i] = True
            break
    return indices, weights, found


def structured_bilinear_weights(source_xy, source_mask, target_xy, *, batch_size=20_000):
    """Return (source indices, weights) with shape ``(n_targets, 4)``.

    ``source_xy`` must contain points selected by ``source_mask`` in NumPy
    C-order (as for source_grid[source_mask]). Cells require all four
    corners in that mask. The mask is a crop/availability mask; land-based
    field corrections belong to the caller. A ValueError indicates no
    containing source cell, never an implicit extrapolation.
    """
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    mask = np.asarray(source_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] < 2 or mask.shape[1] < 2:
        raise ValueError('Source mask must be a 2D grid of at least 2x2')
    if source_xy.shape != (int(mask.sum()), 2):
        raise ValueError('Source coordinates must match flattened source mask')
    if target_xy.ndim != 2 or target_xy.shape[1] != 2:
        raise ValueError('Target coordinates must be shaped (n, 2)')
    if not np.isfinite(source_xy).all() or not np.isfinite(target_xy).all():
        raise ValueError('Source and target coordinates must be finite')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')

    lookup = np.full(mask.shape, -1, dtype=np.int64)
    lookup[mask] = np.arange(mask.sum(), dtype=np.int64)
    # Quad ordering: top-left, top-right, bottom-right, bottom-left
    quads = np.stack((lookup[:-1, :-1], lookup[:-1, 1:],
                      lookup[1:, 1:], lookup[1:, :-1]), axis=-1).reshape(-1, 4)
    quads = np.ascontiguousarray(quads[np.all(quads >= 0, axis=1)])
    if not len(quads):
        raise ValueError('No complete 2x2 cells within cropped source mask')
    centers = source_xy[quads].mean(axis=1)
    tree = cKDTree(centers)

    # A convex polygon lies in the disk centered on its vertex-mean with
    # radius max(distance from center to a vertex). Thus searching all
    # centers inside the global maximum radius is a COMPLETE fallback,
    # unlike an arbitrary cap of 512 nearest centers.
    max_radius = float(np.max(np.linalg.norm(source_xy[quads] - centers[:, None, :], axis=2)))
    max_radius = np.nextafter(max_radius, np.inf)

    out_idx = np.empty((len(target_xy), 4), dtype=np.int64)
    out_weights = np.empty((len(target_xy), 4), dtype=np.float64)
    missing_count = 0
    missing_samples = []
    for begin in range(0, len(target_xy), batch_size):
        end = min(begin+batch_size, len(target_xy))
        targets = np.ascontiguousarray(target_xy[begin:end])
        pending = np.arange(len(targets))
        idx = np.empty((len(targets), 4), dtype=np.int64)
        w = np.full((len(targets), 4), np.nan, dtype=np.float64)
        for k in (8, 32, 128):
            if not len(pending):
                break
            k = min(k, len(quads))
            nearest = tree.query(targets[pending], k=k, workers=1)[1]
            nearest = np.asarray(nearest, dtype=np.int64).reshape((len(pending), k))
            partial_idx, partial_w, found = _solve_batch(
                targets[pending], nearest, quads, source_xy)
            idx[pending[found]] = partial_idx[found]
            w[pending[found]] = partial_w[found]
            pending = pending[~found]
            if k == len(quads):
                break
        # Guaranteed-complete candidate enumeration for unresolved targets.
        # Typically almost no points reach this fallback, so a variable-size
        # radius query is cheaper here than allocating a large global kNN.
        for p in pending:
            candidate = tree.query_ball_point(targets[p], r=max_radius)
            if not candidate:
                continue
            candidate = np.asarray(candidate, dtype=np.int64)
            # Cheap bbox rejection before the inverse map.
            q = source_xy[quads[candidate]]
            tol = max(1e-9, 1e-10 * max_radius)
            inside_box = (np.all(targets[p] >= q.min(axis=1) - tol, axis=1) &
                          np.all(targets[p] <= q.max(axis=1) + tol, axis=1))
            candidate = candidate[inside_box]
            if not len(candidate):
                continue
            # Visit candidates in nearest-center order for determinism.
            dist = np.sum((centers[candidate] - targets[p]) ** 2, axis=1)
            candidate = candidate[np.argsort(dist, kind='stable')]
            pi, pw, ok = _solve_batch(targets[p:p+1], candidate[None, :], quads, source_xy)
            if ok[0]:
                idx[p], w[p] = pi[0], pw[0]
        # Mark resolved positions explicitly rather than assuming the
        # intermediate kNN search handled the entire batch.
        resolved = np.isfinite(w).all(axis=1) & np.isclose(w.sum(axis=1), 1., atol=1e-12)
        pending = np.flatnonzero(~resolved)
        if len(pending):
            missing_count += len(pending)
            if len(missing_samples) < 5:
                missing_samples.extend(targets[pending[:5-len(missing_samples)]].tolist())
        out_idx[begin:end] = idx
        out_weights[begin:end] = w

    if missing_count:
        raise ValueError(f'{missing_count} FVCOM targets fall outside complete structured-grid interpolation cells '
                         f'(or inside a crop hole); examples (x,y): {missing_samples}')
    return out_idx, out_weights
