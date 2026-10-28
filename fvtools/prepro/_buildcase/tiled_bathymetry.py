"""Memory-limited linear interpolation from large unstructured (N,3) .npy data.

A disk-backed uniform-bin index is constructed in two streaming passes. Local
Delaunay triangulations are expanded until each queried simplex is provably
unaffected by bathymetry points outside the local window (circumcircle test).
No globally sized point tree or Delaunay triangulation is created in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.spatial import Delaunay, ConvexHull, QhullError, cKDTree


@dataclass(frozen=True)
class TilingOptions:
    """Limits in *points*, rather than relying on potentially unsafe RAM guesses."""

    source_chunk_rows: int = 250_000
    max_triangulation_points: int = 350_000
    nodes_per_tile: int = 4096
    bins_per_axis: int = 64
    initial_halo_fraction: float = 0.15
    workers: int | None = None


class _TooManyPoints(Exception):
    pass


class _Progress:
    """Notebook-friendly progress reporting without requiring ipywidgets.

    Reports every 10 percentage points, or approximately every 10 seconds when
    the current unit of work finishes. Messages are printed on separate lines
    because carriage-return progress bars are unreliable in Jupyter output.
    """

    def __init__(self, label, total):
        self.label = label
        self.total = int(total)
        self.started = perf_counter()
        self.last_report = self.started
        self.last_milestone = -1
        self.update(0, force=True)

    @staticmethod
    def _duration(seconds):
        seconds = max(0, int(seconds))
        if seconds >= 3600:
            return f'{seconds // 3600}h {(seconds // 60) % 60:02d}m'
        if seconds >= 60:
            return f'{seconds // 60}m {seconds % 60:02d}s'
        return f'{seconds}s'

    def update(self, done, *, force=False):
        now = perf_counter()
        done = max(0, min(int(done), self.total))
        percent = 100. * done / self.total if self.total else 100.
        milestone = int(percent) // 10
        if (not force and milestone <= self.last_milestone
                and now - self.last_report < 10.):
            return
        self.last_report = now
        self.last_milestone = max(self.last_milestone, milestone)
        elapsed = now - self.started
        msg = (f'  - {self.label}: {percent:.1f}% '
               f'({done:,}/{self.total:,}) | elapsed {self._duration(elapsed)}')
        # Approximate ETA for this stage, after a useful sample is available.
        if 0 < done < self.total and elapsed >= 10.:
            remaining = elapsed * (self.total - done) / done
            msg += f' | ETA ~{self._duration(remaining)}'
        print(msg, flush=True)

    def activity(self, message, *, force=False):
        """Show activity when a single halo search/triangulation takes a while."""
        now = perf_counter()
        if force or now - self.last_report >= 10.:
            self.last_report = now
            print(f'    {message} | elapsed {self._duration(now-self.started)}',
                  flush=True)


def _hull_vertices(points):
    """Vertices of a streamed subset, preserving the exact global hull."""
    xy = np.asarray(points, dtype=np.float64)
    if len(xy) < 3:
        return xy.copy()
    try:
        return xy[ConvexHull(xy).vertices]
    except QhullError:
        # Collinear chunks have only two extreme points; keep both.
        axis = int(np.argmax(np.ptp(xy, axis=0)))
        return xy[[np.argmin(xy[:, axis]), np.argmax(xy[:, axis])]]


def _outside_convex_hull(xy, equations, extent):
    """Mask nodes outside the global source hull, with boundary tolerance."""
    if equations is None:
        return np.zeros(len(xy), dtype=bool)
    tol = 64 * np.finfo(float).eps * max(float(extent), 1.)
    # Hull equations are affine halfplanes; evaluate each separately to
    # avoid an N_query x N_hull temporary allocation.
    outside = np.zeros(len(xy), dtype=bool)
    for a, b, c in equations:
        outside |= a * xy[:, 0] + b * xy[:, 1] + c > tol
    return outside


def _bins_for_xy(x, y, bounds, count):
    xmin, ymin, xmax, ymax = bounds
    dx, dy = max(xmax - xmin, np.finfo(float).eps), max(ymax - ymin, np.finfo(float).eps)
    ix = np.clip(((x - xmin) / dx * count).astype(np.int64), 0, count - 1)
    iy = np.clip(((y - ymin) / dy * count).astype(np.int64), 0, count - 1)
    return ix * count + iy


class _DiskBins:
    """Two-pass on-disk spatial bin index of cropped XYZ points."""

    def __init__(self, data, bounds, directory, options):
        print(f'  - Indexing {len(data):,} bathymetry points on disk (two passes)', flush=True)
        self.data = data
        self.bounds = bounds
        self.count = options.bins_per_axis
        self.chunk_rows = options.source_chunk_rows
        self.nbin = self.count ** 2
        self.counts = np.zeros(self.nbin, dtype=np.int64)
        xmin, ymin, xmax, ymax = bounds
        self.global_extent = [np.inf, np.inf, -np.inf, -np.inf]
        self.nonfinite_rows = 0
        # Maintain the exact convex hull incrementally without storing the cloud.
        hull_points = np.empty((0, 2), dtype=np.float64)

        index_pass = _Progress('Index pass 1/2 – counting source points', len(data))
        for start in range(0, len(data), self.chunk_rows):
            stop = min(start + self.chunk_rows, len(data))
            xyz = np.asarray(data[start:stop, :3])
            valid = np.isfinite(xyz).all(axis=1)
            self.nonfinite_rows += int((~valid).sum())
            if not valid.any():
                index_pass.update(stop)
                continue
            points = xyz[valid]
            self.global_extent[0] = min(self.global_extent[0], float(points[:, 0].min()))
            self.global_extent[1] = min(self.global_extent[1], float(points[:, 1].min()))
            self.global_extent[2] = max(self.global_extent[2], float(points[:, 0].max()))
            self.global_extent[3] = max(self.global_extent[3], float(points[:, 1].max()))
            inside = (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
            points = points[inside]
            if len(points):
                # hull(A union B) == hull(hull(A) union hull(B)); using
                # vertices from each streamed chunk is exact, not a sample.
                chunk_hull = _hull_vertices(points[:, :2])
                hull_points = _hull_vertices(np.concatenate((hull_points, chunk_hull)))
                ids = _bins_for_xy(points[:, 0], points[:, 1], bounds, self.count)
                self.counts += np.bincount(ids, minlength=self.nbin)
            index_pass.update(stop)

        total = int(self.counts.sum())
        self.hull_equations = None
        self.hull_vertices = hull_points  # Disk fallback can enclose sparse-gap queries.
        if len(hull_points) >= 3:
            try:
                self.hull_equations = ConvexHull(hull_points).equations
            except QhullError:
                pass
        if total < 3:
            raise ValueError('Fewer than 3 finite bathymetry points are inside the mesh crop area')
        self.offsets = np.r_[0, np.cumsum(self.counts)]
        self.path = Path(directory) / 'bathymetry_spatial_index.bin'
        self.dtype = data.dtype
        self.record_bytes = 3 * data.dtype.itemsize
        with self.path.open('wb') as f:
            f.truncate(total * self.record_bytes)
        cursor = self.offsets[:-1].copy()
        index_pass = _Progress('Index pass 2/2 – writing disk index', len(data))
        with self.path.open('r+b') as dest:
          for start in range(0, len(data), self.chunk_rows):
            stop = min(start + self.chunk_rows, len(data))
            xyz = np.asarray(data[start:stop, :3])
            valid = np.isfinite(xyz).all(axis=1)
            inside = valid & (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
            if not inside.any():
                index_pass.update(stop)
                continue
            points = xyz[inside]
            bins = _bins_for_xy(points[:, 0], points[:, 1], bounds, self.count)
            sorted_order = np.argsort(bins, kind='stable')
            sorted_bins = bins[sorted_order]
            unique, first, sizes = np.unique(sorted_bins, return_index=True, return_counts=True)
            for b, first_idx, size in zip(unique, first, sizes):
                pos = cursor[b]
                dest.seek(int(pos) * self.record_bytes)
                block = np.ascontiguousarray(points[sorted_order[first_idx:first_idx + size]])
                dest.write(block.tobytes())
                cursor[b] += size
            index_pass.update(stop)
        if not np.array_equal(cursor, self.offsets[1:]):
            raise RuntimeError('Disk index construction incomplete')
        self.total = total

    def region(self, bounds, limit):
        """Return source points within rectangle without materializing other bins."""
        x0, y0, x1, y1 = bounds
        xmin, ymin, xmax, ymax = self.bounds
        nx = self.count
        def coord(value, lo, hi):
            return int(np.clip(np.floor((value - lo) / (hi - lo) * nx), 0, nx-1))
        ix0, ix1 = coord(x0,xmin,xmax), coord(x1,xmin,xmax)
        iy0, iy1 = coord(y0,ymin,ymax), coord(y1,ymin,ymax)
        chunks = []
        count = 0
        with self.path.open('rb') as f:
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    b = ix * nx + iy
                    lo, hi = int(self.offsets[b]), int(self.offsets[b+1])
                    # Read at most 64k consecutive rows from the temporary file.
                    for start in range(lo, hi, 65_536):
                        n = min(start+65_536,hi)-start
                        f.seek(start * self.record_bytes)
                        points = np.fromfile(f, dtype=self.dtype, count=n*3).reshape(n,3)
                        keep = (points[:,0]>=x0)&(points[:,0]<=x1)&(points[:,1]>=y0)&(points[:,1]<=y1)
                        if keep.any():
                            selected = points[keep]
                            count += len(selected)
                            if count > limit:
                                raise _TooManyPoints
                            chunks.append(selected)
        if not chunks:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(chunks, axis=0)


def _circumcircles_certified(tri, simplex_ids, window, whole):
    """An empty circumcircle wholly inside the available source window is
    also empty in the globally cropped bathymetry point cloud.

    If the local window reaches a boundary of the global crop, no missing
    source points can occur beyond that side of the circumcircle.
    """
    simplices = tri.simplices[simplex_ids]
    p = tri.points[simplices]  # (n,3,2)
    a = p[:,1] - p[:,0]
    b = p[:,2] - p[:,0]
    aa = (a*a).sum(axis=1)*0.5
    bb = (b*b).sum(axis=1)*0.5
    det = a[:,0]*b[:,1]-a[:,1]*b[:,0]
    safe = np.abs(det) > 64*np.finfo(float).eps * np.maximum((a*a).sum(axis=1)+(b*b).sum(axis=1), 1.)
    centers = np.full((len(p),2), np.nan)
    centers[safe,0] = p[safe,0,0] + (aa[safe]*b[safe,1]-a[safe,1]*bb[safe])/det[safe]
    centers[safe,1] = p[safe,0,1] + (a[safe,0]*bb[safe]-aa[safe]*b[safe,0])/det[safe]
    radii = np.linalg.norm(centers-p[:,0],axis=1)
    left,bottom,right,top = window
    wl,wb,wr,wt = whole
    tol = 2e-12 * max(wr-wl,wt-wb,1.)
    return (safe & ((centers[:,0]-radii >= left+tol) | (left<=wl))
                 & ((centers[:,1]-radii >= bottom+tol) | (bottom<=wb))
                 & ((centers[:,0]+radii <= right-tol) | (right>=wr))
                 & ((centers[:,1]+radii <= top-tol) | (top>=wt)))



def _certify_no_omitted_points(tri, simplex_ids, full_tree):
    """Prove the candidate Delaunay triangle is empty in the FULL local cloud.

    Sparse triangulation uses only a union of k-NN points. Its circumcircle
    must contain no other bathymetry point, including those omitted from the
    sparse triangulation. The full local cKDTree provides this exact test.
    """
    vertices = tri.points[tri.simplices[simplex_ids]]
    a = vertices[:, 1] - vertices[:, 0]
    b = vertices[:, 2] - vertices[:, 0]
    a2 = (a*a).sum(axis=1) * 0.5
    b2 = (b*b).sum(axis=1) * 0.5
    det = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
    safe = np.abs(det) > 64 * np.finfo(float).eps * np.maximum(2*(a2+b2), 1.)
    passed = np.zeros(len(simplex_ids), dtype=bool)
    if not safe.any():
        return passed
    center = np.empty((safe.sum(),2), dtype=float)
    center[:, 0] = vertices[safe, 0, 0] + (
        a2[safe]*b[safe,1] - a[safe,1]*b2[safe])/det[safe]
    center[:, 1] = vertices[safe, 0, 1] + (
        a[safe,0]*b2[safe] - a2[safe]*b[safe,0])/det[safe]
    radius = np.linalg.norm(center - vertices[safe, 0], axis=1)
    dist, _ = full_tree.query(center, k=1)
    # Slack is on the scale of the local simplex, not the entire domain.
    tol = 5e-12 * np.maximum(radius,1.)
    passed[safe] = dist >= radius-tol
    return passed


def _linear_interpolate(tri, xyz, xy, simplex_ids):
    """SciPy's barycentric transform, equivalent to LinearNDInterpolator."""
    transforms = tri.transform[simplex_ids]
    delta = xy - transforms[:,2]
    bary = np.einsum('nij,nj->ni', transforms[:,:2], delta)
    weights = np.column_stack((bary, 1. - bary.sum(axis=1)))
    return (xyz[tri.simplices[simplex_ids],2] * weights).sum(axis=1)


# The worker entry points are module-level functions: Windows and notebooks use
# multiprocessing spawn, which cannot pickle functions nested inside a method.
_WORKER_STATE = None


def _init_worker(index_info, mesh_path, options, whole, limits):
    """Open shared *read-only* disk files once per spawned process.

    No bathymetry source array or large mesh-coordinate array is pickled.
    The spatial index is constructed by the parent exactly once.
    """
    global _WORKER_STATE
    idx = object.__new__(_DiskBins)
    idx.path = Path(index_info['path'])
    idx.offsets = index_info['offsets']
    idx.bounds = index_info['bounds']
    idx.count = index_info['count']
    idx.record_bytes = index_info['record_bytes']
    idx.dtype = np.dtype(index_info['dtype'])
    idx.hull_vertices = index_info.get('hull_vertices')
    _WORKER_STATE = (idx, np.load(mesh_path, mmap_mode='r'), options, whole, limits)


def _ordered_bins_for_point(idx, target):
    """Occupied disk bins in increasing *minimum possible* distance to target.

    The bounding-box distance is a rigorous lower bound for all points in a
    bin. It lets k-NN stop without reading the rest of a large source file.
    """
    xmin, ymin, xmax, ymax = idx.bounds
    n = idx.count
    ids = np.flatnonzero(np.diff(idx.offsets))
    ix, iy = ids // n, ids % n
    dx = (xmax - xmin) / n
    dy = (ymax - ymin) / n
    lx = xmin + ix * dx
    ly = ymin + iy * dy
    ux = lx + dx
    uy = ly + dy
    distance2 = np.maximum(np.maximum(lx - target[0], target[0] - ux), 0.)**2
    distance2 += np.maximum(np.maximum(ly - target[1], target[1] - uy), 0.)**2
    order = np.argsort(distance2, kind='stable')
    return ids[order], distance2[order]


def _read_bin_chunks(file, idx, bin_id):
    """Read a disk bin in bounded chunks (never a whole dense bin at once)."""
    lo, hi = int(idx.offsets[bin_id]), int(idx.offsets[bin_id + 1])
    for start in range(lo, hi, 65_536):
        size = min(65_536, hi - start)
        file.seek(start * idx.record_bytes)
        yield np.fromfile(file, dtype=idx.dtype, count=3 * size).reshape(size, 3)


def _nearest_on_disk(idx, target, k):
    """Exact k-nearest source points using RAM bounded by one disk chunk.

    Crucially, this routine does not construct a KDTree of the full window.
    """
    target = np.asarray(target, dtype=np.float64)
    bins, lower_bounds2 = _ordered_bins_for_point(idx, target)
    best = np.empty((0, 3), dtype=idx.dtype)
    dist2_best = np.empty(0, dtype=np.float64)
    with idx.path.open('rb') as handle:
        for bin_id, lower2 in zip(bins, lower_bounds2):
            if len(dist2_best) >= k and lower2 > dist2_best.max():
                break
            for points in _read_bin_chunks(handle, idx, int(bin_id)):
                dist2 = np.sum((points[:, :2].astype(np.float64) - target)**2, axis=1)
                if len(dist2_best) >= k:
                    consider = dist2 <= dist2_best.max()
                    if not np.any(consider):
                        continue
                    points, dist2 = points[consider], dist2[consider]
                if len(points) == 0:
                    continue
                if len(points) > k:
                    selected = np.argpartition(dist2, k - 1)[:k]
                    points, dist2 = points[selected], dist2[selected]
                together = np.concatenate((best, points), axis=0)
                distances = np.concatenate((dist2_best, dist2))
                if len(together) > k:
                    selected = np.argpartition(distances, k - 1)[:k]
                    best, dist2_best = together[selected], distances[selected]
                else:
                    best, dist2_best = together, distances
    if not len(best):
        return best
    return best[np.argsort(dist2_best, kind='stable')]


def _simplex_circle(tri, simplex_id):
    """Return circumcenter and radius for a triangle or None if degenerate."""
    p = tri.points[tri.simplices[simplex_id]]
    a, b = p[1] - p[0], p[2] - p[0]
    aa, bb = a @ a / 2., b @ b / 2.
    det = a[0]*b[1] - a[1]*b[0]
    if abs(det) <= 64 * np.finfo(float).eps * max(a@a+b@b, 1.):
        return None
    center = p[0] + np.array([(aa*b[1]-a[1]*bb)/det,
                              (a[0]*bb-aa*b[0])/det])
    radius = float(np.linalg.norm(center-p[0]))
    if not np.isfinite(radius):
        return None
    return center, radius


def _intruders_on_disk(idx, center, radius, max_points=12, *, origin=None):
    """Search the exact circumcircle on disk, returning a few interior points.

    Zero results certifies the simplex against *every* cropped source point;
    the scan never materializes the circle's potentially huge contents.
    """
    tol = 5e-12 * max(radius, 1.)
    threshold = max(0., radius-tol)**2
    if not threshold:
        return np.empty((0, 3), dtype=idx.dtype)
    # A Delaunay triangulation on UTM coordinates can lose precision if its
    # circumcenter is formed in the full projected coordinate system.
    # In the dense fallback, both the triangulation and circumcenter use a
    # query-centered frame. Keep all circle tests in that same frame.
    origin = (np.zeros(2, dtype=np.float64) if origin is None
              else np.asarray(origin, dtype=np.float64))
    absolute_center = center + origin
    bins, minimum_dist2 = _ordered_bins_for_point(idx, absolute_center)
    # Rounding the local center back to UTM units can shift it by a few ulps.
    # The slack affects which bins are visited, NOT the point-in-circle test.
    slack = 8 * float(np.max(np.spacing(np.abs(absolute_center))))
    search_threshold = (max(0., radius - tol) + slack)**2
    found = []
    total = 0
    with idx.path.open('rb') as handle:
        for bin_id, bound2 in zip(bins, minimum_dist2):
            if bound2 >= search_threshold:
                break
            for points in _read_bin_chunks(handle, idx, int(bin_id)):
                # Subtract the same origin from each stored point BEFORE
                # comparing its distance with the local circumcenter.
                local_xy = points[:, :2].astype(np.float64) - origin
                dist2 = np.sum((local_xy - center)**2, axis=1)
                inside = np.flatnonzero(dist2 < threshold)
                if len(inside):
                    take = min(max_points-total, len(inside))
                    found.append(points[inside[:take]])
                    total += take
                    if total >= max_points:
                        return np.concatenate(found, axis=0)
    return (np.concatenate(found, axis=0) if found
            else np.empty((0, 3), dtype=idx.dtype))


def _hull_support_for_query(idx, target):
    """Select just three global hull vertices enclosing a sparse-gap query.

    The cropped hull is computed exactly during disk indexing. A convex fan
    from its first vertex covers the hull without building another Delaunay.
    Hull vertices' XYZ values are recovered using the bounded disk k-NN lookup.
    """
    polygon = getattr(idx, 'hull_vertices', None)
    if polygon is None or len(polygon) < 3:
        return np.empty((0,3),dtype=idx.dtype)
    origin = polygon[0]
    q = np.asarray(target,dtype=np.float64)-origin
    tol = 1e-12 * max(np.ptp(polygon, axis=0).max(), 1.)**2
    for i in range(1,len(polygon)-1):
        a = polygon[i] - origin
        b = polygon[i+1] - origin
        cross = a[0]*b[1]-a[1]*b[0]
        if abs(cross) < tol:
            continue
        v = (q[0]*b[1]-q[1]*b[0])/cross
        w = (a[0]*q[1]-a[1]*q[0])/cross
        if v>=-1e-12 and w>=-1e-12 and v+w<=1+1e-12:
            return np.concatenate([_nearest_on_disk(idx, vxy, 1)
                                   for vxy in (origin,polygon[i],polygon[i+1])])
    return np.empty((0,3),dtype=idx.dtype)


def _interpolate_dense_single(target, idx, options, activity=None):
    """Exact, bounded-memory fallback when even one node exceeds the halo cap.

    Select a small exact k-nearest neighborhood straight from disk. Check the
    containing Delaunay triangle against *all* on-disk bathymetry points, then
    add any circle intruders to the triangulation. No full-cloud Delaunay and
    no enlarged memory cap. An unresolved query fails explicitly, never with
    a guessed or approximate depth value.
    """
    report = activity or (lambda *args, **kwargs: None)
    report('Dense bathymetry: switching to disk-streamed single-node interpolation', force=True)
    limit = options.max_triangulation_points
    k = min(32, max(3, limit - 3))
    cloud = _nearest_on_disk(idx, target, k)
    support = _hull_support_for_query(idx,target) if limit >= 6 else np.empty((0,3),dtype=idx.dtype)
    seen = {tuple(point[:2]) for point in cloud}
    additional = [point for point in support if tuple(point[:2]) not in seen]
    if additional:
        cloud = np.concatenate((cloud,np.asarray(additional,dtype=cloud.dtype)))
        seen.update(tuple(point[:2]) for point in additional)
    total_steps = 0
    while len(cloud) <= limit:
        total_steps += 1
        if total_steps > limit:
            raise RuntimeError('Disk-streamed bathymetry refinement did not converge')
        try:
            # Query-centered geometry avoids Qhull/circumcenter roundoff
            # when bathymetry spacing is tiny compared with UTM east/north.
            local_xy = np.ascontiguousarray(cloud[:, :2] - target, dtype=np.float64)
            tri = Delaunay(local_xy)
            simplex = int(tri.find_simplex(np.zeros((1, 2), dtype=np.float64))[0])
        except QhullError:
            simplex = -1
            tri = None
        if simplex >= 0:
            circle = _simplex_circle(tri, simplex)
            if circle is not None:
                intruders = _intruders_on_disk(idx, *circle, origin=target)
                if not len(intruders):
                    value = float(_linear_interpolate(
                        tri, cloud, np.zeros((1, 2), dtype=np.float64),
                        np.array([simplex],dtype=np.int64))[0])
                    return value, total_steps
                new = [point for point in intruders if tuple(point[:2]) not in seen]
                if new:
                    cloud = np.concatenate((cloud, np.asarray(new,dtype=cloud.dtype)))
                    seen.update(tuple(point[:2]) for point in new)
                    continue
        # The query lies outside the current subset hull or simplex is nearly
        # degenerate. Expand the exact nearest subset, without a global tree.
        k = min(limit, max(k * 2, len(cloud) + 32))
        neighbors = _nearest_on_disk(idx, target, k)
        new = [point for point in neighbors if tuple(point[:2]) not in seen]
        if not new:
            break
        cloud = np.concatenate((cloud,np.asarray(new,dtype=cloud.dtype)))
        seen.update(tuple(point[:2]) for point in new)
    raise RuntimeError(
        'Could not certify the exact Delaunay interpolation of a dense bathymetry node '
        f'at x={target[0]:.3f}, y={target[1]:.3f} within the {limit:,}-point cap; '
        'no approximate value was substituted.'
    )


def _interpolate_piece(nodes, xy, idx, options, whole, limits, activity=None):
    """Interpolate one disjoint group of mesh-node indices.

    The original sparse-Delaunay + two circumcircle certifications and the
    adaptive halo logic are preserved. Results are returned in `nodes` order;
    workers never mutate arrays owned by the parent process.
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    piece_result = np.full(len(nodes), np.nan, dtype=np.float64)
    x0, y0, x1, y1 = limits
    calls = 0
    triangulations = 0

    def report(message, *, force=False):
        if activity is not None:
            activity(message, force=force)

    def run(positions):
        nonlocal calls, triangulations
        if not len(positions):
            return
        calls += 1
        queries = xy[nodes[positions]]
        lo = queries.min(axis=0)
        hi = queries.max(axis=0)
        domain_width = max(x1-x0, y1-y0, 1.)
        halo = max(float(max(hi-lo))*options.initial_halo_fraction,
                   domain_width/10000., 0.1)
        last_window = None
        pending = np.arange(len(positions))
        while len(pending):
            window = (max(whole[0], lo[0]-halo), max(whole[1], lo[1]-halo),
                      min(whole[2], hi[0]+halo), min(whole[3], hi[1]+halo))
            is_full = window == whole
            if window == last_window:
                raise RuntimeError('Unable to expand local Delaunay window')
            last_window = window
            report(f'Searching bathymetry points for {len(pending):,} unresolved grid nodes')
            try:
                cloud = idx.region(window, options.max_triangulation_points)
            except _TooManyPoints:
                if len(positions) > 1:
                    work = positions[pending]
                    if len(work) == 1:
                        run(work)
                        return
                    pts = xy[nodes[work]]
                    axis = int(np.argmax(np.ptp(pts, axis=0)))
                    order = work[np.argsort(pts[:,axis], kind='stable')]
                    middle = len(order)//2
                    run(order[:middle]); run(order[middle:])
                    return
                value, refinements = _interpolate_dense_single(
                    xy[nodes[positions[0]]], idx, options, activity=activity)
                piece_result[positions[0]] = value
                triangulations += refinements
                return
            if len(cloud) >= 3:
                cloud_xy = np.ascontiguousarray(cloud[:, :2], dtype=np.float64)
                tree = cKDTree(cloud_xy)
                last_k = 0
                for wanted in (16, 32, 64, 128):
                    if not len(pending):
                        break
                    max_k = max(3, options.max_triangulation_points // len(pending))
                    k = min(wanted, len(cloud), max_k)
                    if k <= last_k:
                        continue
                    last_k = k
                    pending_xy = xy[nodes[positions[pending]]]
                    _, near = tree.query(pending_xy, k=k)
                    sparse_ids = np.unique(np.asarray(near).reshape(-1))
                    if len(sparse_ids) > options.max_triangulation_points:
                        del tree
                        work = positions[pending]
                        if len(work) <= 1:
                            raise MemoryError('Sparse point cap exceeded for one node')
                        pts = xy[nodes[work]]
                        axis = int(np.argmax(np.ptp(pts, axis=0)))
                        ordered = work[np.argsort(pts[:,axis], kind='stable')]
                        middle = len(ordered)//2
                        run(ordered[:middle]);run(ordered[middle:])
                        return
                    sparse = cloud[sparse_ids]
                    report(f'triangulating {len(sparse):,} selected points '
                           f'for {len(pending):,} nodes (k={k})',
                           force=(triangulations == 0))
                    triangulations += 1
                    try:
                        tri = Delaunay(np.ascontiguousarray(
                            sparse[:, :2], dtype=np.float64))
                    except QhullError:
                        continue
                    simplex = tri.find_simplex(pending_xy)
                    finite = simplex >= 0
                    certified = np.zeros(len(pending), dtype=bool)
                    if finite.any():
                        candidates = np.flatnonzero(finite)
                        within_window = _circumcircles_certified(
                            tri, simplex[finite], window, whole)
                        candidates = candidates[within_window]
                        if len(candidates):
                            valid = _certify_no_omitted_points(
                                tri, simplex[candidates], tree)
                            certified[candidates[valid]] = True
                    if certified.any():
                        solved_indices = pending[certified]
                        piece_result[positions[solved_indices]] = _linear_interpolate(
                            tri, sparse, pending_xy[certified], simplex[certified])
                        pending = pending[~certified]
                    del tri
                del tree
                if not len(pending):
                    return
            if is_full:
                report(f'Full Delaunay fallback for {len(pending):,} unresolved nodes', force=True)
                triangulations += 1
                tri = Delaunay(np.ascontiguousarray(cloud[:, :2], dtype=np.float64))
                unresolved_xy = xy[nodes[positions[pending]]]
                simplex = tri.find_simplex(unresolved_xy)
                good = simplex >= 0
                if good.any():
                    piece_result[positions[pending[good]]] = _linear_interpolate(
                        tri, cloud, unresolved_xy[good], simplex[good])
                return
            report(f'Expanding halo ({len(pending):,} unresolved nodes; '
                   f'{len(cloud):,} source points)')
            halo *= 2

    run(np.arange(len(nodes), dtype=np.int64))
    return nodes, piece_result, calls, triangulations


def _worker_piece(nodes):
    """Picklable worker function imported by spawned Python interpreters."""
    if _WORKER_STATE is None:
        raise RuntimeError('Bathymetry worker has not been initialized')
    idx, xy, options, whole, limits = _WORKER_STATE
    return _interpolate_piece(nodes, xy, idx, options, whole, limits)


def _partition_nodes(xy, nodes, max_nodes):
    """Produce disjoint spatially contiguous work packets."""
    if len(nodes) <= max_nodes:
        if len(nodes):
            yield nodes
        return
    pts = xy[nodes]
    axis = int(np.argmax(np.ptp(pts, axis=0)))
    order = nodes[np.argsort(pts[:,axis],kind='stable')]
    middle = len(order)//2
    yield from _partition_nodes(xy, order[:middle], max_nodes)
    yield from _partition_nodes(xy, order[middle:], max_nodes)


def _cpu_slots():
    """Respect CPU affinity and a SLURM allocation where available."""
    slots = max(1, os.cpu_count() or 1)
    if hasattr(os, 'sched_getaffinity'):
        try:
            slots = min(slots, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    env = os.environ.get('SLURM_CPUS_PER_TASK')
    if env:
        try:
            slots = min(slots, max(1, int(env)))
        except ValueError:
            pass
    return slots


def _num_workers(options, ntasks):
    count = options.workers
    if count is None:
        env = os.environ.get('FVTOOLS_BATHY_WORKERS')
        count = int(env) if env is not None else 2
    if count < 1:
        raise ValueError('bathymetry workers must be at least 1')
    return min(count, _cpu_slots(), max(1, ntasks))


def interpolate_unstructured_npy(filename, mesh_xy, *, options=None):
    """Memory-limited unstructured linear interpolation; optionally CPU-parallel.

    Worker processes read the same disk bin index and a read-only `.npy` file
    of mesh coordinates. The source .npy is never pickled or sent to workers.
    Use FVTOOLS_BATHY_WORKERS=1 to disable multiprocessing; default is 2.
    """
    options = options or TilingOptions()
    for key in ('source_chunk_rows','max_triangulation_points','nodes_per_tile','bins_per_axis'):
        if getattr(options,key) <= 0:
            raise ValueError(f'{key} must be > 0')
    if options.max_triangulation_points < 3:
        raise ValueError('max_triangulation_points must be at least 3')
    if not (0 < options.initial_halo_fraction <= 1):
        raise ValueError('initial_halo_fraction must be between 0 and 1')
    xy = np.asarray(mesh_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or len(xy) < 1:
        raise ValueError('mesh_xy must be a nonempty, finite (nodes,2) array')
    src = np.load(filename, mmap_mode='r', allow_pickle=False)
    if src.ndim != 2 or src.shape[1] < 3 or src.dtype.kind not in 'fiu':
        raise ValueError('Expected an unstructured numeric .npy array with at least columns [x,y,depth]')
    x0,y0 = xy.min(axis=0)
    x1,y1 = xy.max(axis=0)
    whole = (float(x0-5000.),float(y0-5000.),float(x1+5000.),float(y1+5000.))
    limits = (float(x0),float(y0),float(x1),float(y1))
    result = np.full(len(xy), np.nan, dtype=np.float64)
    with TemporaryDirectory(prefix='fvtools_bathymetry_') as temp:
        idx = _DiskBins(src, whole, temp, options)
        if idx.nonfinite_rows:
            print(f'  - Ignoring {idx.nonfinite_rows} bathymetry rows with non-finite data', flush=True)
        sl,sb,sr,st = idx.global_extent
        if sr < x1 or sl > x0 or st < y1 or sb > y0:
            raise ValueError('The bathymetry file does not cover the model domain!')
        print(f'  - Indexed {idx.total:,} bathymetry points on disk ({idx.count}x{idx.count} bins)', flush=True)
        print(f'  - Interpolating {len(xy):,} grid nodes in memory-limited tiles', flush=True)
        progress = _Progress('Interpolation of grid nodes', len(xy))
        outside = _outside_convex_hull(xy, idx.hull_equations, max(x1-x0,y1-y0))
        finished = int(outside.sum())
        if finished:
            print(f'  - {finished:,} grid nodes outside bathymetry convex hull (remain NaN)',flush=True)
            progress.update(finished)
        batches = list(_partition_nodes(xy, np.flatnonzero(~outside), options.nodes_per_tile))
        workers = _num_workers(options, len(batches))
        print(f'  - {workers} CPU worker(s), {len(batches):,} work batches '
              f'(up to {options.nodes_per_tile:,} nodes/batch)',flush=True)
        if batches:
            print(f'    Batch 1: processing {len(batches[0]):,} grid nodes', flush=True)
        if workers == 1:
            calls = triangulations = 0
            for i, nodes in enumerate(batches,1):
                done, vals, c, t = _interpolate_piece(
                    nodes, xy, idx, options, whole, limits,
                    activity=progress.activity)
                result[done] = vals
                calls += c
                triangulations += t
                finished += len(done)
                progress.update(finished)
        else:
            # Allocate once, share through filesystem. On Windows `spawn` is
            # required; it is also safer than fork from a Jupyter kernel.
            meshfile = Path(temp) / 'mesh_nodes.npy'
            np.save(meshfile, xy)
            index_info = dict(path=str(idx.path), offsets=idx.offsets,
                              bounds=idx.bounds, count=idx.count,
                              record_bytes=idx.record_bytes, dtype=str(idx.dtype),
                              hull_vertices=idx.hull_vertices)
            print('  - Parallel workers triangulating selected points in independent tiles', flush=True)
            ctx = mp.get_context('spawn')
            calls = triangulations = 0
            finished_batches = 0
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=ctx,
                initializer=_init_worker,
                initargs=(index_info, str(meshfile), options, whole, limits),
            ) as pool:
                batch_iter = iter(batches)
                # Bound queued work; if a worker encounters a failure we don't
                # need to wait for hundreds of unnecessary queued batches.
                pending = {}
                for _ in range(min(len(batches),2*workers)):
                    nodes = next(batch_iter)
                    pending[pool.submit(_worker_piece,nodes)] = len(nodes)
                while pending:
                    completed, _ = wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                    if not completed:
                        progress.activity(f'{len(pending):,} batches running/queued '
                                          f'with {workers} CPU workers; '
                                          f'{finished_batches:,}/{len(batches):,} batches complete',
                                          force=True)
                        continue
                    for future in completed:
                        pending.pop(future)
                        done, vals, c, t = future.result()  # propagate worker exceptions
                        result[done] = vals
                        calls += c
                        triangulations += t
                        finished += len(done)
                        finished_batches += 1
                        progress.update(finished)
                        nodes = next(batch_iter,None)
                        if nodes is not None:
                            pending[pool.submit(_worker_piece,nodes)] = len(nodes)
        if finished != len(xy):
            raise RuntimeError('Incomplete tiled interpolation: not all grid nodes were processed')
        print(f'  - Finished: {calls:,} node batches, '
              f'{triangulations:,} local Delaunay triangulations', flush=True)
    return result
