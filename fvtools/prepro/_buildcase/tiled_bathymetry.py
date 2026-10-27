"""Memory-limited linear interpolation from large unstructured (N,3) .npy data.

A disk-backed uniform-bin index is constructed in two streaming passes. Local
Delaunay triangulations are expanded until each queried simplex is provably
unaffected by bathymetry points outside the local window (circumcircle test).
No globally sized point tree or Delaunay triangulation is created in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def interpolate_unstructured_npy(filename, mesh_xy, *, options=None):
    """Interpolate (N,3) unstructured .npy to mesh nodes with bounded RAM.

    Returns NaN for queries outside the global data convex hull, matching
    scipy.interpolate.LinearNDInterpolator. The caller handles the minimum-depth
    replacement as in existing BuildCase.make_depth().
    """
    options = options or TilingOptions()
    for key in ('source_chunk_rows','max_triangulation_points','nodes_per_tile','bins_per_axis'):
        if getattr(options,key) <= 0:
            raise ValueError(f'{key} must be > 0')
    if options.max_triangulation_points < 3:
        raise ValueError('max_triangulation_points must be at least 3')
    if not (0 < options.initial_halo_fraction <= 1):
        raise ValueError('initial_halo_fraction must be between 0 and 1')
    xy = np.asarray(mesh_xy,dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or len(xy) < 1:
        raise ValueError('mesh_xy must be a nonempty, finite (nodes,2) array')
    src = np.load(filename, mmap_mode='r', allow_pickle=False)
    if src.ndim != 2 or src.shape[1] < 3 or src.dtype.kind not in 'fiu':
        raise ValueError('Expected an unstructured numeric .npy array with at least columns [x,y,depth]')
    x0,y0 = xy.min(axis=0)
    x1,y1 = xy.max(axis=0)
    # Matches existing DepthHandler.crop(): points within mesh bbox +/- 5000 m.
    whole = (float(x0-5000.), float(y0-5000.), float(x1+5000.),float(y1+5000.))
    result = np.full(len(xy),np.nan,dtype=np.float64)
    with TemporaryDirectory(prefix='fvtools_bathymetry_') as temp:
        idx = _DiskBins(src, whole, temp, options)
        if idx.nonfinite_rows:
            print(f'  - Ignoring {idx.nonfinite_rows} bathymetry rows with non-finite data')
        # Preserve the old crop() coverage check before expensive triangulations.
        sl,sb,sr,st = idx.global_extent
        if sr<x1 or sl>x0 or st<y1 or sb>y0:
            raise ValueError('The bathymetry file does not cover the model domain!')
        print(f'  - Indexed {idx.total:,} bathymetry points on disk ({idx.count}x{idx.count} bins)', flush=True)
        print(f'  - Interpolating {len(xy):,} grid nodes in memory-limited tiles', flush=True)
        progress = _Progress('Interpolation of grid nodes', len(xy))
        calls = 0
        triangulations = 0
        finished = 0
        # Nodes outside the global convex hull must be NaN, independently of
        # tile layout. Classifying them first avoids pathological halo expansion.
        outside = _outside_convex_hull(
            xy, idx.hull_equations, max(x1-x0, y1-y0))
        if np.any(outside):
            finished = int(outside.sum())
            print(f'  - {finished:,} grid nodes outside bathymetry convex hull '
                  '(will remain NaN)', flush=True)
            progress.update(finished)

        def run(nodes):
            """One or more adaptive local Delaunay triangulations."""
            nonlocal calls, triangulations, finished
            if not len(nodes):
                return
            calls += 1
            queries = xy[nodes]
            lo = queries.min(axis=0)
            hi = queries.max(axis=0)
            domain_width = max(x1-x0, y1-y0, 1.)
            halo = max(float(max(hi-lo))*options.initial_halo_fraction,
                       domain_width/10000., 0.1)
            last_window = None
            pending = np.arange(len(nodes))
            while len(pending):
                window = (max(whole[0],lo[0]-halo),max(whole[1],lo[1]-halo),
                          min(whole[2],hi[0]+halo),min(whole[3],hi[1]+halo))
                is_full = window==whole
                if window==last_window:
                    raise RuntimeError('Unable to expand local Delaunay window')
                last_window=window
                progress.activity(f'Working on batch {calls:,}: searching bathymetry points '
                                  f'for {len(pending):,} unresolved grid nodes')
                try:
                    cloud = idx.region(window,options.max_triangulation_points)
                except _TooManyPoints:
                    if len(nodes)>1:
                        # Restart with a smaller bounding box, even if there
                        # is only one outstanding node in the current group.
                        work = nodes[pending]
                        if len(work) == 1:
                            run(work)
                            return
                        pts = xy[work]
                        axis = int(np.argmax(np.ptp(pts, axis=0)))
                        order = work[np.argsort(pts[:, axis],kind='stable')]
                        middle=len(order)//2
                        run(order[:middle]); run(order[middle:])
                        return
                    raise MemoryError(
                        'Even a single bathymetry query requires more than '
                        f'{options.max_triangulation_points:,} local points. '
                        'Increase max_triangulation_points or check point density.'
                    ) from None
                if len(cloud)>=3:
                    cloud_xy = np.ascontiguousarray(cloud[:, :2], dtype=np.float64)
                    tree = cKDTree(cloud_xy)
                    # Start with 16 nearest source points per query. Increase
                    # to 32/64/128 ONLY for unresolved mesh nodes. A circle
                    # emptiness test against the FULL source tree guarantees
                    # that the selected simplex is globally Delaunay-valid.
                    last_k = 0
                    for wanted in (16, 32, 64, 128):
                        if not len(pending):
                            break
                        max_k = max(3, options.max_triangulation_points // len(pending))
                        k = min(wanted, len(cloud), max_k)
                        if k <= last_k:
                            continue
                        last_k = k
                        pending_xy = xy[nodes[pending]]
                        _, near = tree.query(pending_xy, k=k)
                        sparse_ids = np.unique(np.asarray(near).reshape(-1))
                        if len(sparse_ids) > options.max_triangulation_points:
                            # A custom nodes_per_tile may be unusually large.
                            # Never violate the configured Delaunay RAM cap.
                            del tree
                            work = nodes[pending]
                            if len(work) <= 1:
                                raise MemoryError('Sparse point cap exceeded for one node')
                            pts = xy[work]
                            axis = int(np.argmax(np.ptp(pts, axis=0)))
                            ordered = work[np.argsort(pts[:, axis], kind='stable')]
                            middle = len(ordered)//2
                            run(ordered[:middle]);run(ordered[middle:])
                            return
                        sparse = cloud[sparse_ids]
                        progress.activity(
                            f'Batch {calls:,}: triangulating {len(sparse):,} selected '
                            f'points (from {len(cloud):,}) for {len(pending):,} '
                            f'grid nodes (k={k})', force=(triangulations == 0))
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
                            result[nodes[solved_indices]] = _linear_interpolate(
                                tri, sparse, pending_xy[certified], simplex[certified])
                            finished += len(solved_indices)
                            progress.update(finished)
                            pending = pending[~certified]
                        del tri
                    del tree
                    if not len(pending):
                        return
                if is_full:
                    # Fall back to the full local cloud only when sparse
                    # certification failed. This path also handles points
                    # genuinely outside the hull if there was no precomputed
                    # hull (degenerate source data).
                    progress.activity(f'Batch {calls:,}: full Delaunay fallback '
                                      f'for {len(pending):,} unresolved nodes', force=True)
                    triangulations += 1
                    tri = Delaunay(np.ascontiguousarray(cloud[:, :2], dtype=np.float64))
                    unresolved_xy = xy[nodes[pending]]
                    simplex = tri.find_simplex(unresolved_xy)
                    good = simplex >= 0
                    if good.any():
                        result[nodes[pending[good]]] = _linear_interpolate(
                            tri, cloud, unresolved_xy[good], simplex[good])
                    finished += len(pending)
                    progress.update(finished)
                    return
                progress.activity(f'Batch {calls:,}: expanding search halo '
                                  f'({len(pending):,} unresolved nodes; '
                                  f'{len(cloud):,} source points)')
                halo *= 2

        def partition(nodes):
            # Start with a much larger spatial group (4096 instead of 128
            # nodes). The source point cap, not this node count, controls RAM.
            # Subdivide only when a local triangulation would exceed the cap.
            if len(nodes)<=options.nodes_per_tile:
                run(nodes)
                return
            pts=xy[nodes]
            axis=int(np.argmax(np.ptp(pts,axis=0)))
            order=nodes[np.argsort(pts[:,axis],kind='stable')]
            split=len(order)//2
            partition(order[:split]);partition(order[split:])
        partition(np.flatnonzero(~outside))
        if finished != len(xy):
            raise RuntimeError('Incomplete tiled interpolation: not all grid nodes were processed')
        print(f'  - Finished: {calls:,} node batches, '
              f'{triangulations:,} local Delaunay triangulations', flush=True)
    return result
