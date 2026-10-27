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

import numpy as np
from scipy.spatial import Delaunay, QhullError


@dataclass(frozen=True)
class TilingOptions:
    """Limits in *points*, rather than relying on potentially unsafe RAM guesses."""

    source_chunk_rows: int = 250_000
    max_triangulation_points: int = 350_000
    nodes_per_tile: int = 128
    bins_per_axis: int = 64
    initial_halo_fraction: float = 0.15


class _TooManyPoints(Exception):
    pass


def _bins_for_xy(x, y, bounds, count):
    xmin, ymin, xmax, ymax = bounds
    dx, dy = max(xmax - xmin, np.finfo(float).eps), max(ymax - ymin, np.finfo(float).eps)
    ix = np.clip(((x - xmin) / dx * count).astype(np.int64), 0, count - 1)
    iy = np.clip(((y - ymin) / dy * count).astype(np.int64), 0, count - 1)
    return ix * count + iy


class _DiskBins:
    """Two-pass on-disk spatial bin index of cropped XYZ points."""

    def __init__(self, data, bounds, directory, options):
        self.data = data
        self.bounds = bounds
        self.count = options.bins_per_axis
        self.chunk_rows = options.source_chunk_rows
        self.nbin = self.count ** 2
        self.counts = np.zeros(self.nbin, dtype=np.int64)
        xmin, ymin, xmax, ymax = bounds
        self.global_extent = [np.inf, np.inf, -np.inf, -np.inf]
        self.nonfinite_rows = 0

        for start in range(0, len(data), self.chunk_rows):
            stop = min(start + self.chunk_rows, len(data))
            xyz = np.asarray(data[start:stop, :3])
            valid = np.isfinite(xyz).all(axis=1)
            self.nonfinite_rows += int((~valid).sum())
            if not valid.any():
                continue
            points = xyz[valid]
            self.global_extent[0] = min(self.global_extent[0], float(points[:, 0].min()))
            self.global_extent[1] = min(self.global_extent[1], float(points[:, 1].min()))
            self.global_extent[2] = max(self.global_extent[2], float(points[:, 0].max()))
            self.global_extent[3] = max(self.global_extent[3], float(points[:, 1].max()))
            inside = (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
            points = points[inside]
            if len(points):
                ids = _bins_for_xy(points[:, 0], points[:, 1], bounds, self.count)
                self.counts += np.bincount(ids, minlength=self.nbin)

        total = int(self.counts.sum())
        if total < 3:
            raise ValueError('Fewer than 3 finite bathymetry points are inside the mesh crop area')
        self.offsets = np.r_[0, np.cumsum(self.counts)]
        self.path = Path(directory) / 'bathymetry_spatial_index.bin'
        self.dtype = data.dtype
        self.record_bytes = 3 * data.dtype.itemsize
        with self.path.open('wb') as f:
            f.truncate(total * self.record_bytes)
        cursor = self.offsets[:-1].copy()
        with self.path.open('r+b') as dest:
          for start in range(0, len(data), self.chunk_rows):
            stop = min(start + self.chunk_rows, len(data))
            xyz = np.asarray(data[start:stop, :3])
            valid = np.isfinite(xyz).all(axis=1)
            inside = valid & (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
            if not inside.any():
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
        print(f'  - Indexed {idx.total:,} bathymetry points on disk ({idx.count}x{idx.count} bins)')
        print('  - Interpolating bathymetry in memory-limited tiles')
        calls = 0
        def run(nodes):
            nonlocal calls
            calls += 1
            queries = xy[nodes]
            lo = queries.min(axis=0)
            hi = queries.max(axis=0)
            domain_width = max(x1-x0,y1-y0,1.)
            halo = max(float(max(hi-lo))*options.initial_halo_fraction, domain_width/10000., 0.1)
            last_window = None
            while True:
                window = (max(whole[0],lo[0]-halo),max(whole[1],lo[1]-halo),
                          min(whole[2],hi[0]+halo),min(whole[3],hi[1]+halo))
                is_full = window==whole
                if window==last_window:
                    raise RuntimeError('Unable to expand local Delaunay window')
                last_window=window
                try:
                    cloud = idx.region(window,options.max_triangulation_points)
                except _TooManyPoints:
                    if len(nodes)>1:
                        # Reduce target area rather than exceeding memory cap.
                        axis = int(np.argmax(hi-lo))
                        order = nodes[np.argsort(xy[nodes,axis],kind='stable')]
                        middle=len(order)//2
                        run(order[:middle]);run(order[middle:])
                        return
                    raise MemoryError(
                        'Even a single bathymetry query requires more than '
                        f'{options.max_triangulation_points:,} local points. '
                        'Increase max_triangulation_points or check point density.'
                    ) from None
                if len(cloud)>=3:
                    try:
                        tri = Delaunay(np.ascontiguousarray(cloud[:,:2],dtype=np.float64))
                    except QhullError:
                        if is_full:
                            raise ValueError('Bathymetry coordinates are degenerate; cannot triangulate') from None
                    else:
                        simplex = tri.find_simplex(queries)
                        finite = simplex>=0
                        certified = np.zeros(len(nodes),dtype=bool)
                        if finite.any():
                            certified[finite]=_circumcircles_certified(tri,simplex[finite],window,whole)
                        if certified.all() or is_full:
                            if finite.any():
                                result[nodes[finite]]=_linear_interpolate(tri,cloud,queries[finite],simplex[finite])
                            return
                if is_full:
                    return
                halo *= 2
        # Divide mesh nodes into spatially compact groups, without building a tree.
        def partition(nodes):
            if len(nodes)<=options.nodes_per_tile:
                run(nodes)
                return
            pts=xy[nodes]
            axis=int(np.argmax(np.ptp(pts,axis=0)))
            order=nodes[np.argsort(pts[:,axis],kind='stable')]
            split=len(order)//2
            partition(order[:split]);partition(order[split:])
        partition(np.arange(len(xy)))
        print(f'  - Completed {calls} local triangulation tiles')
    return result
