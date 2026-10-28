"""Memory-budgeted global kNN index with *local*, certified Delaunay interpolation.

The source cloud remains in a disk memmap. Only the KDTree (not a global
Delaunay triangulation) is built in memory. Every accepted local triangle
has its circumcircle checked against the entire source KDTree.
"""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
from scipy.spatial import cKDTree, Delaunay, QhullError


# Measured allocations depend on platform and tree balance. Reserve room for
# tree nodes, indices and the disk-mapped coordinate table.
_ESTIMATED_BYTES_PER_SOURCE = 120


class FastNearestIndex:
    def __init__(self, disk, directory, *, budget_mb):
        self.count = int(disk.total)
        estimate = self.count * _ESTIMATED_BYTES_PER_SOURCE
        if estimate > budget_mb * 1024**2:
            raise ValueError(
                f'Fast kNN index estimates {estimate / 1024**3:.2f} GiB; '
                f'budget is {budget_mb / 1024:.2f} GiB')
        self.origin = np.array([disk.bounds[0], disk.bounds[1]], dtype=np.float64)
        print(f'  - Fast kNN: preparing {self.count:,} source coordinates', flush=True)
        source = np.memmap(disk.path, mode='r', dtype=disk.dtype, shape=(self.count, 3))
        self.src = source
        self.coord_file = Path(directory) / 'fast_coordinates.npy'
        coords = np.lib.format.open_memmap(
            self.coord_file, mode='w+', dtype=np.float64, shape=(self.count, 2))
        for start in range(0, self.count, 250_000):
            end = min(self.count, start+250_000)
            coords[start:end] = source[start:end, :2].astype(np.float64) - self.origin
        coords.flush()
        # compact_nodes=False avoids expensive median split passes on huge clouds.
        print('  - Fast kNN: building global nearest-neighbour tree', flush=True)
        self.tree = cKDTree(coords, leafsize=64, balanced_tree=False,
                            compact_nodes=False, copy_data=False)
        self.coords = coords
        hull = np.asarray(disk.hull_vertices, dtype=float)
        if len(hull):
            _, hull_idx = self.tree.query(hull - self.origin, k=1)
            self.hull_indices = np.unique(np.asarray(hull_idx, dtype=np.int64))
        else:
            self.hull_indices = np.empty(0, dtype=np.int64)


    def close(self):
        """Release memory maps before TemporaryDirectory cleanup on Windows."""
        self.tree = None
        for attr in ('coords','src'):
            data = getattr(self,attr,None)
            if data is not None:
                mapped = getattr(data,'_mmap',None)
                if mapped is not None:
                    mapped.close()
                setattr(self,attr,None)

def _solve_single_with_tree(target, index, point_cap):
    """Exact simplex refinement by adding global circumcircle intruders.

    Unlike the old disk fallback this uses O(log N) KDTree queries and never
    scans millions of source records for every problematic grid node.
    """
    from fvtools.prepro._buildcase.tiled_bathymetry import (
        _simplex_circle, _linear_interpolate)
    query = np.asarray(target, dtype=np.float64) - index.origin
    k = min(32, index.count)
    initial = np.atleast_1d(index.tree.query(query, k=k)[1]).ravel()
    selected = set(map(int, initial))
    if len(index.hull_indices) < max(10, point_cap // 10):
        selected.update(map(int, index.hull_indices))
    calls = 0
    while len(selected) <= point_cap and calls < 256:
        calls += 1
        indices = np.fromiter(selected, dtype=np.int64)
        # Query-centric Delaunay geometry is crucial for dense, sub-meter
        # bathymetry near seven-figure UTM northings.
        local = np.ascontiguousarray(index.coords[indices] - query)
        try:
            tri = Delaunay(local)
            simplex = int(tri.find_simplex(np.zeros((1,2),dtype=float))[0])
        except QhullError:
            simplex = -1
        if simplex >= 0:
            circle = _simplex_circle(tri, simplex)
            if circle is not None:
                center, radius = circle
                dist, intruder = index.tree.query(query + center, k=1)
                # Nearest point on a valid Delaunay circumcircle is one of the
                # triangle vertices, up to floating-point roundoff.
                precision = 16 * np.finfo(float).eps * max(1.,np.max(np.abs(query + center)))
                tol = 5e-12 * max(radius, 1.) + precision
                if dist >= radius - tol:
                    value = float(_linear_interpolate(tri,index.src[indices],
                        np.zeros((1,2)), np.array([simplex]))[0])
                    return value, calls
                if int(intruder) not in selected:
                    selected.add(int(intruder))
                    continue
        # Expand the exact nearest neighborhood only when we cannot get a
        # containing simplex or the next circumcircle point is already known.
        k = min(index.count, point_cap, max(2*k, len(selected) + 8))
        neighbors = np.atleast_1d(index.tree.query(query,k=k)[1]).ravel()
        before = len(selected)
        selected.update(map(int, neighbors))
        if len(selected) == before:
            break
    return None, calls


def _query_piece(nodes, xy, index, point_cap, activity=None):
    """Interpolate groups with verified local Delaunay and global kNN tests."""
    from fvtools.prepro._buildcase.tiled_bathymetry import (
        _certify_no_omitted_points, _linear_interpolate, _interpolate_dense_single)
    nodes = np.asarray(nodes,dtype=np.int64)
    result = np.full(len(nodes),np.nan,dtype=np.float64)
    queries = xy[nodes] - index.origin
    pending = np.arange(len(nodes),dtype=np.int64)
    triangulations = 0
    fallbacks = 0

    def run(pending, ks=(16,32,64,128)):
        nonlocal triangulations,fallbacks
        if not len(pending):
            return
        for k in ks:
            if activity is not None:
                activity(f'Fast kNN: {len(pending):,} unsolved nodes; k={k}')
            if not len(pending):
                return
            k = min(k,index.count)
            nearest = index.tree.query(queries[pending],k=k,workers=1)[1]
            selected = np.unique(np.asarray(nearest,dtype=np.int64).reshape(-1))
            if len(index.hull_indices) < point_cap // 10:
                selected = np.union1d(selected,index.hull_indices)
            if len(selected) > point_cap:
                if len(pending)>1:
                    axis = int(np.argmax(np.ptp(queries[pending],axis=0)))
                    ordered = pending[np.argsort(queries[pending,axis],kind='stable')]
                    half=len(ordered)//2
                    run(ordered[:half], ks)
                    run(ordered[half:], ks)
                    return
                break
            # Full precision in a local coordinate frame. Only a small point
            # subset is ever Delaunay-triangulated.
            origin = queries[pending[0]]
            cloud = np.ascontiguousarray(index.coords[selected]-origin)
            try:
                tri=Delaunay(cloud)
            except QhullError:
                continue
            triangulations+=1
            simplex=tri.find_simplex(queries[pending]-origin)
            valid_idx=np.flatnonzero(simplex>=0)
            accepted=np.zeros(len(pending),dtype=bool)
            if len(valid_idx):
                # Local triangle empty with respect to EVERY global source point.
                # Shift the circumcenters to the index's coordinate frame by
                # giving a translated read-only KDTree adapter below.
                good = _certify_global_local(tri,simplex[valid_idx],index.tree,origin)
                accepted[valid_idx[good]]=True
            if accepted.any():
                solved=pending[accepted]
                result[solved]=_linear_interpolate(
                    tri,index.src[selected],queries[solved]-origin,simplex[accepted])
                pending=pending[~accepted]
            del tri
        for j in pending:
            if activity is not None:
                activity(f'Fast kNN: refining difficult grid node {int(nodes[j]):,}')
            value, attempts=_solve_single_with_tree(xy[nodes[j]],index,point_cap)
            triangulations+=attempts
            if value is None:
                fallbacks+=1
                value, attempts=_interpolate_dense_single(xy[nodes[j]],index.disk,index.options)
                triangulations+=attempts
            result[j]=value

    run(pending)
    return nodes,result,triangulations,fallbacks


def _certify_global_local(tri, simplex_ids, global_tree, origin):
    """Certify all unique simplices with ONE batched global KDTree query."""
    unique, inverse = np.unique(np.asarray(simplex_ids, dtype=np.int64),
                                return_inverse=True)
    points = tri.points[tri.simplices[unique]]
    a, b = points[:,1] - points[:,0], points[:,2] - points[:,0]
    aa = np.einsum('ij,ij->i',a,a) * .5
    bb = np.einsum('ij,ij->i',b,b) * .5
    det = a[:,0]*b[:,1]-a[:,1]*b[:,0]
    safe = np.abs(det)>64*np.finfo(float).eps*np.maximum(2*(aa+bb),1.)
    centers = np.empty((len(unique),2),dtype=float)
    centers[:]=np.nan
    centers[safe,0] = points[safe,0,0]+(
        aa[safe]*b[safe,1]-a[safe,1]*bb[safe])/det[safe]
    centers[safe,1] = points[safe,0,1]+(
        a[safe,0]*bb[safe]-aa[safe]*b[safe,0])/det[safe]
    radii = np.linalg.norm(centers - points[:,0],axis=1)
    good = safe & np.isfinite(radii)
    passed=np.zeros(len(unique),dtype=bool)
    if good.any():
        absolute = centers[good] + origin
        distance, _ = global_tree.query(absolute,k=1,workers=1)
        precision = 16*np.finfo(float).eps*np.maximum(
            np.max(np.abs(absolute),axis=1),1.)
        tol=5e-12*np.maximum(radii[good],1.)+precision
        passed[good] = distance >= radii[good] - tol
    return passed[inverse]
