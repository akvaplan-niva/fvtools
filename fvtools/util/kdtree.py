"""Nearest-neighbour KD-tree backend for fvtools.

Prefer pykdtree when available (typically Python <=3.14 or conda-forge);
use SciPy's cKDTree when pykdtree has no compatible binary wheel.
Both expose KDTree(points).query(query_points, k=...).
"""

try:
    from pykdtree.kdtree import KDTree
    BACKEND = "pykdtree"
except ImportError:
    from scipy.spatial import cKDTree as KDTree
    BACKEND = "scipy"
