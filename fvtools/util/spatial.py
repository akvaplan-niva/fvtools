# fvtools/util/spatial.py
from fvtools.grid.fvcom_grd import FVCOM_grid

import numpy as np
try:
    from pyproj import Transformer, CRS
except Exception:
    Transformer = None
    CRS = None

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None  # optional accel

__all__ = [
    "load_stations_txt",
    "nearest_node_ids", "nodeids_for_section",
    "nearest_cell_ids", "cellids_for_section",
    "load_nodes_xy", "load_cells_xy",
]

# ---------- internals

def _unwrap_maybe_object_array(M):
    if isinstance(M, np.ndarray) and M.dtype == object and M.shape == ():
        return M.item()
    return M

def _extract_xy_any(M, field_x="x", field_y="y"):
    if hasattr(M, field_x) and hasattr(M, field_y):
        return np.asarray(getattr(M, field_x)).ravel(), np.asarray(getattr(M, field_y)).ravel()
    if isinstance(M, dict) and field_x in M and field_y in M:
        return np.asarray(M[field_x]).ravel(), np.asarray(M[field_y]).ravel()
    A = np.asarray(M)
    if A.dtype.names is not None and field_x in A.dtype.names and field_y in A.dtype.names:
        return np.asarray(A[field_x]).ravel(), np.asarray(A[field_y]).ravel()
    if A.ndim == 2 and A.shape[1] == 2:
        return A[:, 0].ravel(), A[:, 1].ravel()
    if A.ndim == 2 and A.shape[0] == 2:
        return A[0, :].ravel(), A[1, :].ravel()
    avail = list(getattr(M, "__dict__", {}).keys()) if hasattr(M, "__dict__") else (list(M.keys()) if isinstance(M, dict) else [])
    raise ValueError(f"M does not expose '{field_x},{field_y}'. Available fields/attrs: {avail}")

#def _load_M(m_path_or_obj):
#    return _unwrap_maybe_object_array(
#        np.load(m_path_or_obj, allow_pickle=True) if isinstance(m_path_or_obj, str) else m_path_or_obj
#    )

def _load_M(m_path_or_obj):
    return FVCOM_grid(m_path_or_obj)



def _project_lonlat(lon, lat, crs_to):
    """
    Input lon/lat are WGS84 (EPSG:4326). If crs_to is:
      - None or 'lonlat' -> return (lon in [-180,180), lat)
      - 'EPSG:xxxx' or any pyproj CRS string -> project to x/y using pyproj
    If pyproj is unavailable and projection requested, raises a clear error.
    """
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)

    if crs_to is None or str(crs_to).lower() in ("lonlat", "epsg:4326", "wgs84"):
        return _to_lon180(lon), lat

    if Transformer is None:
        raise RuntimeError(
            f"pyproj not available, cannot project to {crs_to}. "
            "Install pyproj or use reference='lonlat'."
        )

    T = Transformer.from_crs("EPSG:4326", CRS.from_user_input(crs_to), always_xy=True)
    x, y = T.transform(lon, lat)
    return np.asarray(x), np.asarray(y)

# ---------- public API (nodes)

def load_nodes_xy(m_path_or_obj):
    M = _load_M(m_path_or_obj)
    return _extract_xy_any(M, "x", "y")

def load_stations_txt(txt_path, reference=None):
    names, lons, lats = [], [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            name = parts[0]
            vals = []
            for t in parts[1:]:
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
            if len(vals) < 2:
                raise ValueError(f"Need lon lat on line: {ln}")
            names.append(name); lons.append(vals[0]); lats.append(vals[1])
    lons = np.array(lons, float)
    lats = np.array(lats, float)

    xs, ys = _project_lonlat(lons, lats, crs_to=reference)
    return np.array(names, dtype=object), np.column_stack([xs, ys])
    

def nearest_node_ids(stations_xy, x_nodes, y_nodes):
    nodes = np.column_stack([np.asarray(x_nodes, float), np.asarray(y_nodes, float)])
    P = np.asarray(stations_xy, float)
    if KDTree is not None:
        tree = KDTree(nodes); _, idx = tree.query(P)
    else:
        dif = nodes[None, :, :] - P[:, None, :]
        idx = np.argmin((dif**2).sum(axis=2), axis=1)
    idx = idx.astype(int)
    return idx 

def nodeids_for_section(m_npy_path, stations_txt_path, reference,  return_names=False,depth_field="siglayz"):
    M = _load_M(m_npy_path)
    x_nodes, y_nodes = load_nodes_xy(m_npy_path)
    names, stations_xy = load_stations_txt(stations_txt_path, M.reference)
    ids = nearest_node_ids(stations_xy, x_nodes, y_nodes)
    # --------------depth    
    sig = getattr(M, depth_field, None) if not isinstance(M, dict) else M.get(depth_field, None)
    if sig is None:
        raise AttributeError(f"M has no field '{depth_field}'")
    sig = np.asarray(sig)
    Nnodes = x_nodes.size
    if sig.ndim != 2:
        raise ValueError(f"{depth_field} must be 2D, got shape {sig.shape}")
    if   sig.shape[0] == Nnodes: sig_nodes_first = sig
    elif sig.shape[1] == Nnodes: sig_nodes_first = sig.T
    else: raise ValueError(f"{depth_field} shape {sig.shape} incompatible with Nnodes={Nnodes}")

    depths = sig_nodes_first[ids,:] 
    out = {
        "node_id": ids.astype(int),
        "node_depth": depths.astype(float),
    }
    if return_names:
        out["station_name"] = names
        out["station_xy"] = stations_xy
    return out               # (space, nz)


# ---------- public API (cells)

def load_cells_xy(m_path_or_obj):
    M = _load_M(m_path_or_obj)
    return _extract_xy_any(M, "xc", "yc")

def nearest_cell_ids(stations_xy, x_cells, y_cells):
    cells = np.column_stack([np.asarray(x_cells, float), np.asarray(y_cells, float)])
    P = np.asarray(stations_xy, float)
    if KDTree is not None:
        tree = KDTree(cells); _, idx = tree.query(P)
    else:
        dif = cells[None, :, :] - P[:, None, :]
        idx = np.argmin((dif**2).sum(axis=2), axis=1)
    idx = idx.astype(int)
    return idx

def cellids_for_section(m_npy_path, stations_txt_path, return_names=False,depth_field="siglayz_uv"):
    M = _load_M(m_npy_path)
    xc, yc = load_cells_xy(m_npy_path)
    names, stations_xy = load_stations_txt(stations_txt_path, M.reference)
    ids = nearest_cell_ids(stations_xy, xc, yc)

    # --------------depth    
    sig = getattr(M, depth_field, None) if not isinstance(M, dict) else M.get(depth_field, None)
    if sig is None:
        raise AttributeError(f"M has no field '{depth_field}'")
    sig = np.asarray(sig)
    Ncell = xc.size
    if sig.ndim != 2:
        raise ValueError(f"{depth_field} must be 2D, got shape {sig.shape}")
    if   sig.shape[0] == Ncell: sig_cells_first = sig
    elif sig.shape[1] == Nnodes: sig_cells_first = sig.T
    else: raise ValueError(f"{depth_field} shape {sig.shape} incompatible with Nnodes={Nnodes}")

    depths = sig_cells_first[ids,:]    
    out = {
        "cell_id": ids.astype(int),
        "cell_depth": depths.astype(float),
    }
    if return_names:
        out["station_name"] = names
        out["station_xy"] = stations_xy
    return out               # (space, nz)
        









