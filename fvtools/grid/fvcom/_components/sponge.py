"""FVCOM sponge damping, matching mod_setup.F:SETUP_SPONGE.

The sponge file contains 1-based node IDs, influence radii in metres, and
maximum damping coefficients. The resulting ``cc_sponge`` lives on *elements*.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from scipy.spatial import cKDTree
from fvtools.setup.initial._helpers.restart_io import atomic_update_restart

_EARTH_RADIUS_M = 6_371_000.0  # FVCOM mod_main.F:REARTH
_HEADER = re.compile(r"^\s*Sponge\s+Node\s+Number\s*=\s*(\d+)\s*$", re.IGNORECASE)


def _read_sponge_file(path, node_count):
    """
    Read BuildCase/FVCOM sponge file (node, radius [m], factor).
    """
    # Read the sponge file
    with Path(path).open(encoding="utf-8-sig") as fh:
        lines = [line.strip() for line in fh if line.strip()]

    # Check that the file is valid
    if not lines or (match := _HEADER.fullmatch(lines[0])) is None:
        raise ValueError(f"Invalid sponge header in {path!s}: expected 'Sponge Node Number = N'")
    
    declared = int(match.group(1))
    data = [line for line in lines[1:] if not line.startswith(("#", "!"))]
    if len(data) != declared:
        raise ValueError(f"{path!s}: header declares {declared} sponge nodes, found {len(data)} rows")

    indices = np.empty(declared, dtype=np.int64)
    radii = np.empty(declared, dtype=np.float64)
    factors = np.empty(declared, dtype=np.float64)
    for i, row in enumerate(data):
        parts = row.split()
        # Check that each row has valid values
        if len(parts) != 3:
            raise ValueError(f"{path!s}: sponge row {i + 1} needs node, radius, factor")

        # Read node, sponge radius for each node and the sponge factor
        try:
            number = int(parts[0])
            radius, factor = (float(x.replace("D", "E").replace("d", "e")) for x in parts[1:])

        # Check that the values in the sponge file are valid (node number, positive and finite values for the sponge radius / factor)
        except ValueError as exc:
            raise ValueError(f"{path!s}: invalid sponge row {i + 1}: {row!r}") from exc
        if not 1 <= number <= node_count:
            raise ValueError(f"{path!s}: sponge node {number} outside grid (1..{node_count})")
        if not np.isfinite(radius) or not np.isfinite(factor):
            raise ValueError(f"{path!s}: non-finite sponge radius/factor on row {i + 1}")
        if factor > 0 and radius <= 0:
            raise ValueError(f"{path!s}: positive sponge factor needs positive radius on row {i + 1}")

        # Prepare the values to be returned if they pass the checks
        indices[i], radii[i], factors[i] = number - 1, radius, factor
    return indices, radii, factors


def _sphere_xyz(lon, lat):
    """
    Longitude/latitude in degrees to unit-sphere Cartesian coordinates.
    """
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    clat = np.cos(lat)
    return np.column_stack((clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)))


class Sponge:
    """Calculate element sponge coefficients and update existing restart files."""
    @property
    def sponge_nodes(self):
        '''
        Sponges from the obc
        - In BuildCase, we assume that sponges are always applied near the OBC, but remember that they can also be put in the domain.
        '''
        if self.sponge_factor > 0:
            sponges = []
            for nodestring in self.nodestrings:
                sponges.extend(nodestring.tolist())
            return np.array(sponges)
        else:
            return np.empty(0)

    def find_sponge_file(self, sponge_file=None):
        """Find ``input/<casename>_spg.dat`` relative to the original grid.

        Return None when no default file exists. An explicitly supplied path
        must exist; it is never silently ignored. The original grid directory
        is saved at construction, so CMEMS temporary working directories work.
        """
        root = Path(getattr(self, "_sponge_source_directory", Path.cwd()))
        if sponge_file is not None:
            path = Path(sponge_file).expanduser()
            if not path.is_absolute() and not path.is_file():
                path = root / path
            if not path.is_file():
                raise FileNotFoundError(f"FVCOM sponge file not found: {sponge_file}")
            return path.resolve()

        name = f"{self.casename}_spg.dat"
        candidates = []
        if self.filepath is not None:
            source = Path(self.filepath)
            if not source.is_absolute():
                source = root / source
            parent = source.parent
            candidates.extend((parent / "input" / name, parent / name,
                               parent.parent / "input" / name))
        candidates.extend((root / "input" / name, root / name))
        info = getattr(self, "info", None)
        if isinstance(info, dict) and info.get("directory"):
            folder = Path(info["directory"])
            candidates.extend((folder / "input" / name, folder / name))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _sponge_is_spherical(self, spherical):
        if spherical is not None:
            return bool(spherical)
        info = getattr(self, "info", None)
        if isinstance(info, dict) and "latlon" in info:
            return bool(info["latlon"])
        # For grid data loaded from a spherical FVCOM restart, x/y are lon/lat.
        # M.npy from BuildCase may retain projected x/y even with SPHERICAL;
        # pass spherical=True explicitly for legacy M.npy without latlon metadata.
        return (np.allclose(self.x, self.lon, atol=1e-5, rtol=0) and
                np.allclose(self.y, self.lat, atol=1e-5, rtol=0))

    def calculate_cc_sponge(self, sponge_file=None, *, spherical=None):
        """
        Return float32 ``cc_sponge[nele]`` as FVCOM's SETUP_SPONGE does.

        ``spherical=True`` uses great-circle distances (FVCOM SPHERICAL);
        otherwise metre-based projected x/y distances are used. For BuildCase
        M.npy files the spherical flag is read from ``info['latlon']``.
        An absent default sponge file or a header with zero nodes means zero
        damping, as does a sponge factor <= 0.
        """
        ncell = len(self.xc)
        result = np.zeros(ncell, dtype=np.float32)
        path = self.find_sponge_file(sponge_file)
        if path is None:
            return result
        indices, radii, factors = _read_sponge_file(path, len(self.x))
        active = (factors > 0) & (radii > 0)
        if not np.any(active):
            return result
        indices, radii, factors = indices[active], radii[active], factors[active]
        sphere = self._sponge_is_spherical(spherical)
        if sphere:
            source = _sphere_xyz(np.asarray(self.lon)[indices], np.asarray(self.lat)[indices])
            target = _sphere_xyz(np.asarray(self.lonc), np.asarray(self.latc))
        else:
            source = np.column_stack((np.asarray(self.x)[indices], np.asarray(self.y)[indices]))
            target = np.column_stack((np.asarray(self.xc), np.asarray(self.yc)))
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("Sponge calculation requires finite node and cell coordinates")

        tree = cKDTree(target)
        # Float64 arithmetic, then FVCOM's single-precision element field.
        coeff = np.zeros(ncell, dtype=np.float64)
        for pos, radius, factor in zip(source, radii, factors):
            query_radius = (2.0 * np.sin(min(radius / _EARTH_RADIUS_M, np.pi) / 2.0)
                            + 1e-12 if sphere else radius)
            cells = tree.query_ball_point(pos, query_radius)
            if not cells:
                continue
            cell_ids = np.asarray(cells, dtype=np.intp)
            difference = target[cell_ids] - pos
            sqdist = np.einsum("ij,ij->i", difference, difference)
            if sphere:
                # Same great-circle expression as FVCOM mod_spherical.F:ARC.
                distances = _EARTH_RADIUS_M * np.arccos(np.clip(1.0 - 0.5 * sqdist, -1.0, 1.0))
            else:
                distances = np.sqrt(sqdist)
            weights = factor * np.maximum(1.0 - distances / radius, 0.0)
            coeff[cell_ids] = np.maximum(coeff[cell_ids], weights)
        return coeff.astype(np.float32)

    def update_restart_sponge(self, restart_file, sponge_file=None, *, spherical=None):
        """Atomically replace only ``cc_sponge`` in an existing FVCOM restart.

        The model grid and restart must have matching element counts and
        geographical element coordinates. All other fields are untouched.
        """
        values = self.calculate_cc_sponge(sponge_file, spherical=spherical)
        with atomic_update_restart(restart_file) as nc:
            if "cc_sponge" not in nc.variables:
                raise ValueError("Restart has no cc_sponge variable")
            var = nc["cc_sponge"]
            if var.dimensions != ("nele",) or var.shape != values.shape:
                raise ValueError("Restart cc_sponge dimension does not match this FVCOM grid")
            for name, expected in (("lonc", self.lonc), ("latc", self.latc)):
                if name in nc.variables:
                    actual = np.ma.filled(np.ma.asarray(nc[name][:]), np.nan)
                    if not np.allclose(actual, expected, atol=1e-4, rtol=0):
                        raise ValueError(f"Restart {name} differs from the FVCOM grid")
            var[:] = values
