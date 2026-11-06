"""FVCOM restart field names and fixed node/element dimensions."""
from __future__ import annotations

RESTART_DIMS = {
    'zeta': ('time', 'node'),
    'temp': ('time', 'siglay', 'node'),
    'salinity': ('time', 'siglay', 'node'),
    'u': ('time', 'siglay', 'nele'),
    'v': ('time', 'siglay', 'nele'),
    'ua': ('time', 'nele'),
    'va': ('time', 'nele'),
}
HYDROGRAPHY = ('zeta', 'temp', 'salinity')
VELOCITY = ('u', 'v', 'ua', 'va')
ROMS_TO_FVCOM = {'salt': 'salinity'}


def required_fields(uv=False):
    """Return variables to initialize; velocities are always element-centered."""
    return HYDROGRAPHY + VELOCITY if uv else HYDROGRAPHY


def validate_restart_fields(nc, names):
    """Check presence and dimension order; raises before any file is modified."""
    for name in names:
        dims = RESTART_DIMS[name]
        if name not in nc.variables or nc[name].dimensions != dims:
            raise ValueError(f'FVCOM restart missing {name} or dimensions {dims}')
