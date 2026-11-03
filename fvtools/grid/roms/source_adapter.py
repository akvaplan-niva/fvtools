"""Stable source-description for operational ROMS and collocated NorKyst hindcast.

The two products share ROMS sigma levels, but use different field names,
source grid locations and velocity coordinate frames.  This object contains
only metadata, so it is safe to send through multiprocessing/SLURM jobs.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ROMSSourceAdapter:
    fields: dict[str, str]
    collocated: bool = False
    velocity_frame: str = 'grid'  # 'grid' (xi/eta) or 'east_north'

    def __post_init__(self):
        if self.velocity_frame not in ('grid', 'east_north'):
            raise ValueError(f'Invalid ROMS velocity frame: {self.velocity_frame}')

    def field(self, logical_name: str) -> str:
        return self.fields.get(logical_name, logical_name)


NATIVE_ROMS = ROMSSourceAdapter(fields={})

# Names verified against the 2024-04-18 NorKyst v3 sdepth archive metadata.
# Fields on the collocated (Y,X) grid; the source has s_rho and s_w sigma axes.
NORKYST_V3_HINDCAST = ROMSSourceAdapter(
    collocated=True,
    velocity_frame='east_north',
    fields={
        'salt': 'salinity',
        'temp': 'temperature',
        'zeta': 'zeta',
        'u': 'u_eastward',
        'v': 'v_northward',
        'ubar': 'ubar_eastward',
        'vbar': 'vbar_northward',
    },
)
