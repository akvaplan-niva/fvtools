"""FVCOM sigma-layer weighted barotropic correction for any forcing source.

The reference ua/va remain unchanged; only u/v are offset by a constant in
sigma for each element. Delegates to the already tested CMEMS implementation.
"""
from __future__ import annotations

from fvtools.interpolators.glorys_barotropic import correct_fvcom_barotropic

__all__ = ('correct_fvcom_barotropic',)
