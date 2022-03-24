"""
Some helpers for biogeochemistry.

General interest: https://www.ices.dk/data/tools/Pages/Unit-conversions.aspx
"""

g_N_per_mol = 14.007
mol_N_per_g = 1/g_N_per_mol
mmol_N_per_kg = mol_N_per_g * 1e6

g_N_per_day_per_PE = 12

# SCOR WG 142:
# https://repository.oceanbestpractices.org/bitstream/handle/11329/417/56281.pdf?sequence=1&isAllowed=y
umol_DOXY_per_ml_at_STP = 44.6596

ml_DOXY_per_mg_at_STP = 0.7
