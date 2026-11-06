# Shared FVCOM restart helpers

`restart_time.py` reads precise Itime/Itime2 (or the legacy NetCDF time fallback
for CMEMS). `fields.py` describes fixed FVCOM field dimensions; u/v/ua/va are
always cell-centered. `restart_io.py` provides same-filesystem atomic updates.

`empty_initial.py` creates initial NetCDF files; `plot_restart.py` provides QC
plots. TPXO's own lock/idempotency mechanism remains in `nesting.tpxo`.
