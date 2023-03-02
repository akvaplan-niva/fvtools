# ---------------------------------------------------------------------
#           Create a ngrd.npy file for the nesting routines
# ---------------------------------------------------------------------
import sys
import fvtools.nesting.get_roms_ngrd as grn
import fvtools.nesting.get_fvcom_ngrd as gfn

def main(mesh, 
         R        = None, 
         mother   = None):
    """
    Generic "ngrd.npy" builder for nestfile creation

    Parameters:
    ----
    mesh:     'M.npy' file
    R:        Nestingzone width  (ROMS  - FVCOM nesting)
    mother:   fvcom-mother mesh  (FVCOM - FVCOM nesting)

    hes@akvaplan.niva.no
    """
    if mother is not None:
        gfn.main(mesh, mother)
        
    if R is not None:
        grn.main(mesh, R)

    if mother is None and R is None:
        raise ValueError('You must either provide a mother FVCOM mesh, or a nestingzone radius')
