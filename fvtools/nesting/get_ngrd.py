# ---------------------------------------------------------------------
#           Create a ngrd.npy file for the nesting routines
# ---------------------------------------------------------------------
import sys
import fvtools.nesting.get_roms_ngrd as grn
import fvtools.nesting.get_fvcom_ngrd as gfn

def main(mesh, **kwargs):
    """
    Generic "ngrd.npy" builder for nestfile creation

    Parameters:
    ----
    mesh:                'M.npy' file
    nrows:               Nestingzone width (in number of rows)  (ROMS  - FVCOM nesting)
    remove_land_squares: remove squares near land (other than the first obc row)
    mother:              fvcom-mother mesh  (FVCOM - FVCOM nesting)

    hes@akvaplan.niva.no
    """

    if 'R' in kwargs:
        raise ValueError('the nest grid maker was changed so that we extract rows, not an area within a search radius')
    
    elif 'mother' in kwargs:
        gfn.main(mesh, kwargs['mother'])
        
    elif 'nrows' in kwargs:
        if 'remove_land_squares' not in kwargs:
            kwargs['remove_land_squares'] = True
        grn.main(mesh, nrows = kwargs['nrows'], remove_land_squares = kwargs['remove_land_squares'])

    else:
        raise ValueError('You must either provide a mother FVCOM mesh, or a number of nestingzone rows')
