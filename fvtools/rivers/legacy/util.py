import f90nml
import pandas as pd

def load_river_nml(fname):
    """
    Translates a river namelist into a pandas DataFrame.
    """
    nml = f90nml.read('input/RiverNamelist.nml').todict()['nml_river']
    df = pd.DataFrame([
        tuple(dct.values())[:-1] for dct in nml
    ], columns=['river_name', 'river_file', 'river_grid_location'])
    return df
