"""BuildRivers public interface and FVCOM river-forcing workflow.

Implementation classes live in fvtools.prepro._buildrivers.
Public imports from this module remain available.
"""
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np

from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.prepro._buildrivers.constants import version
from fvtools.prepro._buildrivers.sources import HBVRunoff, RiverPositions, RiverTemperatures
from fvtools.prepro._buildrivers.interaction import DraggablePoints, CropRivers
from fvtools.prepro._buildrivers.forcing import FVCOM_rivers

async def main(start, stop, vassdrag, mesh_dict, info = None):
    '''
    BuildRiver use data from the NVE HBV model and feed all the mapped rivers leading to the ocean to FVCOM

    Parameters:
    ----
    start:     yyyy-mm-dd
    stop:      yyyy-mm-dd
    vassdrag:  [233, 234] etc, any array containing all ids (integers) will do.

    Optional:
    ----
    mesh_dict: (M.npy by default)
    info:      Dict where all paths and river specific settings are stored. Change the basic settings by giving it as
               an input: info. Get the basic settings by calling BuildRivers.get_input(),
               which can then be edited and passed back to main if other paths etc. are
               needed in the given experiment.

    hes@akvaplan.niva.no
    '''
    # Convert time
    start = datetime(*[int(t) for t in start.split('-')], tzinfo = timezone.utc)
    stop  = datetime(*[int(t) for t in stop.split('-')], tzinfo = timezone.utc)

    if info is None:
        info = get_input()

    # 1. Load the FVCOM grid, the national runoff data, the river outlet positions and the river temperature data. 
    #    Initialize the object that will move rivers to the mesh and ensure numerical stability
    # -------------------------------------------------------------------------------
    # - Read the mesh
    M = FVCOM_grid(mesh_dict)
    info['grid_projection'] = M.reference
    M.re_project(info['river_projection'])
    M.get_cfl(verbose=False)

    print('----------------------------------------------------------------------------')
    print(f'                       BuildRivers: {M.casename}')
    print('----------------------------------------------------------------------------')

    # - Load the necessary data and data handlers
    print('Compute grid connectivity and identify landpoints')
    Forcing   = FVCOM_rivers(info, M, vassdrag, start, stop)
    Runoff    = HBVRunoff(info['runoff'], start = start - timedelta(days = 1), stop = stop + timedelta(days=1))
    Positions = RiverPositions(info['riverpositions'], vassdrag, info['river_projection'])
    Temp      = RiverTemperatures(info['rivertemp'], start = start - timedelta(days=1), stop = stop + timedelta(days=1))

    # 2. Reduce the rivers to only load those that are relevant to this simulation
    #    and prepare to force the model
    # ------------------------------------------------------------------------------
    # - Remove the rivers that are too far away from land and too close to the obc
    print('\nSubset the river database to cover the model domain')
    Positions = await Forcing.crop_rivers_far_from_land_and_close_to_OBC(Positions)

    # - Re-distribute the runoff according to catchment area
    print('- Determine runoff from rivers by the rivers catchment area')
    Forcing.redistribute_runoff(Positions, Runoff)

    # - Add temperatures to the rivers
    print('\nAdd temperature profiles to the rivers')
    print(f'- read temperature databases from {info["rivertemp"]}, calculate the river climatology for each watershed')
    river_temp = Temp.make_individual_river_climatology(Positions)

    print('- Add observed temperatures to the rivers where available')
    river_temp = Temp.insert_observed_temperature(river_temp, vassdrag)

    print('- Set the runoff and temperauture for each river in the model, interpolate to model time')
    Forcing.interpolate_forcing_to_model_time(Runoff, Temp, river_temp)

    print('\nConnect rivers to the mesh')
    Forcing.connect_to_mesh(M)

    # - Write to netCDF, write RiverNamelist.nml
    print('\nWrite forcing to file')
    Forcing.dump()
    Forcing.write_namelist()

    # Do a simple quality control, and write the necessary forcing files
    # - Show the rivers we'll use, and the values they will force FVCOM with
    print('\nFinished, plotting the forcing')
    M.re_project(info['grid_projection'])
    show_forcing(Forcing, M, Positions)
    M.re_project(info['river_projection'])

def get_input(river_data_path = '/nird/datapeak/NS9067K/store/fvcom-setup/Rivers'):
    """
    Pre-defined paths are stored here. They are distributed to other parts of the code via main.

    iloc:        Determine if the input is given as a flux at edge or at the node
    land buffer: For identifying rivers draining to the model domain, will include rivers within *land buffer* from the model coastline
    obc buffer:  For masking rivers within *obc buffer* from the model open boundary
    min_depth:   Minimum depth at rivers (since if h=0 (wetting/drying on), rivers would be infinitely wide to meet the stability criteria)
    Isplit:      Baroclinic split
    tideamp:     Tidal amplityde
    plot:        Show the results on a map
    rivertemp:   River temperature folder
    vassdrag:    File containing runoff data from vassdrag
    LargeRivers: File containing info about the large rivers in Norway
    SmallRivers: File containing info about the small rivers in Norway
    minrcoef:    Tunable parameter to determine the maximum volume of a CV we will let a river fill over a timestep
    river_projection: Coordinate system the river positions are stored in (UTM33)
    """
    info = {
        'iloc': 'edge',
        'land buffer': 1500,
        'buffer domain ratio': 0.1,
        'obc buffer': 5000,
        'min_depth': 3,
        'Isplit': 8,
        'tideamp': 1,
        'rivertemp': f'{river_data_path}/Temperature/',
        'runoff': f'{river_data_path}/Niva_2025_2018v20.05/',
        'riverpositions': f'{river_data_path}/RiverPositions/river_positions.csv',
        'minrcoef': 0.3,
        'river_projection': 'epsg:32633'
    }
    return info







# Show what we will write to the riverdata forcing
# ----
def show_forcing(obj, M, Rivers):
    """
    Simple figures to see that the routine got the basics right
    """
    plt.figure()
    M.georeference()
    plt.plot(obj.x_land, obj.y_land, 'k.', label = 'land nodes', zorder = 1)
    plt.scatter(
        obj.xriv, obj.yriv, 
        s = np.mean(obj.RiverTransport, axis = 0), 
        c = np.mean(obj.RiverTransport, axis = 0), 
        zorder = 5, 
        transform = Rivers.crs
        )
    plt.title('Average transport')
    plt.axis('equal')
    plt.colorbar(label = r'm$^3$ s$^{-1}$')
    plt.show(block = False)

    plt.figure()
    M.georeference()
    plt.scatter(obj.xriv, obj.yriv, obj.RiverTemp.max(axis = 0), c = obj.RiverTemp.max(axis = 0), cmap = 'inferno', zorder = 5, transform = Rivers.crs)
    plt.title('Max temperature in model period')
    plt.axis('equal')
    plt.colorbar(label = 'degrees celcius')
    plt.show(block = False)

class InputError(Exception): pass