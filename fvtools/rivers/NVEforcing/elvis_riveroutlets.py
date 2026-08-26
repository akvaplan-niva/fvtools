import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
import numpy as np
import re
import shapely as shp
from pykdtree.kdtree import KDTree
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def main(
        osm_coastline = 'coastlines/coastlines-split-4326/lines.shp', 
        elvis = 'NVE/NVEData/Elv/Elv_Elvenett.shp', 
        nedbor = 'NVE/NVEData/Nedborfelt/Nedborfelt_NedborfeltTilHav.shp',
        vassdrag = 'NVE/NVEData/Nedborfelt_Vassdragsomr.shp'
        ):
    
    # Load the coastline
    print('Load the coastline')
    coast = load_osm_coast(osm_coastline)

    # Load the river database
    print('\nLoad ELVIS')
    df = load_elvis(elvis)

    print('- Remove duplicate data about each river, find the river outlet')
    rivers = df.groupby('rivers').apply(lambda riv: process_river(riv, coast), include_groups = False).droplevel(1)

    # Load information about the watersheds and associate them with the correct river
    rivers = match_river_and_watershed(rivers, gpd.read_file(nedbor))

    # Add vassdragsområde to the elvis database
    rivers = rivers.reset_index()
    rivers['vassdragsomraade'] = rivers.rivers.apply(lambda x: int(x.split('-')[0]))
    rivers = rivers.set_index('rivers')

    # Calculate the fraction of the vassdragsområde which is covered by each river
    nbf = gpd.read_file(vassdrag).drop(columns = ['ekspType', 'objType'])
    rivers = estimate_river_area_fraction_of_vassdragsomraade(rivers, nbf)
    return rivers

# ---------------------------------------------------------------------------------------------------------------------
#                                          Load the relevant input files
# ---------------------------------------------------------------------------------------------------------------------
def load_osm_coast(osm_coastline):
    '''
    Identify where the river meets the ocean by where the river is closest to the OSM coastline.

    Maybe (?) better to use the same coastline as we used when making the mesh (Kartverket)
    '''
    coast = gpd.read_file(osm_coastline)

    print('- Clip to Norway')
    coast = coast.clip([3.4262222, 57.790177, 31.3430051, 71.1662108])

    print('- Project to UTM33')
    coast = coast.to_crs('epsg:32633')

    print('- Set up a KDTree for the coast')
    # Next up is figuring out where these rivers connect to the ocean
    return KDTree(coast.geometry.get_coordinates().to_numpy())

def load_elvis(elvis):
    '''
    Load the ELVIS database to memory
    '''
    # Read the file
    df = gpd.read_file(elvis)

    # Drop columns we won't use
    print("- Drop columns we won't use")
    df = df[['elvID', 'vassdragNr', 'nbfVassNr', 'elvenavn', 'regulert', 'grenseElv', 'geometry']]

    # Remove segments that exit thorough the border
    print('- Drop rivers crossing the border')
    df = df[np.isnan(df['grenseElv'])]

    # Only keep valid elvID and vassdragNr
    print('- Drop segments with unknown river ID and vassdragsnummer')
    df = df[[type(d) == str for d in df['elvID']]]
    df = df[[type(d) == str for d in df['vassdragNr']]]

    # Identify unique rivers
    print('- Identify which unique river each river segment is a part of')
    df['rivers'] = df['elvID'].apply(lambda x: unique_river(x))

    # Remove segments that contains letters in the range [b-y], since these can not be connected to the coastline
    #print('- Remove segments that can not possibly be connected to the ocean') - some, however, were due to erroneous labelling in Elvis...
    #df = df[~df['vassdragNr'].apply(lambda x: bool(re.search(r"[b-y]", x, re.IGNORECASE)))]

    # Remove segments whose nedbørsfelt does not point to a hovedfelt - will keep rivers we do not know the nedbørsfelt of
    df = df[~df['nbfVassNr'].apply(lambda x: bool(re.search(r"[a-y]", x, re.IGNORECASE)) if type(x) == str else False)]

    # Store which vassdragsområde we're in, remove those that do not drain to Norway
    df['vassdragsomraade'] = df['rivers'].apply(lambda x: int(x.split('-')[0]))
    df = df[df['vassdragsomraade'] < 248]
    return df

#                                 Helper functions for matching rivers with watersheds
# -----------------------------------------------------------------------------------------------------------------------
def match_river_and_watershed(rivers, nbf):
    '''
    We know the size of some watersheds connected to rivers (roughly 1850 of 25 000 rivers).
    - here we match those 1850 rivers with corresponding watershed
    '''
    nbf['vassdragsomraade'] = nbf.vassdragNr.apply(lambda x: int(x.split('.')[0]))
    nbf = nbf.set_index('vassdragNr')
    nbf = nbf[nbf.vassdragsomraade < 248]

    # Hack
    update_these = {
        '177-35': [None, '177.13Z'],
        '161-23': [None, '161.23Z'],
        '161-18': [None, '161.22Z'],
        '050-35': ['050.422Z', '050.42Z'],
        '042-112': ['042.711Z', '042.71Z'],
    }

    # Odd edge case where there is a watershed but not a river
    # - '134.21Z' : has been split into two watersheds in Elvis
    # - '159.81Z' : has been split into multiple watersheds in Elvis

    remove_these = ['134.21Z', '159.81Z']

    for here in update_these:
        if rivers.loc[here, 'nedborsfelt'] == update_these[here][0]:
            rivers.loc[here, 'nedborsfelt'] = update_these[here][1]
        elif rivers.loc[here, 'nedborsfelt'] == update_these[here][1]:
            pass
        else:
            error_message = \
                f'\nWe expected to find nedbørsfelt = {update_these[here][0]} at elvID {here}, but found {rivers.loc[here, 'nedborsfelt']}.' + \
                f'\nThe Elvis or the Nedbørsfelt database no longer contain the same error as when we inserted this fix.' + \
                f'\nYou have to check if error handling is still required, and if new errors have been introduced to Elvis.' +\
                f'\nThis can be following the method outlined in riveroutlets.py, check_if_error_control_is_required.'
            raise ValueError(error_message)
    
    # Drop
    for to_drop in remove_these:
        nbf = nbf.drop(to_drop)
    
    rivers = rivers.reset_index()
    rivers = rivers.set_index('nedborsfelt')
    rivers.areal = None

    rivers.loc[nbf.reset_index().vassdragNr, 'areal'] = nbf.areal_km2
    rivers = rivers.reset_index().set_index('rivers')
    return rivers

#                                     Helper functions for the river outlet search
# -----------------------------------------------------------------------------------------------------------------------

def unique_river(x):
    '''
    elvID (num-num-num) is structured so that the first two numbers indicate vassdragsnummer and unique river,
    the last number indicates the unique river segment ID. Here we make a new dict entry so that we can group
    rivers by their unique river vassdragsnumber-id
    '''
    return '-'.join([str(n) for n in x.split('-')[:2]])

def is_numeric_regex(s):
  """
  Checks if a string contains only numeric digits using a regular expression (e.g. look for small rivers)

  Args:
    s: The input string.

  Returns:
    True if the string contains only digits, False otherwise.
  """
  return bool(re.fullmatch(r'\d+', s))

def is_hoved_regex(s):
  """
  Checks if a string contains only numeric digits using a regular expression.

  Args:
    s: The input string.

  Returns:
    True if the string contains only digits, False otherwise.
  """
  return s[0] == 'A'

def til_hav(x):
    '''
    Finn elvesegmenter som muligens løper ut til havet
    '''
    try:
        if is_numeric_regex(x.split('.')[-1]) or is_hoved_regex(x.split('.')[-1]):
            return True
        else:
            return False
    except:
        raise ValueError(f'something went wrong with {x}')

def process_river(g, coast):
    '''
    From each river, we need:
     - potential endpoints
     - river ID
     - river name / hierarchy
     - regulated or not?
     - potential endpoints
     
     # Add vassdragsområde, at least one river has another first number than vassdragsområde
    '''
    # Copy the rivers name
    elvenavn   = [n for n in g.elvenavn.unique() if n != None]
    if not any(elvenavn):
        elvenavn = None
    else:
        elvenavn = elvenavn[0]

    # This one will identify unique vassdrag
    nbf = [n for n in g.nbfVassNr.unique() if n != None]
    if not any(nbf):
        nbf = None
    else:
        nbf = nbf[0]

    if len(g) > 1:
        coords = []
        for f in g.geometry:
            if type(f) == shp.MultiLineString:
                all_coords = []
                for l in f.geoms:
                    all_coords.extend(list(l.coords))
                coords.append(shp.LineString(all_coords).coords[-1][:2])
            else:
                coords.append(f.coords[-1][:2])

        # As array
        coords = np.array(coords)
        d, _ = coast.query(coords)
        coords = np.array(coords[d==d.min(), :])[0]
    else:
        # When we've only got one polygon
        coords = np.array(g.geometry.iloc[0].coords[-1][:2])

    return gpd.GeoDataFrame(
        {
            'elvenavn': elvenavn,
            'nedborsfelt': nbf,
            'regulated': g.regulert.iloc[0],
            'x_outlet': [float(coords[0])],
            'y_outlet': [float(coords[1])]
        }
    )

def quality_control(rivers):
    url='https://wms.geonorge.no/skwms1/wms.topograatone?service=wms&request=getcapabilities'
    layers=['topograatone']
    plt.close('all')
    crs = ccrs.Mercator()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=ccrs.epsg(32633))
    fig.canvas.draw()         
    #fig.set_tight_layout(True)
    ax.add_wms(wms=url, layers=layers)
    plt.plot(rivers.x_outlet, rivers.y_outlet, 'r.')
    elv = rivers[rivers.elvenavn == 'Kilelva']
    print(elv)
    plt.scatter(elv.x_outlet, elv.y_outlet, c = 'g')

# -----> since we had to hardcode a fix to deal with mismatch between NVE databases

def check_if_error_control_is_required(osm_coastline, elvis, nedbor):
    '''
    The Elvis and the Nedbørsfelt databases somehow did not match up well. 
    '''
    print('Load the coastline')
    coast = load_osm_coast(osm_coastline)

    # Load the river database
    print('\nLoad ELVIS')
    df = load_elvis(elvis)

    print('- Remove duplicate data about each river')
    rivers = df.groupby('rivers').apply(lambda riv: process_river(riv, coast), include_groups=False).droplevel(1)

    # Also load information about the nedbørsfelt
    nbf = gpd.read_file(nedbor).set_index('vassdragNr')

    # Load the information about watersheds areas into the river datastructure
    test = rivers.set_index('nedborsfelt')
    nbf = nbf.reset_index()
    nbf['vassdragsomraade'] = nbf['vassdragNr'].apply(lambda x: int(x.split('.')[0]))
    nbf = nbf[nbf.vassdragsomraade < 248]

    # Set watershed area to the rivers we know of - If this fails, geopandas will tell you which watersheds are missing in the rivers database
    test.loc[nbf.vassdragNr, 'areal'] = nbf.areal_km2

    # Have a look at which watersheds are missing - these are the ones I struggled with
    # ----
    nbf.set_index('vassdragNr').loc[['177.13Z', '161.23Z', '161.22Z', '159.81Z', '134.21Z', '050.42Z', '042.71Z']]

    # Open https://atlas.nve.no/Html5Viewer/index.html?viewer=nveatlas# and figure out which rivers are in the missing watersheds

    # Have a look at what the nedbørsfelt is in one of the missing rivers
    rivid = '161-18'
    rivers.loc[rivid]

    # Plot to see that nothing is crazy wrong with the position of the river (that was never a problem for me, but who knows...)
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.UTM(33))
    ax.add_wms('https://wms.geonorge.no/skwms1/wms.topograatone?service=wms&request=getcapabilities', layers = ['topograatone'])
    plt.scatter(rivers.loc[rivid].x_outlet, rivers.loc[rivid].y_outlet, c = 'r')

    # Once you have understood what's wrong, we can either remove the problematic watersheds from the watershed file -- or 
    # update the rivers database with the correct watershed number

    # Plot the watershed to see where it is (sometimes mismatch between watershed database and what's on atlas for some weird reason)
    geom = nbf[nbf.vassdragNr == '050.42Z'].geometry
    g = np.array(np.array(geom.boundary.values)[0].coords)

    fig = plt.figure()
    ax = plt.axes(projection=ccrs.UTM(33))
    ax.add_wms('https://wms.geonorge.no/skwms1/wms.topograatone?service=wms&request=getcapabilities', layers = ['topograatone'])
    ax.plot(g[:, 0], g[:, 1], c = 'r')

    # Do this the pragmatic way for now: Manually adjust the places that are wrong, and send NVE an email to notify them.
    # Store the value it has now, and if that's the value when we try to overwrite, we continue
    update_these = {
        '177-35': [None, '177.13Z'],
        '161-23': [None, '161.23Z'],
        '161-18': [None, '161.22Z'],
        '050-35': ['050.422Z', '050.42Z'],
        '042-112': ['042.711Z', '042.71Z'],
    }

    # Odd edge case where there is a watershed but not a river
    # - '134.21Z' : has been split into two watersheds in Elvis

    remove_these = ['134.21Z']

    for here in update_these:
        if rivers.loc[here, 'nedborsfelt'] == update_these[here][0]:
            rivers.loc[here, 'nedborsfelt'] = update_these[here][1]
        elif rivers.loc[here, 'nedborsfelt'] == update_these[here][1]:
            pass
        else:
            error_message = \
                f'\nWe expected to find nedbørsfelt = {update_these[here][0]} at elvID {here}, but found {rivers.loc[here, 'nedborsfelt']}.' + \
                f'\nThe Elvis or the Nedbørsfelt database no longer contain the same error.' + \
                f'\nYou have to check if the error handling is still required.' +\
                f'\nThis can be following the method outlined in riveroutlets.py, check_if_error_control_is_required.'
            raise ValueError(error_message)
    
    # Drop
    for to_drop in remove_these:
        nbf = nbf.drop(to_drop)

def estimate_river_area_fraction_of_vassdragsomraade(rivers, vassdrag):
    '''
    Based on the vassdragsområde area and known river watersheds, we 
    '''
    vassdrag.vassOmrNr = vassdrag.vassOmrNr.astype(int)
    vassdrag = vassdrag[vassdrag.vassOmrNr < 248].set_index('vassOmrNr')

    large_rivers = rivers[~np.isnan(rivers.areal)]
    small_rivers = rivers[np.isnan(rivers.areal)]

    known_watershed = large_rivers.groupby('vassdragsomraade').sum('areal')
    vassdrag['small_rivers_area'] = vassdrag.arealLand - known_watershed.areal

    indices =  np.isnan(vassdrag['small_rivers_area'])
    vassdrag.loc[indices, "small_rivers_area"]  = vassdrag.loc[indices, "arealLand"]

    large_rivers.loc[:, 'area_fraction'] = large_rivers[['vassdragsomraade', 'areal']].apply(lambda x: x.areal/vassdrag.loc[x.vassdragsomraade, 'arealLand'], axis = 1)
    small_rivers.loc[:, 'num_small'] = small_rivers.vassdragsomraade.apply(lambda x: len(np.where(small_rivers.vassdragsomraade == x)[0]))
    small_rivers.loc[:, 'areal'] = small_rivers[['vassdragsomraade', 'num_small']].apply(lambda x: vassdrag.loc[x.vassdragsomraade, 'small_rivers_area']/x.num_small, axis = 1)
    small_rivers.loc[:, 'area_fraction'] = small_rivers[['vassdragsomraade', 'areal']].apply(lambda x: x.areal/vassdrag.loc[x.vassdragsomraade, 'arealLand'], axis = 1)

    return pd.concat([large_rivers, small_rivers.drop(columns = 'num_small')]).reset_index().set_index('rivers')
