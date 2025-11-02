import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import re
import shapely as shp
from pykdtree.kdtree import KDTree
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def main(
        osm_coastline = 'coastlines/coastlines-split-4326/lines.shp', 
        elvis = 'elvis/NVEKartdata/NVEData/Elv/Elv_Elvenett.shp', 
        nedbor = 'elvis/NVEKartdata/NVEData/Nedborfelt/Nedborfelt_NedborfeltTilHav.shp'
        ):
    
    # Load the coastline
    print('Load the coastline')
    coast = load_osm_coast(osm_coastline)

    # Load the river database
    print('\nLoad ELVIS')
    df = load_elvis(elvis)

    # Also load information about the nedbørsfelt
    nbf = gpd.read_file(nedbor).set_index('vassdragNr')

    print('- Remove duplicate data about each river')
    rivers = df.groupby('rivers').apply(lambda riv: process_river(riv, coast), include_groups=False).droplevel(1)

    # So this is the river database. Now we want to connect it to the nedbørsfelt database when we differentiate
    # between big and small rivers

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
    print('- Remove segments that can not possibly be connected to the ocean')
    df = df[~df['vassdragNr'].apply(lambda x: bool(re.search(r"[b-y]", x, re.IGNORECASE)))]

    # Remove segments whose nedbørsfelt does not point to a hovedfelt - will keep rivers we do not know the nedbørsfelt of
    df = df[~df['nbfVassNr'].apply(lambda x: bool(re.search(r"[a-y]", x, re.IGNORECASE)) if type(x) == str else False)]

    # Store which vassdragsområde we're in, remove those that do not drain to Norway
    df['vassdragsomraade'] = df['rivers'].apply(lambda x: int(x.split('-')[0]))
    df = df[df['vassdragsomraade'] < 248]
    return df

#                                             Helper functions
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