from urllib.request import Request, urlopen
from pykdtree.kdtree import KDTree
from scipy.signal import medfilt
from datetime import datetime
import json
import numpy as np
import pandas as pd
import re

def main(rivers, api_key = 'Qzi9yp/PWUyd5cx2Age5pw==', max_distance_to_ocean = 5000, max_meters_over_sealevel = 50):
    '''
    Download observed river temperatures from the NVE hydapi portal. Calculate seasonal statistics, clean the data for
    spikes and create a watershed dependent river temperature profile for the country.
    - reads a rivers file created by riveroutlets

    Choose observed river temperatures from stations that are
    - at most max_distance_to_ocean meters from the river outlet
    - at most max_meters_over_sealevel meters over sealevel
    '''
    # For the api request
    request_headers = {
        "Accept": "application/json",
        "X-API-Key": api_key
    }

    print('Read available stations from hydapi')
    stations = _find_relevant_stations(request_headers, max_meters_over_sealevel)
    num_stations = len(stations)
    print(f'- found {num_stations} stations\n')

    print('Figure out which river connects to the stations')
    stations = _connect_rivers_to_stations(stations, rivers)

    print(f'- Found stations recording water temperature at {num_stations} unique rivers.')

    num_stations_near_river = len(stations)
    stations = stations[stations.distance_to_outlet < max_distance_to_ocean]

    print(f'- Filtered out {num_stations_near_river - len(stations)} due to distance to the coastline\n')

    download_from_n_stations = len(stations)
    print(f'Download data from {download_from_n_stations} stations')
    stations, observations = _download_temperatures_hydapi(stations, request_headers)
    num_with_data = len(stations)

    if num_with_data != num_stations:
        print(f'- Found data at {num_with_data} stations, {download_from_n_stations - num_with_data} stations were empty or not quality controlled.\n')
    else:
        print('- Found data at all requested stations.\n')

    print(f'Process temperature data')
    print(f'- Removing erroneous spikes (values > 30 C) and running a filter to remove noise')
    observations, climatology = _process_temperatures_at_stations(observations)

    # Create a temperature climatology for each vassdragsområde
    print(f'- Create a full climatology for the entire coast based on filtered temperatures')
    climatology_full = _create_watershed_temperatures(climatology)

    return observations, climatology_full

def _find_relevant_stations(request_headers, meters_over_sealevel):
    '''
    Check which (active) stations we can fetch data from
    '''
    seriesurl = "https://hydapi.nve.no/api/v1/Stations?Parameter=1003&Active=OnlyActive"

    r = Request(seriesurl, headers = request_headers)
    response = urlopen(r)
    content = response.read().decode('utf-8')
    stations = pd.DataFrame(json.loads(content)['data'])[
        ['stationId', 'stationName', 'utmEast_Z33', 'utmNorth_Z33', 'masl', 'riverName', 'hierarchy', 'regineNo', 'seriesList']
        ]
        
    # Remove stations that have not recorded water temperature
    filters = []
    for _, here in stations.iterrows():
        filters.append(any([d['parameter'] == 1003 for d in here.seriesList]))
    stations = stations[filters].drop(columns = ['seriesList'])

    # Remove stations with invalid vassdragNr
    stations = stations[stations.regineNo.apply(lambda x: _clean_vassdragNr(x))]

    # Do not look at high altitude staions
    stations = stations[stations.masl < meters_over_sealevel]

    # Get the vassdragsområde
    stations['vassdragsomraade'] = stations.regineNo.apply(lambda x: int(x.split('.')[0]))

    # Remove stations not draining to the norwegian coast
    stations = stations[stations.vassdragsomraade < 248]

    # Figure out which "catchment to ocean" this stations is on
    stations['nedborsfelt'] = stations.regineNo.apply(lambda x: _get_nedborsfelt_til_hav(x))

    # Remove stations that does not have a defined area (does not have Z in it)
    return stations[stations.nedborsfelt.apply(lambda x: _remove_unconnected(x))].reset_index().drop(columns = ['index', 'regineNo'])

def _connect_rivers_to_stations(stations, rivers):
    '''
    Uses the watershed id to connect observation stations to specific rivers
    '''
    # Find the river ID assosiated with the stations nedbørsfelt
    rivers_connected_to_observations = rivers.reset_index().set_index('nedborsfelt').loc[stations.nedborsfelt]

    # Compute the distance (as the crow flies) from the river outlet to the station
    river_tree = KDTree(np.array([rivers_connected_to_observations.x_outlet, rivers_connected_to_observations.y_outlet]).T)
    stations['distance_to_outlet'] = river_tree.query(np.array([stations.utmEast_Z33, stations.utmNorth_Z33]).T)[0]

    # Store the riverID to the station, and use it as the index
    inds = stations.groupby('nedborsfelt').distance_to_outlet.idxmin()
    stations = stations.loc[inds]
    stations['rivers'] = rivers.reset_index().set_index('nedborsfelt').loc[stations.nedborsfelt].rivers.values
    return stations.set_index('rivers')

def _download_temperatures_hydapi(stations, request_headers):
    '''
    Download observed temperatures from NVE
    '''
    # API request to hydapi. We're looking for water temperature (1003) with daily resolution (1440), but just data from Jan 1.st 2010 and onwards
    data = []
    for station in stations.stationId:
        # Find all stations 
        try:
            seriesurl = f"https://hydapi.nve.no/api/v1/Observations?StationId={station}&Parameter=1003&ResolutionTime=1440&ReferenceTime=2010-01-01/"
            r = Request(seriesurl, headers = request_headers)
            response = urlopen(r)
            content = response.read().decode('utf-8')
            data.append(pd.DataFrame(json.loads(content)['data']))
        except:
            print(f'- Did not find station {station}')

    # Remove empty values, convert the data read from hydapi (dicts) to dataframes
    observed_temperature = {}
    data_available = []
    for d in data:
        tmp = pd.DataFrame(d.observations[0])

        # Remove data that has not at least been primary controlled
        try:
            tmp.loc[tmp.quality < 2, 'value'] = np.nan
            tmp = tmp[~np.isnan(tmp.value)]
            is_any = tmp.any().any()
        except:
            is_any = any(tmp)
            

        if is_any:
            tmp = tmp.set_index('time')

            # Store with the riverId, not the station ID
            observed_temperature[stations[stations.stationId == d.stationId.iloc[0]].index[0]] = tmp
            data_available.append(True)
        else:
            data_available.append(False)
    return stations[data_available], observed_temperature

def _get_nedborsfelt_til_hav(x):
    '''
    From delnedbørsfelt to nedbørsfelt til hav
    '''
    vassdragsomraade, delnedbor = x.split('.')

    if re.match('[a-zA-Z]', delnedbor[0]):
        # we're in a hovedfelt
        here = 'Z'
    else:
        out = re.split(r"[A-Z]", delnedbor)
        if len(out) > 1:
            here = out[0] + 'Z'
        else:
            here = out[0]
    return f'{vassdragsomraade}.{here}'

def _clean_vassdragNr(x):
    '''
    The vassdragnr in the database is actually not always valid (just the vassdragsområde, and sometimes not a string)
    - remove values we can't interpret and dump.
    '''
    if type(x) == str:
        if len(x) > 3:
            return True
        else:
            return False
    else:
        return False

def _remove_unconnected(x):
    if 'Z' in x:
        return True
    else:
        return False
    
def _process_temperatures_at_stations(observations, kernel_size = 7):
    '''
    Remove spikes and compute yearly climatology for each station
    '''
    climatology = {}

    # Loop over all observation stations
    for obs in observations:
        # Remove spikes and negative values
        observations[obs].loc[observations[obs].value > 30, 'value'] = np.nan

        filtered = medfilt(observations[obs].value, kernel_size = kernel_size)
        filtered[filtered < 0] = 0

        # Store to the dataframe
        observations[obs]['filtered_temperature'] = filtered

        # Interpolate values (though no extrapolation)
        observations[obs].filtered_temperature = observations[obs].filtered_temperature.interpolate(method='linear')

        # Create annual climatology for each river
        observations[obs]['Dates'] = pd.to_datetime(observations[obs].index)
        observations[obs]['DayOfYear'] = observations[obs]['Dates'].dt.dayofyear

        # Only make a climatology for rivers with long observational records
        diff = datetime.fromisoformat(observations[obs].index[-1]) - datetime.fromisoformat(observations[obs].index[0])

        if diff.days > 5*365:
            climatology[obs] = observations[obs].groupby(observations[obs]['DayOfYear']).mean()[['value', 'filtered_temperature']]

    return observations, climatology

def _create_watershed_temperatures(climatology):
    '''
    Use station climatologies to create temperature forcing for watershed areas
    '''
    # We want to use the the filtered values when making the climatology
    climatology_filtered = pd.DataFrame({c: climatology[c].filtered_temperature for c in climatology}).T.reset_index()

    # Identify each observations vassdragsområde
    climatology_filtered['vassdragsomraade'] = climatology_filtered['index'].apply(lambda x: int(x.split('-')[0]))

    # Make room for all vassdragsområder
    climatology_observed_vassdrag = climatology_filtered.drop(columns = 'index').groupby('vassdragsomraade').mean()

    # Prepare to fill in the blanks at watersheds without data
    climatology_full = pd.DataFrame(data = np.nan*np.ones((248, 366)), index = range(1,249), columns = range(1,367))
    climatology_full.loc[climatology_observed_vassdrag.index, climatology_observed_vassdrag.columns] = climatology_observed_vassdrag
    return climatology_full.interpolate(method = 'linear').ffill().bfill()