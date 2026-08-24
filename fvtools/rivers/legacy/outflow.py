"""
This module helps finding where rivers drain into the sea.

It relies on publicly available NVE datasets (Regine units and ELVIS river database)
"""

import pandas as pd
import geopandas as gpd


def find_river_outflows(df):
    """
    Find the outflow (into the sea) of a shapefile with river networks.

    df: GeoDataFrame containing (subset of) ELVIS database
    """

    river_agg = gpd.GeoDataFrame(
        geometry=df.groupby('vassdragNr')['geometry'].apply(unary_union)
    )

    def get_most_downstream_river_stretches(e):
        """
        Find the most downstream portions of (directed) shapely linestrings
        """
        if isinstance(e, shp.geometry.LineString):
            return [e]
        elif isinstance(e, shp.geometry.MultiLineString):
            most_downstream = []
            for k, this_line in enumerate(e.geoms):
                ball_around_outflow = this_line.boundary.geoms[1].buffer(10)
                remaining_rivers = shp.geometry.MultiLineString([l for l in e.geoms if not this_line.equals(l)])
                this_is_most_downstream = not ball_around_outflow.intersects(remaining_rivers)
                if this_is_most_downstream:
                    most_downstream.append(k)
            return [e.geoms[i] for i in most_downstream]

    gdfs = []
    for k, geom in river_agg.geometry.iteritems():
        river_stretches = get_most_downstream_river_stretches(geom)
        gdf = gpd.GeoDataFrame(geometry=river_stretches, index=[k]*len(river_stretches))
        gdfs.append(gdf)

    outflow = pd.concat(gdfs).boundary.apply(lambda pt: pt.geoms[1]).to_frame(name='geometry')
    return outflow, river_agg
