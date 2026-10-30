# --------------------------------------------------
# Reads two coast files, merges the coasts, removes
# duplicate islands and stores this new coast 
# as a text file.
# --------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from shapely.geometry.polygon import Polygon

def main(infile1,infile2,outfile='merged_coast.txt'):
    '''
    read two coast files, merge them and store them to outfile
    '''
    # read coast files
    mainland1, islands1 = read_poly_data(infile1)
    mainland2, islands2 = read_poly_data(infile2)
    
    # remove overlapping islands
    print('removing duplicate islands')
    islands             = merge_islands(islands1,islands2)

    # Combine two mainlands to make one unique
    print('merging the mainlands')
    mainland            = merge_mainlands(mainland1,mainland2)

    # Add islands and mainland polygon to get the complete coast
    merged_coast        = mainland+islands

    # Write these polygons to a coastfile
    write_poly_data(outfile, merged_coast)
 
def read_poly_data(file):
    '''
    Read coast line data from a text file.

    File format: First line is name, second line number of polygons,
    then comes all the polygon data. First line of polygon data is
    number of vertices, then follows the x, y coordinates for each
    vertex.
    
    Output: A shapely MultiPolygon object containing the polygon data
    '''
    print "Reading polygons from: ", file
    with open(file) as f:
        data  = f.readlines()

    data      = map(str.rstrip, data)
    name      = data[0]
    n_poly    = int(data[1])
    p_data    = map(str.split, data[2:])
    poly_list = np.array(np.zeros(n_poly), dtype=object)

    i = 0
    for n in xrange(n_poly):
        n_points = int(p_data[i][0])
        i += 1
        poly = p_data[i:i+n_points]
        poly_list[n] = np.array(poly, dtype=float)
        i += n_points

    print('Polygons read: ', n_poly)

    # remove the mainland (assume this is the first in the list)
    # -------------------------------------------------------------
    mainland  = Polygon(poly_list[0])
    poly_list = np.delete(poly_list,0)
    islands   = MultiPolygon(map(Polygon, poly_list))
    
    return mainland, islands

def merge_islands(coast1,coast2):
    '''
    combine two multipolygons to one multipolygon
    '''
    polys = []

    # no need to loop over islands outside the overlapping domain:
    # --------------------------------
    coords = north_south(coast1,coast2)

    # Extract all polygons in the multipolygons, store in lists
    # ------------------------------------------------------
    polys1_overlap = []
    polys1_unique  = []
    for poly in coast1:
        bound = poly.bounds
        if bound[3] < coords.north and bound[1] > coords.south:
            polys1_overlap.append(poly)
        
        else:
            polys1_unique.append(poly)

    polys2_overlap = []
    polys2_unique  = []
    for poly in coast2:
        bound = poly.bounds
        if bound[3] <= coords.north and bound[1] >= coords.south:
            polys2_overlap.append(poly)
        
        else:
            polys2_unique.append(poly)

    # loop over the overlapping polygons to remove those who truly overlap
    # --------------------------------------------------------
    uniqpolies    = []
    suspect_polys = polys1_overlap + polys2_overlap
    initial       = len(suspect_polys)
    for poly in suspect_polys:
        if not any(p.equals(poly) for p in uniqpolies):
            uniqpolies.append(poly)
    final = len(uniqpolies)
    print(str(initial-final) + ' islands overlapped')
    islands = uniqpolies+polys1_unique+polys2_unique

    return islands

def north_south(coast1,coast2):
    # Find northern/southern boundary
    # ---------------------------------------------------------

    class coords: pass
    # Find the northernmost polygon
    # ------------------------
    if coast1.boundary.bounds[3]>coast2.boundary.bounds[3]:
        northernmost = coast1.boundary.bounds
    else:
        northernmost = coast2.boundary.bounds

    # Southernmost
    # ------------------------
    if coast1.boundary.bounds[1]<coast2.boundary.bounds[1]:
        southernmost = coast1.boundary.bounds
    else:
        southernmost = coast2.boundary.bounds

    # store those coordinates
    # ---------------------------------------------------------
    coords.north     = southernmost[3]+2000
    coords.south     = northernmost[1]-2000

    #  eastern/western boundary
    # ---------------------------------------------------------
    if coast1.boundary.bounds[2]>coast2.boundary.bounds[2]:
        easternmost  = coast1.boundary.bounds
        westernmost  = coast2.boundary.bounds
    else:
        easternmost  = coast2.boundary.bounds
        westernmost  = coast1.boundary.bounds
        
    coords.east      = easternmost[2]+2000

    if coast1.boundary.bounds[0]>coast2.boundary.bounds[0]:
        easternmost  = coast2.boundary.bounds
        westernmost  = coast1.boundary.bounds
    else:
        easternmost  = coast1.boundary.bounds
        westernmost  = coast2.boundary.bounds

    coords.west      = westernmost[0]-2000
    
    return coords


def merge_mainlands(mainland1, mainland2):
    '''
    Merge mainlands to big mainland coast polygon.
    '''
    mainl = unary_union([mainland1,mainland2])
    plt.plot(*mainl.exterior.xy)
    plt.axis('equal')
    plt.title('Make sure that no parts of the mainland got clipped out')
    plt.show()
    return [mainl]

def write_poly_data(file, mp):
    '''Write coast line data to text file.

    File format: First line is name, second line number of polygons,
    then comes all the polygon data. First line of polygon data is
    number of vertices, then follows the x, y coordinates for each
    vertex.
    
    Input: A shapely MultiPolygon object containing the polygon data
    '''
    with open(file, 'w') as f:
        print "Writing polygons to: ", file
        f.write('COAST\n')
        f.write('%d\n' % len(mp))
        for poly in mp:
            x, y = poly.exterior.xy
            n = len(x)
            f.write('%d 1\n' % n)
            for i in range(n):
                f.write('%.4f %.4f\n' % (x[i],y[i]))
        print "Polygons written: ", len(mp)

def crop_coastfile(coastfile):
    '''
    extract parts of a coastfile to speed up the plotting routines
    '''
    
