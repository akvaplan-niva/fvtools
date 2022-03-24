import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from shapely.geometry.polygon import Polygon



def parse_command_line():
    ''' Parse command line arguments'''

    parser = ArgumentParser(description='')
    parser.add_argument("-coastline_input_file", "-c",
                         help="Path to file with input coastline.")
    parser.add_argument("-subdivision_input_file", "-s", 
                         help="Path to file area polygons (subdivision of domain).")
    parser.add_argument("-scale_input_file", "-i",
                         help="Path to file with scale input.")                          
    parser.add_argument("-coastline_output_file", "-o", 
                         help="Where to store the output file (path+filename).")
    args = parser.parse_args()

   
    return args.coastline_input_file, \
           args.subdivision_input_file, \
           args.scale_input_file, \
           args.coastline_output_file


def crop_coast(limits, coast, buf=10000):
    '''Find the intersection of the polygon defined by the bounds of limits (input 1) and the coast (input 2).'''
    domain_limits = [(limits.bounds[0], limits.bounds[1]),
                     (limits.bounds[2], limits.bounds[1]),
                     (limits.bounds[2], limits.bounds[3]),
                     (limits.bounds[0], limits.bounds[3]),
                     (limits.bounds[0], limits.bounds[1])]
    
    domain_limits = Polygon(domain_limits)
    
    
    for i, c in enumerate(coast):
        if np.mod(i, 1000) == 0:
            print(i)
           
        if i == 0: # mainland contour
           ci = c.intersection(domain_limits.buffer(10000))
           cropped_coast = [p for p in ci]
        else:
           if c.intersects(domain_limits.buffer(10000)):
               cropped_coast.append(c)
               
    cr = MultiPolygon(cropped_coast) 
    return cr




def plot_coast(coast, color='k', p=False):
   '''Plot coast line'''
   fig, ax = plt.subplots()
   ax.set_aspect('equal')
   if  type(coast) == list:
       if p:
            coast = coast[p]
       
       coast = unary_union(coast)
   
   if  coast.type== 'Polygon':
       x, y = coast.exterior.coords.xy
       ax.plot(x, y, color)
   elif coast.type == 'MultiPolygon':
       for n, c in enumerate(coast):
           if np.mod(n, 1000) == 0:   
               print(n)
           x, y = c.exterior.coords.xy
           ax.plot(x, y, color)

   plt.show()

    
   
def coast2np(coast):
    '''Extract all the x,y values from a MultiPolygon and put them into a numpy array'''
    x = np.empty(0)
    y = np.empty(0)
    for n, c in enumerate(coast):
       if np.mod(n, 1000) == 0:
           print(n)
       xi, yi = c.exterior.coords.xy
       x = np.append(x, xi)
       x = np.append(x, np.nan)
       y = np.append(y, yi)
       y = np.append(y, np.nan)
    
    npcoast = np.array([x, y])
    return npcoast


def read_poly_data(file):
    '''Read coast line data from text file.

    File format: First line is name, second line number of polygons,
    then comes all the polygon data. First line of polygon data is
    number of vertices, then follows the x, y coordinates for each
    vertex.
    
    Output: A shapely MultiPolygon object containing the polygon data
    '''
    print("Reading polygons from: " + file)
    with open(file) as f:
        data = f.readlines()
    data = list(map(str.rstrip, data))
    name = data[0]
    n_poly = int(data[1])
    p_data = list(map(str.split, data[2:]))
    poly_list = np.array(np.zeros(n_poly), dtype=object)
    i = 0
    for n in range(n_poly):
        n_points = int(p_data[i][0])
        i += 1
        poly = p_data[i:i+n_points]
        poly_list[n] = np.array(poly, dtype=float)
        i += n_points
    print("Polygons read: " + str(n_poly))
    return MultiPolygon(map(Polygon, poly_list))




def write_poly_data(file, mp):
    '''Write coast line data to text file.

    File format: First line is name, second line number of polygons,
    then comes all the polygon data. First line of polygon data is
    number of vertices, then follows the x, y coordinates for each
    vertex.
    
    Input: A shapely MultiPolygon object containing the polygon data
    '''
    with open(file, 'w') as f:
        print("Writing polygons to: " + file)
        f.write('COAST\n')
        f.write('%d\n' % len(mp))
        for poly in mp:
            x, y = poly.exterior.xy
            n = len(x)
            f.write('%d 1\n' % n)
            for i in range(n):
                f.write('%.4f %.4f\n' % (x[i],y[i]))
        print("Polygons written: " + str(len(mp)))



def process_function_simple(mpoly, scale):
    '''Simplest possible function that does island merging and removal.

    Major downside is that the final buffer contraction may split off
    polygons that are separated by a distance smaller than scale.

    First buffer operation can also be done faster
    '''
    mp = mpoly
    # Expand polygons: this will merge polygons separated on this scale
    mp = mp.buffer(scale)
    # Contract back to original size
    mp = mp.buffer(-scale)
    # Contract polygons: this will remove polygons smaller than scale
    # May also split polygons with thin structures into several polygons
    mp = mp.buffer(-scale)
    # Expand back to original size
    mp = mp.buffer(scale)
    # Make sure we always return a multipolygon
    if type(mp) == Polygon:
        mp = MultiPolygon([mp])
    return mp



# Processesing function
# At the moment only one scale parameter is used
def process_function(mpoly, scale):
    '''Perform buffer operations on MultiPolygon object.

    Will first merge polygons separated by scale and then delete
    remaining polygons smaller than scale.

    The resulting polygons are also smooth on scale.

    There can still be polygons separeted by distance smaller than scale.
    '''
    #mp = mpoly.simplify(0.1*scale)
    mp = mpoly
    # Expand polygons: this will merge polygons separated on this scale
    # faster to expand individual polygons and then merge
    if mp.type == 'MultiPolygon':
        mp = unary_union([p.buffer(scale) for p in mp])
    else:
        mp = mp.buffer(scale)

    # Contract back to original size
    mp = mp.buffer(-scale)
    # Contract polygons: this will remove polygons smaller than scale
    # May also split polygons with thin structures into several polygons
    # The latter split off is probably unwanted?
    mp = mp.buffer(-scale)
    # Expand back to original size
    mp = mp.buffer(scale)
    # Expand again to merge back polygons that got split off
    # Still a few pathological cases remains were separation is smaller than scale.
    mp = mp.buffer(scale)
    mp = mp.buffer(-scale)
    # Buffer operations introduces extra points that are not needed.                        
    # Simplify tries to use minimum number of points for a given tolerance.                 
    mp = mp.simplify(0.1*scale)
    # Make sure we always return a multipolygon                                             
    if type(mp) == Polygon:
        mp = MultiPolygon([mp])
    return mp


def make_coast(mp_coast, mp_sub, scale_in, coastline_out):
    '''
    Make new coastline based on input coastline, subivision polygons and scale factors.
    '''
    # Remove unnecessary coastline
    # ---------------------------------------------------------------------
    print("\nRead subdivision scales from: %s\n" % scale_in)
    scales    = np.loadtxt(scale_in,dtype=str)

    # The input file can have names for each subpolygon, or it can have pure
    # numbers. Better solution for people with too much on their mind to care
    # about memorizing which polygon belongs where =)
    if len(scales[0,:])==3:
        scales = scales[:,1:].astype(float)
    else:
        scales = scales.astype(float)

    # simplify the coast line
    # ---------------------------------------------------------------------
    print('simplify the coastline')
    scale_min = np.min(scales)
    mp_coast  = mp_coast.simplify(0.1*scale_min)

    # Process all polygons according to scale within each subdivision
    # ---------------------------------------------------------------------
    mp_list = list()
    for i, p_sub in enumerate(mp_sub):
        print("Processing subdivision: %d" % i)
        scale1, scale2 = scales[i]
        print("Scale: %d" % scale1)
        if i > -1:
            intersect = []
            for j, c in enumerate(mp_coast):
                #if np.mod(j, 1000) == 0:
                #    print(j)
                intersect.append(c.intersection(p_sub.buffer(scale2)))
            mpi = unary_union(intersect)
        else:
            mpi = mp_coast.intersection(p_sub.buffer(scale2))


        # add buffer to subdivision polygons to ensure they are overlapping
        try:
            print("Polygons before: %d" % len(mpi))
            mpi_scale1 = process_function(mpi, scale1)

            print("Polygons after: %d\n" % len(mpi_scale1))
            mp_list.append(mpi_scale1)

        except:
            print("This subdivision must cover an area either\n"+\
                  "completely covered by land, by the ocean or\n"+\
                  "just covered by one polygon\n")
            mpi_scale1 = process_function(mpi, scale1)
            mp_list.append(mpi_scale1)

    # Finally join all the processed polygons
    mp_join = unary_union(mp_list)

    if mp_join.geom_type == 'Polygon':
        mp_join = MultiPolygon([mp_join])

    print("Total number of polygons after processing: %d\n" % len(mp_join))
    print("Write coast line data")
    write_poly_data(coastline_out, mp_join)

    # plot polygons on top of complete coastline
    plot_divisions(mp_coast,mp_join,mp_sub)
    
    # plot polygons on top of smoothed coastline
    #plot_divisions(mp_join,mp_sub,title='after smoothing')

    return mp_list, mp_coast, mp_sub


def make_coast_full(coastline_in, subdivision_in, scale_in, coastline_out, cropcoast = True):
    '''
    Make new coastline based on input coastline, subivision polygons and scale factors.
    '''

    print("Read coast line data")
    mp_coast = read_poly_data(coastline_in)

    print("Read subdivision polygons")
    mp_sub   = read_poly_data(subdivision_in)

    # Remove unnecessary coastline
    # ---------------------------------------------------------------------
    if cropcoast:
        mp_coast = crop_coast(mp_sub, mp_coast)
        print(len(mp_coast))

    print("Read subdivision scales from: %s\n" % scale_in)
    scales    = np.loadtxt(scale_in,dtype=str)

    # The input file can have names for each subpolygon, or it can have pure
    # numbers. Better solution for people with too much on their mind to care
    # about memorizing which polygon belongs where =)
    # ---------------------------------------------------------------------
    if len(scales[0,:])==3:
        scales = scales[:,1:].astype(float)
    else:
        scales = scales.astype(float)

    # simplify coast line
    # ---------------------------------------------------------------------
    scale_min = np.min(scales)
    mp_coast = mp_coast.simplify(0.1*scale_min)

    # Process all polygons according to scale within each subdivision
    # ---------------------------------------------------------------------
    mp_list = list()
    for i, p_sub in enumerate(mp_sub):
        print("Processing subdivision: %d" % i)
        scale1, scale2 = scales[i]
        print("Scale: %d" % scale1)
        if i > -1:
            intersect = []
            for j, c in enumerate(mp_coast):
                #if np.mod(j, 1000) == 0:
                #    print(j)
                intersect.append(c.intersection(p_sub.buffer(scale2)))
            mpi = unary_union(intersect)
        else:
            mpi = mp_coast.intersection(p_sub.buffer(scale2))


        # add buffer to subdivision polygons to ensure they are overlapping
        try:
            print("Polygons before: %d" % len(mpi))
            mpi_scale1 = process_function(mpi, scale1)
            print("Polygons after: %d\n" % len(mpi_scale1))
            mp_list.append(mpi_scale1)

        # Espenes 31/01/2020
        except:
            print("Polygons before: %d" % 0)
            print("This subdivision must cover an area either\n"+\
                  "completely covered by land or completely by the ocean\n")
            mpi_scale1 = process_function(mpi, scale1)
            mp_list.append(mpi_scale1)

    # Finally join all the processed polygons
    mp_join = unary_union(mp_list)
    print(type(mp_join))
    if mp_join.geom_type == 'Polygon':
        mp_join = MultiPolygon([mp_join])
    print("Total number of polygons after processing: %d\n" % len(mp_join))
    print("Write coast line data")
    write_poly_data(coastline_out, mp_join)

    # plot polygons on top of complete coastline
    plot_divisions(mp_coast,mp_join,mp_sub)
    
    # plot polygons on top of smoothed coastline
    #plot_divisions(mp_join,mp_sub,title='after smoothing')

    return mp_list, mp_coast, mp_sub


def make_coast_iteration(mp_list, polygons, poly_number, scale1, scale2, mp_coast, outfile):
    '''Make new coastline for a subsection of the domain'''
    
    if type(poly_number) == list:
        for pn, s1, s2 in zip(poly_number, scale1, scale2):
            poly = polygons[pn]
            
            intersect = []
	    
            for j, c in enumerate(mp_coast):
                intersect.append(c.intersection(poly.buffer(s2)))

            mpi = unary_union(intersect)
            mpi_scale1 = process_function(mpi, s1)
            mp_list[pn] = mpi_scale1

    else:
        poly = polygons[poly_number]
        intersect = []
    
        for j, c in enumerate(mp_coast):
            intersect.append(c.intersection(poly.buffer(scale2)))
    
        mpi = unary_union(intersect)
        mpi_scale1 = process_function(mpi, scale1)
        mp_list[poly_number] = mpi_scale1
    
    mp_join = unary_union(mp_list)
    write_poly_data(outfile, mp_join)

    return mp_list

def plot_divisions(mp_coast, mp_join, mp_sub, title=' '):
    '''
    Draw the coastline near the subdivisions, and plot
    the divisions on top of that.
    '''

    # Collect mainland and islands in one vector
    # ---------------------------------------------------------------------
    print('Plotting the smoothed coastline on top of the original one...')
    plt.figure()
    i = 0
    # Add a feature that stores this, no use in doing this many times...
    try:
        raw_coast = np.load('coast.npy', allow_pickle = True).item()

    except:
        for p in mp_coast:
            xt,yt=p.exterior.xy                                
            if i == 0:
                x = xt       
                y = yt
            else:      
                x = np.append(x,xt)
                y = np.append(y,yt)

            # These nans will make plots separated
            x = np.append(x,np.nan)
            y = np.append(y,np.nan)
            i+=1
        raw_coast = {}
        raw_coast['x'] = x
        raw_coast['y'] = y
        np.save('coast.npy', raw_coast)
        
    plt.plot(raw_coast['x'], raw_coast['y'],label='before smoothing')

    i = 0
    for p in mp_join:
        xt,yt=p.exterior.xy                                
        if i == 0:
            x2 = xt       
            y2 = yt
        else:      
            x2 = np.append(x2,xt)
            y2 = np.append(y2,yt)

        # These nans will make plots separated
        x2 = np.append(x2,np.nan)
        y2 = np.append(y2,np.nan)
        i+=1
        smoothed = {}
        smoothed['x'] = x2
        smoothed['y'] = y2
        np.save('smoothed.npy', smoothed)
        
    plt.plot(x2,y2,label='after smoothing')

    # Plot subdivisions
    # ---------------------------------------------------------------------
    for i,p in enumerate(mp_sub):
        x,y = p.exterior.xy
        plt.fill(x,y,alpha=0.3)

    plt.legend()
    plt.title(title)
    plt.axis('equal')
        
if __name__ == '__main__':
    coastline_in, subdivision_in, scale_in, coastline_out = parse_command_line()
    make_coast_full(coastline_in, subdivision_in, scale_in, coastline_out)
