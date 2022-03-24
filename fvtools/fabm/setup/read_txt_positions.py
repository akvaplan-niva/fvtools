import numpy as np

def load_positions(positions_file):
    """
    Load center coordinates from a .txt file
    - Will only support 1 farm at the time
    """
    store_here = np.ndarray((1,100), dtype = object)
    
    # Read the data
    # ----
    lon_cage = []; lat_cage = []
    j = 0
    read_position      = False
    read_position_last = False
    with open(positions_file) as f:
        lines = f.readlines()
   
    for line in lines:
        # Let the routine remember that it is building a position vector
        # ----
        if read_position:
            read_position_last = True

        # See if this line is also a position
        # ----
        pos = line[:-1].split(' ')
        if len(pos) != 2:
            read_position = False
    
        else:
            lon_cage.append(float(pos[0]))
            lat_cage.append(float(pos[1]))
            read_position = True

        # Store as a cage if we already have numbers in the vectors
        # ----
        if read_position == False:
            if read_position_last:
                position = []
                for lat, lon in zip(lat_cage, lon_cage):
                    position.append([lat, lon])

                store_here[0,j] = np.array([position], dtype = np.float32)
                j += 1
                read_position_last = False
                lat_cage = []; lon_cage = []

    # Store the last cage we were building
    # ----
    position = []
    for lat, lon in zip(lat_cage, lon_cage):
        position.append([lat, lon])

    store_here[0,j] = np.array([position], dtype = np.float32)
    j += 1
    read_position_last = False

    return store_here[:,:j]


# old routine
                
#    try:
#        points = np.loadtxt(positions_file, skiprows = 1, delimiter=' ')
#    except:
#        points = np.loadtxt(positions_file, skiprows = 1, delimiter='\t')

    # Assume lon, lat input
    # ----
#    lat = points[:,1]
#    lon = points[:,0]

    # Dump on same format as the excel sheets
    # ----
#    position = []
#    for i in range(len(lat)):
#        position.append([lat[i], lon[i]])

#    store_here[0,0] = np.array([position], dtype = np.float32)
