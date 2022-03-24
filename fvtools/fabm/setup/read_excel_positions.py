import xlrd
import numpy as np

def load_positions(positions_file):
    xl_scen = xlrd.open_workbook(positions_file)
    xl_sheet = xl_scen.sheet_by_index(0)
    
    # Loop to find all locations
    position = []
    location = []
    read_position = False
    read_position_last = False
    i = 0
    j = 1

    # Dummy array
    store_here = np.ndarray((1,200), dtype = object)
    while True:
        if read_position:
            read_position_last = True
            
        # cycle rows
        row, kill = read_row(i, xl_sheet)
        if kill:
            store_here[0,j-1] = np.array([position], dtype = float)
            location.append(position)
            break

        #if lat lon in deg decimal min (DDM) format, as in the cursed olex
        if len(row[1].value.split('.')) == 3:
            read_position = True
            lat = row[1].value
            lon = row[2].value
            pos = sec2frac(lat,lon)
            position.append(pos)
        
        #mad added 'else if' in case already in decimal form
        elif len(row[1].value.split('.')) == 2:
            read_positioin = True
            lat = row[1].value
            lon = row[2].value
            position.append((lat,lon))

        else:
            read_position = False

        if read_position == False:
            if read_position_last:
                store_here[0,j-1] = np.array([position], dtype = float)
                location.append(position)
                position = []
                j+=1
                read_position_last = False
        i+=1

    return store_here[:,:j]
    
def read_row(i, xl_sheet):
    kill = False
    row = []
    try:
        row = xl_sheet.row(i)
    except IndexError:
        kill = True
    return row, kill


def sec2frac(N,E):
    '''
    from "70.11.19" to "70.1886111" format
    '''
    deg  = N.split('.')[0]
    mins = N.split('.')[1]+'.'+N.split('.')[2]
    Nnew = float(deg)+float(mins)/60.0

    deg  = E.split('.')[0]
    mins = E.split('.')[1]+'.'+E.split('.')[2]
    Enew = float(deg)+float(mins)/60.0

    return Nnew, Enew
