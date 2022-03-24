import xlrd
import numpy as np

def read_carbon(shit_file):
    """
    Finds the tracers, stores their numbers. Reads the first row in the carbon file.
    """
    xl_scen   = xlrd.open_workbook(shit_file)
    xl_sheet  = xl_scen.sheet_by_index(0)

    i = 0
    injection_index = []
    injection_name  = []
    injection       = []
    find_ids = True
    while True:
        row, kill = read_row(i, xl_sheet)
        if kill:
            break

        # We want to identify the rows with kg in them, and associate the numbers with a release
        if find_ids:
            for j in range(len(row)):
                if 'kg' in row[j].value:
                    injection_index.append(j)
                    injection_name.append('injection'+row[j].value.split('T')[-1])
                    
        # It is sufficient to read one row
        if find_ids == False:
            for place in injection_index:
                injection.append(float(row[place].value))
            break
        
        if any(injection_name):
            find_ids = False
        i+=1

    # Prepare the input for the receiver
    carbon = {}
    carbon['names']   = injection_name
    carbon['release'] = np.array(injection)
    return carbon

def read_row(i, xl_sheet):
    kill = False
    row = []
    try:
        row = xl_sheet.row(i)
    except IndexError:
        kill = True
    return row, kill
