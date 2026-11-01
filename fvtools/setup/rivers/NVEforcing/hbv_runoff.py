# Reads runoff data from NVEs HBV model.
# We've received such data from James Edward Sample at NIVA, who again receives them from NVE.
import os
from datetime import datetime, timezone
import pandas as pd

class HBVRunoff:
    def __init__(self, hbvfolder):
        '''
        Assumes that all .var files in the hbv folder are runoff data.
        '''
        # Load data for the full country
        data = []
        for i, file in enumerate(os.listdir(hbvfolder)):
            # Figure out which number this vassdragsområde is
            vassdragsomraade = int(file.split('.var')[0][-3:])
        
            # Load the data
            _data = pd.read_csv(f'{hbvfolder}{file}', header = None, sep=r'\s+')
        
            # Load the dates, will have the same coverage in all files
            if i == 0:
                dates = [datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[9:11]), tzinfo = timezone.utc) for d in _data[0]]
        
            data.append(_data.rename(columns = {1: f"runoff_{vassdragsomraade}"}).drop(columns=0))
        
        # Re-scale vassdragsområde 183 (python index 182), as suggested by James Sample at NIVA
        data[182] = 6/16 * data[181]
        self.transport = pd.concat(data, axis = 1)
        self.dates = dates