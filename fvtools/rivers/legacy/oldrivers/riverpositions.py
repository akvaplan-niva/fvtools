import numpy as np
from scipy.io import loadmat

'''
These routines were written to process river poition files that we originally got from the NVE.
These files were in .mat format, and of unknown origin. Some rivers were missing.

We have since developed routines to identify the river outlet from the Elvis database.
'''

class BaseRiver:
    def add_parameters(self, names):
        '''
        Read grid attributes from mfile and add them to FVCOM_grid object
        '''
        rivers = loadmat(self.pathToRiver)
        if type(names) is str:
            names=[names]
        for name in names:
            setattr(self, name, rivers[name])

    def crop_to_vassdrag(self):
        """
        Removes:
        - Rivers outside of the chosen vassdrag
        - Rivers too close to the OBC
        """
        self = crop_object(self, self.rivers_in_vassdrag)

class LargeRivers(BaseRiver):
    """
    Loads river data, at the moment just from mat files, but in the future?
    """
    def __init__(self, info):
        """
        Import large rivers
        """
        self.pathToRiver = info['LargeRivers']
        print('- '+self.pathToRiver)
        if self.pathToRiver[-3:] == 'mat':
            self.add_parameters(['areal','landareal','name','nedborfelt','totalareal','Vl','x','y'])
        else:
            raise NameError(f'.{self.pathToRiver.split(".")[-1]} files are not supported')

    def connect_nedborsfelt(self, vassdrag_tuple):
        """
        Big rivers (nedbørsfelt til hav)
        """
        self.rivers_in_vassdrag  = np.array([ind for ind, i in enumerate(self.Vl) if i in vassdrag_tuple]).astype(int)

    def add_temperature(self, Temp):
        """
        Connect specific rivres to vassdrag, and mean river to the rest of the domain

        Future:
        - Investigate whether rivers can obtain temperatures as a function of distance
          from nearest temperature measurement
        """
        self.river_temp = np.zeros((len(Temp.average_temp), len(self.Vl)))
        for i, vassdrag in enumerate(self.Vl):
            if vassdrag in Temp.vassdrag:
                river = np.where(np.array(Temp.vassdrag) == vassdrag)[0][0]
                self.river_temp[:,i] = Temp.river_temp[:,river] # Ps. this method will be flawed if more than 1 temperature measurement in vassdrag
            else:
                self.river_temp[:,i] = Temp.average_temp
        self.river_time = Temp.river_time

    def get_area_fraction(self):
        """
        To get a reasonable estimate of the total runoff that goes through the main river
        """
        self.Vfrac = self.areal/self.landareal

class SmallRivers(BaseRiver):
    """
    Handle data from small rivers
    """
    def __init__(self, info):
        """
        import small rivers
        """
        self.pathToRiver = info['SmallRivers']
        print(f'- {self.pathToRiver}')
        self.add_parameters(['riv_ids','Vs','x2','y2'])
        self.x = self.__dict__.pop('x2')
        self.y = self.__dict__.pop('y2')

    def add_temperature(self, Temp):
        """
        Set all small rivers to equal the average-temperature.
        We may look into finding better ways to connect small rivers to temperatures
        in the future.
        """
        self.river_temp = np.zeros((len(Temp.average_temp), len(self.Vs)))
        for i, vassdrag in enumerate(self.Vs):
            self.river_temp[:,i] = Temp.average_temp
        self.river_time = Temp.river_time

    def connect_nedborsfelt(self, vassdrag_tuple):
        """
        Big rivers (nedbørsfelt til hav)
        """
        self.rivers_in_vassdrag = np.array([ind for ind, i in enumerate(self.Vs) if i in vassdrag_tuple]).astype(int)

def crop_object(obj, indices):
    keys = obj.__dict__.keys()
    for key in keys:
        var = getattr(obj,key)
        if key == 'rivers_in_vassdrag':
            continue
        if type(var) == str:
            continue
        setattr(obj, key, var[indices])
    return obj