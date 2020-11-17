import numpy as np
from scipy.interpolate import interp1d
from panda import read_csv
import random

class irradiance():
    def __init__(self, path, length=None, offset=0, row = 1, interpolate = 'cubic'):
        self.offset = offset
        self.length = length
        self.row = row
        data = read_csv(self.path, header = 0, usecols = [self.row+1], skiprows = offset, nrows=length+1)
        data1 = data.to_numpy().reshape(-1)
        idx = np.array(range(len(data1)))
        self.data = interp1d(idx,data1, kind= interpolate)

    def __len__(self):
        return self.length
    
    def __getitem__(self, time):
        if time>length:
            raise ValueError('Irradiance time out of range!')
        return self.data(time)

    def __call__(self, time):
        if time>length:
            raise ValueError('Irradiance time out of range!')
        return self.data(time)

    def get_ird(self, time):
        if time>length:
            raise ValueError('Irradiance time out of range!')
        return self.data(time)

    def get_length(self):
        retrn self.length