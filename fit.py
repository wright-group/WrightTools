### import ####################################################################


import os
import collections

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import scipy


### functions to fit against ##################################################


def exponential(p, x):
    '''
    p = [amplitude, decay constant, offset]
    '''
    a, b, c = p
    return a*np.exp(-x/b)+c


def gaussian(p, x):
    '''
    p = [amplitude, center, FWHM, offset]
    '''
    print p
    a, b, c, d = p
    return a*np.exp(-(x-b)**2/(2*np.abs(c/(2*np.sqrt(2*np.log(2))))**2))+d

    
def lorentzian(p, x):
    '''
    p  = [amplitude, center, FWHM, offest]
    '''
    a, center, FWHM, offset = p
    x_var = (center - x) / (0.5*FWHM)
    return (a/(1+x_var**2))+offset


### classes ###################################################################


def error(p, x, y):
    A = gaussian(p[:4], x)
    B = gaussian(p[4:], x)
    return y - A + B

class LeastSquares:
    
    def __init__(self, function):
        self.optimize = scipy.optimize.leastsq
        self.error = error
        
    def _do(self, arr):
        pass
    
    def give_data(self, data, *args):
        self.data = data
        self.dimension_args = args
        self.chopped = self.data.chop(*args, verbose=False)
        print 'LeastSquares recieved data to make %d fits'%len(self.chopped)
        
    def give_guesses(self):
        pass
    
    def run(self, channel=0):
        # create output objects -----------------------------------------------
        outs_dimensions = self.data.axis_names
        at = collections.OrderedDict()
        for arg in self.dimension_args:
            axis = getattr(self.data, arg)
            at[arg] = [axis.points[0], axis.units]
            outs_dimensions.remove(arg)
            outs_dimensions
        args = tuple(outs_dimensions) + tuple([at])
        self.outs = self.data.chop(*args)[0]
        self.model = self.data.copy()
        # do all fitting operations -------------------------------------------
        for data in self.chopped:
            p0A = [0.5, 14400, 250, 0.]
            p0B = [0.5, 15700, 300, 0.]
            p0 = [p0A, p0B]
            xi = data.axes[0].points
            zi = data.channels[channel].values
            # p0 gets flattened?
            #out = self.optimize(self.error, p0, args=(xi, zi))
        return [self.model, self.outs]


### testing ###################################################################


if __name__ == '__main__':
    
    xi = np.linspace(-10, 10, 100)
    p = [1, 1, 2, 0]
    zi = gaussian(p, xi)
    
    plt.plot(xi, zi)
    plt.axvline(0)
    plt.axhline(0.5)
    
