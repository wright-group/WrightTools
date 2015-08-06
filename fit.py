import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy.optimize import leastsq

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
    a, b, c, d = p
    return a*np.exp(-(x-b)**2/(2*np.abs(c/(2*np.sqrt(2*np.log(2))))**2))+d
    
def lorentzian(p, x):
    '''
    p  = [amplitude, center, FWHM, offest]
    '''
    a, center, FWHM, offset = p
    x_var = (center - x) / (0.5*FWHM)
    return (a/(1+x_var**2))+offset

### methods ###################################################################


### testing ###################################################################


if __name__ == '__main__':
    
    xi = np.linspace(-10, 10, 100)
    p = [1, 1, 2, 0]
    zi = gaussian(p, xi)
    
    plt.plot(xi, zi)
    plt.axvline(0)
    plt.axhline(0.5)
    
