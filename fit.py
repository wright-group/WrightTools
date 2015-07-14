import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy.optimize import leastsq

### functions to fit against ##################################################

def gaussian(x, p):
    '''
    p = [peak amplitude, center, sigma]
    '''
    a, mu, sigma = p
    return a*np.exp(-(x-mu)**2 / (2*np.abs(sigma)**2))
    
def lorentz(x, p):
    '''
    p  = [peak amplitude, center, FWHM]
    '''
    a, center, FWHM = p
    x_var = (center - x) / (0.5*FWHM)
    return a/(1+x_var**2)

### methods ###################################################################


