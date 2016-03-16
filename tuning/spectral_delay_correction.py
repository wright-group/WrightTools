'''
Tools for processing spectral delay correction data.
'''


### import ####################################################################


import os
import re
import sys
import imp
import time
import copy
import inspect
import itertools
import subprocess
import ConfigParser
import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

import numpy as np
                
import scipy
from scipy.optimize import leastsq
from scipy.interpolate import griddata, interp1d, interp2d, UnivariateSpline

from .. import artists as wt_artists
from .. import units as wt_units
from .. import kit as wt_kit
from .. import fit as wt_fit
from .. import data as wt_data
import curve as wt_curve


### processing methods ########################################################


def process_wigner(data_filepath, channel):
    '''
    Create a coset file from a measured wigner.
    
    Parameters
    ----------
    data_filepath : str
        Filepath to data file.
    channel : int or str
        The channel to process.
    '''
    # get data
    data = wt_data.from_PyCMDS(data_filepath, verbose=False)
    if data.axes[0].units_kind == 'energy':
        data.transpose()  # prefered shape - delay then color
    ws = data.axes[1].points
    # get channel index
    if type(channel) in [int, float]:
        channel_index = int(channel)
    elif type(channel) in [str]:
        channel_index = data.channel_names.index(channel)
    else:
        print 'channel type not recognized'
        return
    # process
    function = wt_fit.Gaussian()
    fitter = wt_fit.Fitter(function, data, data.axes[0].name)
    outs = fitter.run()
    # spline
    spline = UnivariateSpline(ws, outs.channels[0].values, k=2, s=1000)
    corrections = spline(ws)
    # prepare for plot
    artist = wt_artists.mpl_2D(data)
    # plot outs
    xi = outs.axes[0].points
    yi = outs.channels[0].values
    artist.onplot(xi, yi)
    # plot corrections
    xi = ws
    yi = corrections
    artist.onplot(xi, yi, alpha=1)
    # finish plot
    artist.plot(channel_index, contours=0, lines=False)
    return outs

