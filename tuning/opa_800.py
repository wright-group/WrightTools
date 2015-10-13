'''
Methods for processing OPA 800 tuning data.
'''


### imports ###################################################################


import os
import re
import sys
import imp
import ast
import time
import copy
import inspect
import collections
import subprocess
import ConfigParser
import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as grd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from numpy import sin, cos
                
import scipy
from scipy.interpolate import griddata, interp1d, interp2d, UnivariateSpline
import scipy.integrate as integrate
from scipy.optimize import leastsq

from pylab import *

import curve as wt_curve
from .. import kit as wt_kit
from .. import artists as wt_artists
cmap = wt_artists.colormaps['default']


### helper methods ############################################################


def get_headers(filepath):
    headers = collections.OrderedDict()
    for line in open(filepath):
        if line[0] == '#':
            split = line.split(':')
            key = split[0][2:]
            item = split[1].split('\t')
            if item[0] == '':
                item = [item[1]]
            item = [i.strip() for i in item]  # remove dumb things
            item = [ast.literal_eval(i) for i in item]
            if len(item) == 1:
                item = item[0]
            headers[key] = item
        else:
            # all header lines are at the beginning
            break
    return headers


def expectation_value(y, x):    
    y_internal = np.ma.copy(y)
    x_internal = np.ma.copy(x)
    # get sum
    sum_y = 0.
    for i in range(len(y_internal)):
        if np.ma.getmask(y_internal[i]) == True:
            pass
        elif np.isnan(y_internal[i]):
            pass
        else:
            sum_y = sum_y + y_internal[i]    
    # divide by sum
    for i in range(len(y_internal)):
        if np.ma.getmask(y_internal[i]) == True:
            pass
        elif np.isnan(y_internal[i]):
            pass
        else:
            y_internal[i] = y_internal[i] / sum_y
    # get expectation value    
    value = 0.
    for i in range(len(x_internal)):
        if np.ma.getmask(y_internal[i]) == True:
            pass
        elif np.isnan(y_internal[i]):
            pass
        else:
            value = value + y_internal[i]*x_internal[i]
    return value
    

def gauss_residuals(p, y, x):
    A, mu, sigma = p
    err = y-np.abs(A)*np.exp(-(x-mu)**2 / (2*np.abs(sigma)**2))
    return np.abs(err)


### processing methods ########################################################


def process_motortune(filepath, channel, old_curve_filepath, autosave=True):
    # recognize kind of scan
    start = filepath.index('[') + 1
    end = filepath.index(']')
    dims = filepath[start:end].split(',')
    opa_name = dims[0].strip()
    motor_name = dims[1].strip()
    print 'opa recognized as', opa_name
    print 'motor recognized as', motor_name
    # import array
    headers = get_headers(filepath)
    arr = np.genfromtxt(filepath).T
    opa_col = headers['name'].index(opa_name)
    motor_col = headers['name'].index(motor_name)
    detector_col = headers['name'].index(channel)
    opa_points = arr[opa_col]
    motor_points = arr[motor_col]
    detector_points = arr[detector_col]
    # shape array
    tunepoints = np.unique(opa_points)
    xi = tunepoints
    motor_points.shape = (len(tunepoints), -1)
    motor_points = motor_points.T
    delta_motor = (motor_points[:, 0].max() - motor_points[:, 0].min())/2.
    yi = np.linspace(-delta_motor, delta_motor, len(motor_points))
    detector_points.shape = (len(tunepoints), -1)
    detector_points = detector_points.T
    zi = detector_points
    # plot raw_data
    fig = plt.figure()
    X, Y, Z = wt_artists.pcolor_helper(xi, yi, zi)
    plt.pcolor(X, Y, Z, cmap=cmap)
    plt.xlim(xi.min(), xi.max())
    plt.ylim(yi.min(), yi.max())
    plt.grid()
    plt.xlabel(opa_name)
    plt.ylabel('$\Delta$ ' + motor_name)
    filename = wt_kit.filename_parse(filepath)[1]
    plt.suptitle(filename)
    # process motortune
    m_chosen = np.zeros(len(tunepoints))
    m_old = np.zeros(len(tunepoints))
    for i in range(len(tunepoints)):
        ms = motor_points[:, i]
        m_old[i] = np.average(ms)
        # create masked amplitude array
        amplitude = detector_points.copy()[:, i]
        mask = np.zeros(len(amplitude), dtype=bool)
        for j in range(len(amplitude)):
            if np.abs(amplitude[j]) < amplitude.max()/2.:
                mask[j] = True
        amplitude = np.ma.masked_array(amplitude, mask=mask)
        ms = np.ma.masked_array(ms, mask=mask)
        # find best
        if True:
            m_chosen[i] = expectation_value(amplitude, ms)
        else:
            amplitude_guess = max(amplitude)
            center_guess = m_old[i]
            sigma_guess = 0.25
            p0 = np.array([amplitude_guess, center_guess, sigma_guess])
            outs = leastsq(gauss_residuals, p0, args=(amplitude, ms))[0]
            m_chosen[i] = outs[1]
    # if fail, take old value
    for i in range(len(m_chosen)):
        if m_chosen[i] == 0 or np.isnan(m_chosen[i]):
            m_chosen[i] = m_old[i]
        if not m_old[i]-delta_motor < m_chosen[i] < m_old[i]+delta_motor:
            m_chosen[i] = m_old[i]
    # ensure smoothness with spline
    m_spline = UnivariateSpline(tunepoints, m_chosen, k=2, s=1000)
    m_rough = np.copy(m_chosen)
    for i in range(len(m_chosen)):
        m_chosen[i] = m_spline(tunepoints[i])
    # make plot
    plt.plot(tunepoints, m_rough-m_old, lw=5, c='grey', alpha=0.5)
    plt.plot(tunepoints, m_chosen-m_old, lw=5, c='k', alpha=0.5)
    # generate tuning curve
    old_curve = wt_curve.from_800_curve(old_curve_filepath)
    old_curve.map_colors(tunepoints)
    motors = []
    for name in ['Grating', 'BBO', 'Mixer']:
        if name in motor_name:  # the tuned motor
            motor = wt_curve.Motor(m_chosen, name)
            motors.append(motor)
        else:
            motor = getattr(old_curve, name)
            motors.append(motor)
    curve_name = 'OPA' + opa_name[-1] + ' '
    curve = wt_curve.Curve(tunepoints, 'wn', motors, curve_name, 'opa800', method=wt_curve.Poly)
    curve.map_colors(25)    
    # save
    if autosave:
        out_dir = os.path.dirname(filepath)
        image_path = filepath.replace('.data', '.png')
        plt.savefig(image_path, transparent=True, dpi=300)
        plt.close(fig)
        curve.save(save_directory=out_dir, plot=True)
    else:
        curve.plot()
    # finish
    return curve
