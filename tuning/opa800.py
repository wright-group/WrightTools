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
from .. import artists as wt_artists
from .. import data as wt_data
from .. import fit as wt_fit
from .. import kit as wt_kit
from .. import units as wt_units
cmap = wt_artists.colormaps['default']

### processing methods ########################################################


def process_motortune(filepath, channel_name, old_curve_filepath, autosave=True,
                      cutoff_factor=50, output_points_count=25):
    channel_name = channel_name
    # make data object
    data = wt_data.from_PyCMDS(filepath, verbose=False)
    data.convert('wn', verbose=False)
    # get channel index
    channel_index = data.channel_names.index(channel_name)
    # check if data is compatible
    if not len(data.axes) == 2:
        print 'data must be 2 dimensional'
        return
    # transpose into prefered representation (motors, tune points)
    if len(data.axes[0].name) < len(data.axes[1].name):
        data.transpose(verbose=False)
    tune_points = data.axes[1].points
    # process data
    data.level(channel_index, 0, -3)
    # get centers through expectation value
    motor_axis_name = data.axes[0].name
    function = wt_fit.ExpectationValue()
    function.global_cutoff = data.channels[channel_index].zmax / cutoff_factor
    fitter = wt_fit.Fitter(function, data, motor_axis_name, verbose=False)
    outs = fitter.run(channel_index, verbose=False)
    offsets = outs.value.values
    # make curve
    old_curve = wt_curve.from_800_curve(old_curve_filepath)
    old_curve.map_colors(tune_points, 'wn')
    motors = []
    for motor_index, motor_name in enumerate([m.name for m in old_curve.motors]):
        if motor_name == motor_axis_name.split('_')[-1]:
            positions = data.axes[0].centers + offsets
            motor = wt_curve.Motor(positions, motor_name)
            motors.append(motor)
            tuned_motor_index = motor_index
        else:
            motors.append(old_curve.motors[motor_index])
    curve = wt_curve.Curve(tune_points, 'wn', motors, 
                           name=old_curve.name.split('-')[0],
                           kind='opa800', interaction='DFG', 
                           method=wt_curve.Poly)
    curve.map_colors(output_points_count)
    old_curve.map_colors(curve.colors)
    # plot data
    artist = wt_artists.mpl_2D(data)
    artist.onplot(tune_points, offsets)
    artist.onplot(curve.colors, curve.motors[tuned_motor_index].positions-old_curve.motors[tuned_motor_index].positions, alpha=1)
    artist.plot(channel_index, autosave=True,
                contours=0,
                fname=filepath.replace('.data', ''))
    # plot curve
    curve.save(save_directory=wt_kit.filename_parse(filepath)[0])


def process_tunetest(filepath, channel, autosave=True):
    # recognize kind of scan
    start = filepath.index('[') + 1
    end = filepath.index(']')
    dims = filepath[start:end].split(',')
    opa_name = dims[0].strip()
    print 'opa recognized as', opa_name
    # import array
    headers = get_headers(filepath)
    arr = np.genfromtxt(filepath).T
    opa_col = headers['name'].index(opa_name)
    mono_col = headers['name'].index('wm')
    detector_col = headers['name'].index(channel)
    opa_points = arr[opa_col]
    mono_points = arr[mono_col]
    detector_points = arr[detector_col]
    # shape array
    tunepoints = wt_kit.unique(opa_points, tolerance=1)
    xi = tunepoints
    mono_points.shape = (len(tunepoints), -1)
    mono_points = mono_points.T
    mono_points = wt_units.converter(mono_points, 'nm', 'wn')
    delta_mono = (mono_points[:, 0].max() - mono_points[:, 0].min())/2.
    yi = np.linspace(delta_mono, -delta_mono, len(mono_points))
    detector_points.shape = (len(tunepoints), -1)
    detector_points = detector_points.T
    zi = detector_points
    # plot raw_data
    fig = plt.figure(figsize=[8, 6])
    X, Y, Z = wt_artists.pcolor_helper(xi, yi, zi)
    plt.pcolor(X, Y, Z, cmap=cmap)
    plt.xlim(xi.min(), xi.max())
    plt.ylim(yi.min(), yi.max())
    plt.grid()
    plt.xlabel(opa_name)
    plt.ylabel('$\Delta$ (wn)')
    filename = wt_kit.filename_parse(filepath)[1]
    plt.suptitle(filename)
    # process motortune
    m_chosen = np.zeros(len(tunepoints))
    m_old = np.zeros(len(tunepoints))
    for i in range(len(tunepoints)):
        ms = mono_points[:, i]
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
    plt.plot(tunepoints, m_chosen-m_old, lw=5, c='k', alpha=0.5)
    # generate tuning curve
    motors = []
    for name in ['Grating', 'BBO', 'Mixer']:
        motor_name = '_'.join([opa_name, name])
        motor_col = headers['name'].index(motor_name)
        motor_points = arr[motor_col]
        motor_points.shape = (len(tunepoints), -1)
        positions = motor_points[:, 0]
        motor = wt_curve.Motor(positions, name)
        motors.append(motor)
    curve_name = 'OPA' + opa_name[-1] + ' '
    curve = wt_curve.Curve(m_chosen, 'wn', motors, curve_name, 'opa800', 
                           interaction='DFG', method=wt_curve.Poly)
    # map points
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