'''
Methods for processing OPA 800 tuning data.
'''


### import ####################################################################


from __future__ import absolute_import, division, print_function, unicode_literals

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
import glob

try:
    import configparser as _ConfigParser  # python 3
except ImportError:
    import ConfigParser as _ConfigParser  # python 2'

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from numpy import sin, cos
                
import scipy
from scipy.interpolate import griddata, interp1d, interp2d, UnivariateSpline
import scipy.integrate as integrate
from scipy.optimize import leastsq

from pylab import *

from . import curve as wt_curve
from .. import artists as wt_artists
from .. import data as wt_data
from .. import fit as wt_fit
from .. import kit as wt_kit
from .. import units as wt_units


### define ####################################################################


cmap = wt_artists.colormaps['default']
cmap.set_bad([0.75]*3, 1.)
cmap.set_under([0.75]*3)

### processing methods ########################################################


def intensity(filepath, channel_name, old_curve_filepath, level=False,
              autosave=True, cutoff_factor=0.1):
    # TODO: documentation
    channel_name = channel_name
    # make data object
    data = wt_data.from_PyCMDS(filepath, verbose=False)
    channel_index = data.channel_names.index(channel_name)
    # check if data is compatible
    if not len(data.axes) == 2:
        # TODO: raise error here
        print('data must be 2 dimensional')
        return
    # transpose into prefered representation (motors, tune points)
    if len(data.axes[0].name) < len(data.axes[1].name):
        data.transpose(verbose=False)
    tune_points = data.axes[1].points
    # process data ------------------------------------------------------------
    if level:
        data.level(channel_index, 0, -3)
    # cutoff
    channel = data.channels[channel_index]
    cutoff = np.nanmax(channel.values)*cutoff_factor
    channel.values[channel.values<cutoff] = np.nan
    # get centers through expectation value
    motor_axis_name = data.axes[0].name
    function = wt_fit.Moments()
    function.subtract_baseline = False
    fitter = wt_fit.Fitter(function, data, motor_axis_name, verbose=False)
    outs = fitter.run(channel_index, verbose=False)
    offsets = outs.one.values
    # pass offsets through spline
    spline = wt_kit.Spline(tune_points, offsets)
    offsets_splined = spline(tune_points)
    # make curve --------------------------------------------------------------
    old_curve = wt_curve.from_800_curve(old_curve_filepath)
    motors = []
    for motor_index, motor_name in enumerate([m.name for m in old_curve.motors]):
        if motor_name == motor_axis_name.split('_')[-1]:
            positions = data.axes[0].centers + offsets_splined
            motor = wt_curve.Motor(positions, motor_name)
            motors.append(motor)
            tuned_motor_index = motor_index
        else:
            motors.append(old_curve.motors[motor_index])
    kind = old_curve.kind
    interaction = old_curve.interaction
    curve = wt_curve.Curve(tune_points, 'wn', motors, 
                           name=old_curve.name.split('-')[0],
                           kind=kind, interaction=interaction)
    curve.map_colors(old_curve.colors)
    # plot --------------------------------------------------------------------
    fig, gs = wt_artists.create_figure(nrows=2, default_aspect=0.5)
    # curves
    ax = plt.subplot(gs[0, 0])
    xi = old_curve.colors
    yi = old_curve.motors[tuned_motor_index].positions
    ax.plot(xi, yi, c='k', lw=1)
    xi = curve.colors
    yi = curve.motors[tuned_motor_index].positions
    ax.plot(xi, yi, c='k', lw=5, alpha=0.5)
    ax.grid()
    ax.set_xlim(tune_points.min(), tune_points.max())
    ax.set_ylabel(curve.motor_names[tuned_motor_index], fontsize=18)
    plt.setp(ax.get_xticklabels(), visible=False)
    # heatmap
    ax = plt.subplot(gs[1, 0])
    xi = data.axes[1].points
    yi = data.axes[0].points
    zi = data.channels[channel_index].values
    X, Y, Z = wt_artists.pcolor_helper(xi, yi, zi)
    ax.pcolor(X, Y, Z, vmin=0, vmax=np.nanmax(zi), cmap=cmap)
    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(yi.min(), yi.max())
    ax.grid()
    ax.axhline(c='k', lw=1)
    xi = curve.colors
    yi = offsets
    ax.plot(xi, yi, c='grey', lw=5, alpha=0.5)
    xi = curve.colors
    yi = offsets_splined
    ax.plot(xi, yi, c='k', lw=5, alpha=0.5)
    units_string = '$\mathsf{(' + wt_units.color_symbols[curve.units] + ')}$'
    ax.set_xlabel(' '.join(['setpoint', units_string]), fontsize=18)
    ax.set_ylabel(' '.join(['$\mathsf{\Delta}$', curve.motor_names[tuned_motor_index]]), fontsize=18)
    # colorbar
    cax = plt.subplot(gs[1, -1])
    label = channel_name
    ticks = np.linspace(0, np.nanmax(zi), 7)
    wt_artists.plot_colorbar(cax=cax, cmap=cmap, label=label, ticks=ticks)
    # finish ------------------------------------------------------------------
    if autosave:
        curve.save(save_directory=wt_kit.filename_parse(filepath)[0])
        p = os.path.join(os.path.dirname(filepath), 'intensity.png')
        wt_artists.savefig(p, fig=fig)
    return curve


def tune_test(filepath, channel_name, old_curve_filepath, level=False,
              autosave=True, cutoff_factor=0.01):
    # TODO: document
    # make data object
    data = wt_data.from_PyCMDS(filepath, verbose=False)
    data.bring_to_front(channel_name)
    data.transpose()  # data should be in (detuning, setpoint)
    # process data ------------------------------------------------------------
    # cutoff
    channel_index = data.channel_names.index(channel_name)
    channel = data.channels[channel_index]
    cutoff = np.nanmax(channel.values)*cutoff_factor
    channel.values[channel.values<cutoff] = np.nan
    # fit
    function = wt_fit.Moments()
    fitter = wt_fit.Fitter(function, data, 'wm')
    outs = fitter.run()
    # spline
    xi = outs.axes[0].points
    yi = outs.one.values
    spline = wt_kit.Spline(xi, yi)
    offsets_splined = spline(xi)
    # make curve --------------------------------------------------------------
    curve = wt_curve.from_800_curve(filepath=old_curve_filepath)
    points = curve.colors    
    curve.colors += offsets_splined
    curve.map_colors(points)
    # plot --------------------------------------------------------------------
    fig, gs = wt_artists.create_figure(default_aspect=0.5)
    # heatmap
    ax = plt.subplot(gs[0, 0])
    xi = data.axes[1].points
    yi = data.axes[0].points
    zi = data.channels[channel_index].values
    X, Y, Z = wt_artists.pcolor_helper(xi, yi, zi)
    ax.pcolor(X, Y, Z, vmin=0, vmax=np.nanmax(zi), cmap=cmap)
    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(yi.min(), yi.max())
    # lines
    xi = outs.axes[0].points
    yi = outs.one.values
    ax.plot(xi, yi, c='grey', lw=5, alpha=0.5)
    ax.plot(xi, offsets_splined, c='k', lw=5, alpha=0.5)
    ax.axhline(c='k', lw=1)
    ax.grid()
    units_string = '$\mathsf{(' + wt_units.color_symbols[curve.units] + ')}$'
    ax.set_xlabel(' '.join(['setpoint', units_string]), fontsize=18)
    ax.set_ylabel('$\mathsf{\Delta' + wt_units.color_symbols[curve.units] + '}$', fontsize=18)
    # colorbar
    cax = plt.subplot(gs[:, -1])
    label = channel_name
    ticks = np.linspace(0, np.nanmax(zi), 7)
    wt_artists.plot_colorbar(cax=cax, cmap=cmap, label=label, ticks=ticks)
    # finish ------------------------------------------------------------------    
    if autosave:
        curve.save(save_directory=wt_kit.filename_parse(filepath)[0])
        p = os.path.join(os.path.dirname(filepath), 'tune test.png')
        wt_artists.savefig(p, fig=fig)
    return curve
