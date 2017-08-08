"""
Methods for processing OPA 800 tuning data.
"""


# --- import --------------------------------------------------------------------------------------


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

from . import curve as wt_curve
from .. import artists as wt_artists
from .. import data as wt_data
from .. import fit as wt_fit
from .. import kit as wt_kit
from .. import units as wt_units


# --- define --------------------------------------------------------------------------------------


cmap = wt_artists.colormaps['default']
cmap.set_bad([0.75] * 3, 1.)
cmap.set_under([0.75] * 3)

# --- processing methods --------------------------------------------------------------------------


def intensity(data, curve, channel_name, level=False, cutoff_factor=0.1,
              autosave=True, save_directory=None):
    """

    Parameters
    ----------
    data : wt.data.Data objeect
        should be in (setpoint, motor)

    Returns
    -------
    curve
        New curve object.
    """
    # TODO: documentation
    data.transpose()
    channel_index = data.channel_names.index(channel_name)
    tune_points = curve.colors
    # process data --------------------------------------------------------------------------------
    if level:
        data.level(channel_index, 0, -3)
    # cutoff
    channel = data.channels[channel_index]
    cutoff = np.nanmax(channel.values) * cutoff_factor
    channel.values[channel.values < cutoff] = np.nan
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
    # make curve ----------------------------------------------------------------------------------
    old_curve = curve.copy()
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
    # plot ----------------------------------------------------------------------------------------
    fig, gs = wt_artists.create_figure(nrows=2, default_aspect=0.5, cols=[1, 'cbar'])
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
    ax.set_ylabel(
        ' '.join(['$\mathsf{\Delta}$', curve.motor_names[tuned_motor_index]]), fontsize=18)
    # colorbar
    cax = plt.subplot(gs[1, -1])
    label = channel_name
    ticks = np.linspace(0, np.nanmax(zi), 7)
    wt_artists.plot_colorbar(cax=cax, cmap=cmap, label=label, ticks=ticks)
    # finish --------------------------------------------------------------------------------------
    if autosave:
        if save_directory is None:
            save_directory = os.getcwd()
        curve.save(save_directory=save_directory, full=True)
        p = os.path.join(save_directory, 'intensity.png')
        wt_artists.savefig(p, fig=fig)
    return curve


def tune_test(data, curve, channel_name, level=False, cutoff_factor=0.01,
              autosave=True, save_directory=None):
    """

    Parameters
    ----------
    data : wt.data.Data object
        should be in (setpoint, detuning)
    curve : wt.curve object
        tuning curve used to do tune_test
    channel_nam : str
        name of the signal chanel to evalute
    level : bool (optional)
        does nothing, default is False
    cutoff_factor : float (optoinal)
        minimum value for datapoint/max(datapoints) for point to be included
        in the fitting procedure, default is 0.01
    autosave : bool (optional)
        saves output curve if True, default is True
    save_directory : str
        directory to save new curve, default is None which uses the data source
        directory

    Returns
    -------
    curve
        New curve object.
    """
    # make data object
    data = data.copy()
    data.bring_to_front(channel_name)
    data.transpose()
    # process data --------------------------------------------------------------------------------
    # cutoff
    channel_index = data.channel_names.index(channel_name)
    channel = data.channels[channel_index]
    cutoff = np.nanmax(channel.values) * cutoff_factor
    channel.values[channel.values < cutoff] = np.nan
    # fit
    gauss_function = wt_fit.Gaussian()
    g_fitter = wt_fit.Fitter(gauss_function, data, data.axes[0].name)
    outs = g_fitter.run()
    # spline
    xi = outs.axes[0].points
    yi = outs.mean.values
    spline = wt_kit.Spline(xi, yi)
    offsets_splined = spline(xi)  # wn
    # make curve ----------------------------------------------------------------------------------
    curve = curve.copy()
    curve_native_units = curve.units
    curve.convert('wn')
    points = curve.colors.copy()
    curve.colors += offsets_splined
    curve.map_colors(points, units='wn')
    curve.convert(curve_native_units)
    # plot ----------------------------------------------------------------------------------------
    data.axes[1].convert(curve_native_units)
    fig, gs = wt_artists.create_figure(default_aspect=0.5, cols=[1, 'cbar'])
    fig, gs = wt_artists.create_figure(default_aspect=0.5, cols=[1, 'cbar'])
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
    outs.convert(curve_native_units)
    xi = outs.axes[0].points
    yi = outs.mean.values
    ax.plot(xi, yi, c='grey', lw=5, alpha=0.5)
    ax.plot(xi, offsets_splined, c='k', lw=5, alpha=0.5)
    ax.axhline(c='k', lw=1)
    ax.grid()
    units_string = '$\mathsf{(' + wt_units.color_symbols[curve.units] + ')}$'
    ax.set_xlabel(r' '.join(['setpoint', units_string]), fontsize=18)
    ax.set_ylabel(r'$\mathsf{\Delta' + wt_units.color_symbols['wn'] + '}$', fontsize=18)
    # colorbar
    cax = plt.subplot(gs[:, -1])
    label = channel_name
    ticks = np.linspace(0, np.nanmax(zi), 7)
    wt_artists.plot_colorbar(cax=cax, cmap=cmap, label=label, ticks=ticks)
    # finish --------------------------------------------------------------------------------------
    if autosave:
        if save_directory is None:
            save_directory = os.path.dirname(data.source)
        curve.save(save_directory=save_directory, full=True)
        p = os.path.join(save_directory, 'tune test.png')
        wt_artists.savefig(p, fig=fig)
    return curve
