"""
Tools for processing spectral delay correction data.
"""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import sys
import imp
import time
import copy
import inspect
import itertools
import subprocess
import glob

try:
    import configparser as _ConfigParser  # python 3
except ImportError:
    import ConfigParser as _ConfigParser  # python 2

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
from . import curve as wt_curve
from . import coset as wt_coset


# --- processing methods --------------------------------------------------------------------------


def process_wigner(data_filepath, channel, control_name,
                   offset_name, coset_name, color_units='nm',
                   delay_units='fs', autosave=True, s=1000):
    """ Create a coset file from a measured wigner.

    Parameters
    ----------
    data_filepath : str
        Filepath to data file.
    channel : int or str
        The channel to process.
    """
    # get data
    data = wt_data.from_PyCMDS(data_filepath, verbose=False)
    if data.axes[0].units_kind == 'energy':
        data.transpose()  # prefered shape - delay then color
    data.convert(color_units)
    data.convert(delay_units)
    ws = data.axes[1].points
    # get channel index
    if type(channel) in [int, float]:
        channel_index = int(channel)
    elif type(channel) in [str]:
        channel_index = data.channel_names.index(channel)
    else:
        print('channel type not recognized')
        return
    # process
    function = wt_fit.Gaussian()
    fitter = wt_fit.Fitter(function, data, data.axes[0].name)
    outs = fitter.run(channel_index)
    # clean
    # remove the edges because they are badly behaved...
    # should probably do something more sophisticated...
    outs.amplitude.clip(outs.amplitude.max() * 0.1, outs.amplitude.max())
    #outs.amplitude.clip(outs.amplitude.max()*0.04, outs.amplitude.max())
    outs.amplitude.values[0] = np.nan
    outs.amplitude.values[-1] = np.nan
    outs.width.clip(0, 500)
    outs.mean.clip(-1000, 1000)
    outs.share_nans()
    centers = outs.channels[0].values
    # spline
    spline = wt_kit.Spline(ws, centers, k=2, s=s)
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
    if autosave:
        figure_path = data_filepath.replace('.data', '.png')
        plt.savefig(figure_path, dpi=300, transparent=True, pad_inches=1)
        plt.close()
    # construct, save coset
    coset = wt_coset.CoSet(control_name, color_units, ws, offset_name,
                           delay_units, corrections, coset_name)
    if autosave:
        coset.save(save_directory=os.path.dirname(data_filepath))
    return coset


def process_brute_force(data_filepath, opa_index, channel, color_units='nm',
                        delay_units='fs', amplitude_cutoff_factor=0.5,
                        plot=True, autosave=True):
    """
    This method is a good idea, but it isn't ready for prime time yet.
    - Blaise 2016.03.18
    """
    # get data
    data = wt_data.from_PyCMDS(data_filepath, verbose=False)
    # check if data is valid for this operation
    # TODO:
    # get channel index
    if type(channel) in [int, float]:
        channel_index = int(channel)
    elif type(channel) in [str]:
        channel_index = data.channel_names.index(channel)
    else:
        print('channel type not recognized')
        return
    # pre-process data
    data.normalize(channel_index)
    data.convert(color_units, verbose=False)
    data.convert(delay_units, verbose=False)
    # slice data
    ws = getattr(data, 'w{}'.format(opa_index)).points
    datas = data.chop('d2', 'd1', verbose=False)  # TODO: generalize...
    x_bins = [d.copy() for d in datas]
    y_bins = [d.copy() for d in datas]
    for d in x_bins:
        d.collapse(0, method='sum')
    for d in y_bins:
        d.collapse(1, method='sum')
    # choose corrections
    function = wt_fit.Gaussian()

    def fitit(d):
        xi = d.axes[0].points
        yi = d.channels[channel_index].values
        return function.fit(yi, xi)
    x_outs = [fitit(d) for d in x_bins]
    y_outs = [fitit(d) for d in y_bins]
    x_centers = [o[0] for o in x_outs]
    y_centers = [o[0] for o in y_outs]

    def get_corrections(ws, outs, lower_cutoff, upper_cutoff):
        ws_internal = ws.copy()
        centers = np.array([o[0] for o in outs])
        centers[centers < lower_cutoff] = np.nan
        centers[centers > upper_cutoff] = np.nan
        amplitudes = np.array([o[2] for o in outs])
        if False:
            # HACK - Blaise
            print(ws)
            amplitudes[:-20] = np.nan
            #plt.plot(ws, amplitudes)
        amplitudes[amplitudes < amplitudes.max() * amplitude_cutoff_factor] = np.nan
        ws_internal, centers, amplitudes = wt_kit.remove_nans_1D(
            [ws_internal, centers, amplitudes])
        spline = UnivariateSpline(ws_internal, centers, k=2, s=10000)
        return spline(ws)
    x_corrections = get_corrections(
        ws, x_outs, datas[0].axes[1].points.min(), datas[0].axes[1].points.max())
    y_corrections = get_corrections(
        ws, y_outs, datas[0].axes[0].points.min(), datas[0].axes[0].points.max())
    # plot data
    aspect = (datas[0].d2.max() - datas[0].d2.min()) / (datas[0].d1.max() - datas[0].d1.min())
    cmap = wt_artists.colormaps['default']
    if plot:
        for i, d in enumerate(datas):
            # prepare
            fig, gs = wt_artists.create_figure(aspects=[[[0, 0], aspect]])
            ax = plt.subplot(gs[0])
            # main plot
            xi = d.axes[1].points
            yi = d.axes[0].points
            zi = d.channels[channel_index].values
            X, Y, Z = wt_artists.pcolor_helper(xi, yi, zi)
            mappable = ax.pcolor(X, Y, Z, vmin=0, cmap=cmap)
            ax.set_xlim(xi.min(), xi.max())
            ax.set_ylim(yi.min(), yi.max())
            ax.set_xlabel(d.axes[1].get_label(), fontsize=18)
            ax.set_ylabel(d.axes[0].get_label(), fontsize=18)
            ax.grid()
            ax.axhline(y_centers[i], lw=4, c='grey')
            ax.axvline(x_centers[i], lw=4, c='grey')
            ax.axhline(y_corrections[i], lw=4, c='k')
            ax.axvline(x_corrections[i], lw=4, c='k')
            # x side plot
            sax = wt_artists.add_sideplot(ax, 'x')
            xi = x_bins[i].axes[0].points
            yi = x_bins[i].channels[channel_index].values
            sax.plot(xi, yi, lw=2, c='b')
            yi = function.evaluate(x_outs[i], xi)
            sax.plot(xi, yi, lw=2, c='grey')
            sax.axvline(x_centers[i], lw=4, c='grey')
            sax.axvline(x_corrections[i], lw=4, c='k')
            x_max_amp = max([o[2] for o in x_outs])
            sax.set_ylim(0, x_max_amp * 1.1)
            sax.grid()
            # y side plot
            sax = wt_artists.add_sideplot(ax, 'y')
            xi = y_bins[i].axes[0].points
            yi = y_bins[i].channels[channel_index].values
            sax.plot(yi, xi, lw=2, c='b')
            yi = function.evaluate(y_outs[i], xi)
            sax.plot(yi, xi, lw=2, c='grey')
            sax.axhline(y_centers[i], lw=4, c='grey')
            sax.axhline(y_corrections[i], lw=4, c='k')
            y_max_amp = max([o[2] for o in y_outs])
            sax.set_xlim(0, y_max_amp * 1.1)
            sax.grid()
            # colorbar
            cax = plt.subplot(gs[1])
            plt.colorbar(mappable=mappable, cax=cax)
            # title
            wt_artists._title(fig, str(ws[i]))
            # save
            figure_path = os.path.join(os.path.dirname(data_filepath), str(i).zfill(3) + '.png')
            plt.savefig(figure_path, dpi=300, transparent=True, pad_inches=1)
            plt.close(fig)
    # plot corrections array

    def plot_corrections(centers, corrections, title, name):
        fig, gs = wt_artists.create_figure(cols=[1])
        ax = plt.subplot(gs[0])
        ax.plot(ws, centers, lw=4, c='grey')
        ax.plot(ws, corrections, lw=4, c='k')
        wt_artists._title(fig, title)
        figure_path = os.path.join(os.path.dirname(data_filepath), name + '.png')
        plt.savefig(figure_path, dpi=300, transparent=True, pad_inches=1)
        plt.close(fig)
    if plot:
        plot_corrections(x_centers, x_corrections, datas[0].axes[1].get_label(
        ), '_'.join([datas[0].axes[1].name, 'w{}'.format(opa_index)]))
        plot_corrections(y_centers, y_corrections, datas[0].axes[0].get_label(
        ), '_'.join([datas[0].axes[0].name, 'w{}'.format(opa_index)]))
    # construct coset objects
    # TODO: generalize
    x_coset = wt_coset.CoSet('OPA2 TOPAS-C',
                             color_units,
                             ws,
                             'D1 SMC100',
                             delay_units,
                             x_corrections,
                             name='_'.join(['w{}'.format(opa_index),
                                            datas[0].axes[1].name]))
    y_coset = wt_coset.CoSet('OPA2 TOPAS-C',
                             color_units,
                             ws,
                             'D2 SMC100',
                             delay_units,
                             y_corrections,
                             name='_'.join(['w{}'.format(opa_index),
                                            datas[0].axes[0].name]))
    # save coset files
    if autosave:
        save_directory = os.path.dirname(data_filepath)
        x_coset.save(save_directory=save_directory)
        y_coset.save(save_directory=save_directory)
    # finish
    return [x_coset, y_coset]
