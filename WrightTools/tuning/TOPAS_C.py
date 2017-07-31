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


# --- define --------------------------------------------------------------------------------------


spitfire_output = 800.  # nm


# --- helper methods ------------------------------------------------------------------------------


def _gauss_residuals(p, y, x):

    A, mu, sigma = p

    err = y - np.abs(A) * np.exp(-(x - mu)**2 / (2 * np.abs(sigma)**2))

    return np.abs(err)


def _exp_value(y, x):

    y_internal = np.ma.copy(y)
    x_internal = np.ma.copy(x)

    # get sum
    sum_y = 0.
    for i in range(len(y_internal)):
        if np.ma.getmask(y_internal[i]):
            pass
        elif np.isnan(y_internal[i]):
            pass
        else:
            sum_y = sum_y + y_internal[i]

    # divide by sum
    for i in range(len(y_internal)):
        if np.ma.getmask(y_internal[i]):
            pass
        elif np.isnan(y_internal[i]):
            pass
        else:
            y_internal[i] = y_internal[i] / sum_y

    # get expectation value
    value = 0.
    for i in range(len(x_internal)):
        if np.ma.getmask(y_internal[i]):
            pass
        elif np.isnan(y_internal[i]):
            pass
        else:
            value = value + y_internal[i] * x_internal[i]
    return value


# --- processing methods --------------------------------------------------------------------------


def process_C2_motortune(opa_index, data_filepath, curves, save=True):
    old_curve = wt_curve.from_TOPAS_crvs(curves, 'TOPAS-C', 'NON-NON-NON-Sig')
    # extract information from file
    headers = wt_kit.read_headers(data_filepath)
    wa_index = headers['name'].index('wa')
    zi_index = headers['name'].index('array_signal')
    # fit array data
    outs = []
    function = wt_fit.Gaussian()
    file_slicer = wt_kit.FileSlicer(data_filepath)
    print('fitting wa traces')
    while file_slicer.n < file_slicer.length:
        # get data from file
        lines = file_slicer.get(256)
        arr = np.array([np.fromstring(line, sep='\t') for line in lines]).T
        # fit data
        xi = arr[wa_index]
        xi = wt_units.converter(xi, 'nm', 'wn')
        yi = arr[zi_index]
        mean, width, amplitude, baseline = function.fit(yi, xi)
        mean = wt_units.converter(mean, 'wn', 'nm')
        outs.append([amplitude, mean, width])
        wt_kit.update_progress(100 * file_slicer.n / float(file_slicer.length - 256))
    outs = np.array(outs).T
    amp, cen, wid = outs
    # remove points with amplitudes that are ridiculous
    amp[amp < 0.1] = np.nan
    amp[amp > 5] = np.nan
    # remove points with centers that are ridiculous
    cen[cen < 1150] = np.nan
    cen[cen > 1650] = np.nan
    # remove points with widths that are ridiculous
    wid[wid < 5] = np.nan
    wid[wid > 500] = np.nan
    # finish removal
    amp, cen, wid = wt_kit.share_nans([amp, cen, wid])
    # get axes
    ws = np.array(headers['w%d points' % opa_index])
    c2 = np.array(headers['w%d_Crystal_2 points' % opa_index])
    # reshape
    amp.shape = (ws.size, c2.size)
    cen.shape = (ws.size, c2.size)
    wid.shape = (ws.size, c2.size)
    amp = amp.T
    cen = cen.T
    wid = wid.T
    # create mismatch array
    mismatch = cen - ws
    # fit to second order polynomial, take intercept
    ps = []
    for i in range(ws.size):
        xi = c2
        yi = mismatch[:, i]
        xi, yi = wt_kit.remove_nans_1D([xi, yi])
        if len(xi) == 0:
            ps.append(None)
            continue
        if yi.min() > 0 or yi.max() < 0:
            ps.append(None)
            continue
        p = np.ma.polyfit(yi, xi, 2)
        ps.append(p)
    chosen_ws = []
    chosen_c2 = []
    for i in range(ws.size):
        if ps[i] is not None:
            chosen_ws.append(ws[i])
            chosen_c2.append(old_curve.motors[2].positions[i] + ps[i][-1])
    # ensure smoothness with spline
    spline = UnivariateSpline(chosen_ws, chosen_c2, k=2, s=1000)
    chosen_c2 = spline(ws)
    # create new tuning curve
    curve = old_curve.copy()
    curve.motors[2].positions = chosen_c2
    # preapre for plot
    fig = plt.figure(figsize=[8, 6])
    cmap = wt_artists.colormaps['default']
    cmap.set_bad([0.75] * 3, 1.)
    cmap.set_under([0.75] * 3, 1.)
    gs = grd.GridSpec(2, 1, hspace=0.1, wspace=0.1)
    colors = wt_artists.get_color_cycle(ws.size, rotations=4)
    xroom = (curve.motors[2].positions.max() - curve.motors[2].positions.min()) / 10.
    xlim = [curve.motors[2].positions.min() - xroom, curve.motors[2].positions.max() + xroom]
    # detunings
    ax = plt.subplot(gs[1, 0])
    for i in range(ws.size):
        # real data
        xi = c2 + old_curve.motors[2].positions[i]
        yi = mismatch[:, i]
        ax.scatter(xi, yi, c=colors[i], edgecolor='none')
        # poly fit
        if ps[i] is not None:
            p = ps[i]
            yi = np.linspace(-100, 100, 100)
            xi = np.polyval(p, yi) + old_curve.motors[2].positions[i]
            ax.plot(xi, yi, c=colors[i], lw=1)
        # chosen point
        ax.scatter(curve.motors[2].positions[i], 0, c=colors[i], marker='x', s=100)
    ax.grid()
    x1 = old_curve.motors[2].positions.min()
    x2 = old_curve.motors[2].positions.max()
    plt.plot([x1, x2], [0, 0], c='k', lw=1)
    ax.set_xlabel('C2 (deg)', fontsize=16)
    ax.set_ylabel('detuning (nm)', fontsize=16)
    ax.set_xlim(*xlim)
    ax.set_ylim(-45, 45)
    # curves
    ax = plt.subplot(gs[0, 0])
    ax.plot(old_curve.motors[2].positions, old_curve.colors, c='k', lw=1)
    for i in range(ws.size):
        ax.scatter(curve.motors[2].positions[i], curve.colors[i], c=colors[i], marker='x', s=100)
    ax.set_xlim(ws.min(), ws.max())
    ax.grid()
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('setpoint (nm)', fontsize=16)
    ax.set_xlim(*xlim)
    # finish plot
    title = os.path.basename(data_filepath).replace('.data', '')[-19:]  # extract timestamp
    plt.suptitle(title, fontsize=20)
    # finish
    if save:
        directory = os.path.dirname(data_filepath)
        path = curve.save(save_directory=directory)
        image_path = data_filepath.replace('.data', '.png')
        plt.savefig(image_path, dpi=300, transparent=True)
        plt.close(fig)
    return curve


def process_D2_motortune(opa_index, data_filepath, curves, save=True):
    old_curve = wt_curve.from_TOPAS_crvs(curves, 'TOPAS-C', 'NON-NON-NON-Sig')
    # extract information from file
    headers = wt_kit.read_headers(data_filepath)
    wa_index = headers['name'].index('wa')
    zi_index = headers['name'].index('array_signal')
    # fit array data
    outs = []
    function = wt_fit.Gaussian()
    file_slicer = wt_kit.FileSlicer(data_filepath)
    print('fitting wa traces')
    while file_slicer.n < file_slicer.length:
        # get data from file
        lines = file_slicer.get(256)
        arr = np.array([np.fromstring(line, sep='\t') for line in lines]).T
        # fit data
        xi = arr[wa_index]
        xi = wt_units.converter(xi, 'nm', 'wn')
        yi = arr[zi_index]
        mean, width, amplitude, baseline = function.fit(yi, xi)
        mean = wt_units.converter(mean, 'wn', 'nm')
        outs.append([amplitude, mean, width])
        wt_kit.update_progress(100 * file_slicer.n / float(file_slicer.length - 256))
    print(file_slicer.n, file_slicer.length)
    outs = np.array(outs).T
    amp, cen, wid = outs
    # remove points with amplitudes that are ridiculous
    amp[amp < 0.1] = np.nan
    amp[amp > 5] = np.nan
    # remove points with centers that are ridiculous
    cen[cen < 1150] = np.nan
    cen[cen > 1650] = np.nan
    # remove points with widths that are ridiculous
    wid[wid < 5] = np.nan
    wid[wid > 500] = np.nan
    # finish removal
    amp, cen, wid = wt_kit.share_nans([amp, cen, wid])
    # get axes
    ws = np.array(headers['w%d points' % opa_index])
    d2 = np.array(headers['w%d_Delay_2 points' % opa_index])
    # reshape
    amp.shape = (ws.size, d2.size)
    cen.shape = (ws.size, d2.size)
    wid.shape = (ws.size, d2.size)
    amp = amp.T
    cen = cen.T
    wid = wid.T
    # chose points by expectation value
    function = wt_fit.ExpectationValue()
    chosen_deltas = np.full(ws.size, np.nan)
    for i in range(ws.size):
        yi = amp[:, i]
        outs = function.fit(yi, d2)
        chosen_deltas[i] = outs[0]
    # calculate actual d2 motor positions (rather than just deltas)
    chosen_d2 = old_curve.motors[3].positions + chosen_deltas
    # ensure smoothness with spline
    spline = UnivariateSpline(ws, chosen_d2, k=2, s=1000)
    chosen_d2 = spline(ws)
    # create new tuning curve
    curve = old_curve.copy()
    curve.motors[3].positions = chosen_d2
    # preapre for plot
    fig = plt.figure(figsize=[8, 6])
    cmap = wt_artists.colormaps['default']
    cmap.set_bad([0.75] * 3, 1.)
    cmap.set_under([0.75] * 3, 1.)
    gs = grd.GridSpec(2, 2, hspace=0.1, wspace=0.1, width_ratios=[20, 1])
    # lines
    ax = plt.subplot(gs[0, 0])
    ax.plot(old_curve.colors, old_curve.motors[3].positions, c='k', lw=1)
    ax.plot(curve.colors, curve.motors[3].positions, c='k', lw=5)
    ax.set_xlim(ws.min(), ws.max())
    ax.grid()
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('D2', fontsize=16)
    # pcolor
    ax = plt.subplot(gs[1, 0])
    X, Y, Z = wt_artists.pcolor_helper(ws, d2, amp)
    mappable = ax.pcolor(X, Y, Z, vmin=0, vmax=np.nanmax(Z), cmap=cmap)
    plt.axhline(c='k', lw=1)
    ax.plot(ws, chosen_deltas, c='grey', lw=5)
    final_deltas = curve.motors[3].positions - old_curve.motors[3].positions
    ax.plot(ws, final_deltas, c='k', lw=5)
    ax.set_xlim(ws.min(), ws.max())
    ax.set_ylim(d2.min(), d2.max())
    ax.grid()
    ax.set_xlabel('setpoint (nm)', fontsize=16)
    ax.set_ylabel('$\mathsf{\Delta}$D2', fontsize=16)
    # colorbar
    cax = plt.subplot(gs[1, 1])
    plt.colorbar(mappable=mappable, cax=cax)
    # finish plot
    title = os.path.basename(data_filepath).replace('.data', '')[-19:]  # extract timestamp
    plt.suptitle(title, fontsize=20)
    # finish
    if save:
        directory = os.path.dirname(data_filepath)
        path = curve.save(save_directory=directory)
        image_path = data_filepath.replace('.data', '.png')
        plt.savefig(image_path, dpi=300, transparent=True)
        plt.close(fig)
    return curve


def process_preamp_motortune(OPA_index, data_filepath, curves, save=True):
    # extract information from file
    headers = wt_kit.read_headers(data_filepath)
    arr = np.genfromtxt(data_filepath).T
    old_curve = wt_curve.from_TOPAS_crvs(curves, 'TOPAS-C', 'NON-NON-NON-Sig')
    # get array data
    array_colors = arr[headers['name'].index('wa')]
    array_data = arr[headers['name'].index('array_signal')]
    array_colors.shape = (-1, 256)
    array_data.shape = (-1, 256)
    array_colors = wt_units.converter(array_colors, 'nm', 'wn')
    # fit array data
    outs = []
    for i in range(len(array_data)):
        xi = array_colors[i]
        yi = array_data[i]
        function = wt_fit.Gaussian()
        mean, width, amplitude, baseline = function.fit(yi, xi)
        mean = wt_units.converter(mean, 'wn', 'nm')
        outs.append([amplitude, mean, width])
    outs = np.array(outs).T
    amps, centers, widths = outs
    # get crystal 1 data
    array_c1 = arr[headers['name'].index('w{}_Crystal_1'.format(OPA_index))]
    array_c1.shape = (-1, 256)
    c1_list = array_c1[..., 0].flatten()
    # get delay 1 data
    array_d1 = arr[headers['name'].index('w{}_Delay_1'.format(OPA_index))]
    array_d1.shape = (-1, 256)
    d1_list = array_d1[..., 0].flatten()
    d1_width = (array_d1[:, 0].max() - array_d1[:, 0].min()) / 2.
    # remove points with amplitudes that are ridiculous
    amps[amps < 0.1] = np.nan
    amps[amps > 4] = np.nan
    # remove points with centers that are ridiculous
    centers[centers < 1150] = np.nan
    centers[centers > 1650] = np.nan
    # remove points with widths that are ridiculous
    widths[widths < 5] = np.nan
    widths[widths > 500] = np.nan
    # finish removal
    amps, centers, widths, c1_list, d1_list = wt_kit.remove_nans_1D(
        [amps, centers, widths, c1_list, d1_list])
    # grid data (onto a fine grid)
    points_count = 100
    c1_points = np.linspace(c1_list.min(), c1_list.max(), points_count)
    d1_points = np.linspace(d1_list.min(), d1_list.max(), points_count)
    xi = tuple(np.meshgrid(c1_points, d1_points, indexing='xy'))
    c1_grid, d1_grid = xi
    points = tuple([c1_list, d1_list])
    amp_grid = griddata(points, amps, xi)
    cen_grid = griddata(points, centers, xi)
    # get indicies with centers near setpoints (bin data)
    setpoints = np.linspace(1140, 1620, 25)
    within = 2
    fits_by_setpoint = []
    print('binning points')
    for i in range(len(setpoints)):
        these_fits = []
        for j in range(points_count):
            for k in range(points_count):
                if np.isnan(amp_grid[k, j]):
                    pass
                else:
                    if np.abs(cen_grid[k, j] - setpoints[i]) < within:
                        motor_positions = [c1_grid[0, j], d1_grid[k, 0]]
                        amplitude = amp_grid[k, j]
                        these_fits.append([motor_positions, amplitude])
                    else:
                        pass
        if len(these_fits) > 0:
            max_amplitude = max([f[1] for f in these_fits])
            these_fits = [f for f in these_fits if f[1] > max_amplitude * 0.25]
        fits_by_setpoint.append(these_fits)
        wt_kit.update_progress(100 * i / float(len(setpoints)))
    wt_kit.update_progress(100)
    false_setpoints = []
    for i in range(len(setpoints)):
        if len(fits_by_setpoint[i]) == 0:
            false_setpoints.append(setpoints[i])
    # fit each setpoint
    preamp_chosen = []
    print('fitting points')
    for i in range(len(setpoints)):
        c1s = np.zeros(len(fits_by_setpoint[i]))
        d1s = np.zeros(len(fits_by_setpoint[i]))
        y = np.zeros(len(fits_by_setpoint[i]))
        chosen = np.zeros(3)  # color, c1, d1
        if len(y) == 0:
            continue
        for j in range(len(fits_by_setpoint[i])):
            c1s[j] = fits_by_setpoint[i][j][0][0]
            d1s[j] = fits_by_setpoint[i][j][0][1]
            y[j] = fits_by_setpoint[i][j][1]
        if len(y) < 4:
            # choose by expectation value if you don't have many points (failsafe)
            chosen[0] = setpoints[i]
            chosen[1] = _exp_value(y, c1s)
            chosen[2] = _exp_value(y, d1s)
        else:
            # fit to a guassian
            # c1
            amplitude_guess = max(y)
            center_guess = old_curve.get_motor_positions(setpoints[i])[0]
            sigma_guess = 100.
            p0 = np.array([amplitude_guess, center_guess, sigma_guess])
            try:
                out_c1 = leastsq(_gauss_residuals, p0, args=(y, c1s))[0]
            except RuntimeWarning:
                print('runtime')
            # d1
            amplitude_guess = max(y)
            center_guess = old_curve.get_motor_positions(setpoints[i])[1]
            sigma_guess = 100.
            p0 = np.array([amplitude_guess, center_guess, sigma_guess])
            try:
                out_d1 = leastsq(_gauss_residuals, p0, args=(y, d1s))[0]
            except RuntimeWarning:
                print('runtime')
            # write to preamp_chosen
            chosen[0] = setpoints[i]
            chosen[1] = out_c1[1]
            chosen[2] = out_d1[1]
        if chosen[1] < c1s.min() or chosen[1] > c1s.max():
            chosen[0] = setpoints[i]
            chosen[1] = _exp_value(y, c1s)
            chosen[2] = _exp_value(y, d1s)
        elif chosen[2] < d1s.min() or chosen[2] > d1s.max():
            chosen[0] = setpoints[i]
            chosen[1] = _exp_value(y, c1s)
            chosen[2] = _exp_value(y, d1s)
        else:
            pass
        preamp_chosen.append(chosen)
        wt_kit.update_progress(100 * i / float(len(setpoints)))
    wt_kit.update_progress(100)
    colors_chosen = [pc[0] for pc in preamp_chosen]
    c1s_chosen = [pc[1] for pc in preamp_chosen]
    d1s_chosen = [pc[2] for pc in preamp_chosen]
    # extend curve using spline
    if True:
        c1_spline = UnivariateSpline(colors_chosen, c1s_chosen, k=2, s=1000)
        d1_spline = UnivariateSpline(colors_chosen, d1s_chosen, k=2, s=1000)
        preamp_chosen = np.zeros([len(setpoints), 3])
        for i in range(len(setpoints)):
            preamp_chosen[i][0] = setpoints[i]
            preamp_chosen[i][1] = c1_spline(setpoints[i])
            preamp_chosen[i][2] = d1_spline(setpoints[i])
        false_points = np.zeros([len(false_setpoints), 3])
        for i in range(len(false_setpoints)):
            false_points[i][0] = false_setpoints[i]
            false_points[i][1] = c1_spline(false_setpoints[i])
            false_points[i][2] = d1_spline(false_setpoints[i])
    # create new curve
    colors = np.array([pc[0] for pc in preamp_chosen])
    motors = []
    old_curve_copy = old_curve.copy()
    old_curve_copy.map_colors(colors)
    for i, name in zip(range(1, 3), ['Crystal_1', 'Delay_1']):
        motors.append(wt_curve.Motor([pc[i] for pc in preamp_chosen], name))
    for i in range(2, 4):
        motors.append(old_curve_copy.motors[i])
    curve = old_curve.copy()
    curve.colors = colors
    curve.motors = motors
    curve.map_colors(setpoints)
    # preapre for plot
    fig, gs = wt_artists.create_figure(width='single', cols=[1, 'cbar'])
    cmap = wt_artists.colormaps['default']
    cmap.set_bad([0.75] * 3, 1.)
    cmap.set_under([0.75] * 3, 1.)
    # plot amplitude data
    ax = plt.subplot(gs[0, 0])
    X, Y, Z = wt_artists.pcolor_helper(c1_points, d1_points, amp_grid)
    mappable = ax.pcolor(X, Y, Z, vmin=0, vmax=np.nanmax(Z), cmap=cmap)
    # plot and label contours of constant color
    CS = plt.contour(c1_points, d1_points, cen_grid, colors='grey', levels=setpoints)
    clabel_positions = np.zeros([len(preamp_chosen), 2])
    clabel_positions[:, 0] = preamp_chosen[:, 1]
    clabel_positions[:, 1] = preamp_chosen[:, 2]
    plt.clabel(CS, inline=0, fontsize=9, manual=clabel_positions, colors='w', fmt='%1.0f')
    # plot old points, edges of acquisition
    xi = old_curve.motors[0].positions
    yi = old_curve.motors[1].positions
    plt.plot(xi, yi, c='k')
    plt.plot(xi, yi + d1_width, c='k', ls='--')
    plt.plot(xi, yi - d1_width, c='k', ls='--')
    for x, y in zip([xi[0], xi[-1]], [yi[0], yi[-1]]):
        xs = [x, x]
        ys = [y - d1_width, y + d1_width]
        plt.plot(xs, ys, c='k', ls='--')
    # plot points chosen by fits
    xi = [pc[1] for pc in preamp_chosen]
    yi = [pc[2] for pc in preamp_chosen]
    plt.plot(xi, yi, c='grey', lw=5)
    # plot smoothed points
    xi = curve.motors[0].positions
    yi = curve.motors[1].positions
    plt.plot(xi, yi, c='k', lw=5)
    # finish plot
    plt.xlabel('C1 (deg)', fontsize=18)
    plt.ylabel('D1 (mm)', fontsize=18)
    title = os.path.basename(data_filepath)
    plt.suptitle(title)
    plt.gca().patch.set_facecolor([0.75] * 3)
    plt.xlim(xi.min() - 0.25, xi.max() + 0.25)
    plt.ylim(yi.min() - 0.05, yi.max() + 0.05)
    # colorbar
    cax = plt.subplot(gs[:, -1])
    plt.colorbar(mappable=mappable, cax=cax)
    cax.set_ylabel('intensity', fontsize=18)
    # plot at an index (for debugging purposes only)
    # TODO: remove this eventually...
    if False:
        setpoint_index = 20
        fig2 = plt.figure()
        for i in range(len(fits_by_setpoint[setpoint_index])):
            c1_point = fits_by_setpoint[setpoint_index][i][0][0]
            d1_point = fits_by_setpoint[setpoint_index][i][0][1]
            amp = fits_by_setpoint[setpoint_index][i][1]
            fig.gca().scatter(c1_point, d1_point)
            fig2.gca().scatter(c1_point, amp)
            plt.title('c1 vs amp - {} nm'.format(setpoints[setpoint_index]))
        plt.show()
    # write files
    if save:
        directory = os.path.dirname(data_filepath)
        path = curve.save(save_directory=directory)
        image_path = data_filepath.replace('.data', '.png')
        # TODO: figure out how to get transparent background >:-(
        plt.savefig(image_path, dpi=300, transparent=False)
        plt.close(fig)
    # finish
    return curve


def process_SHS_motortune(OPA_index, data_filepath, curves, save=True):
    old_curve = wt_curve.from_TOPAS_crvs(curves, 'TOPAS-C', 'NON-SH-NON-Sig')
    # extract information from headers
    headers = wt_kit.read_headers(data_filepath)
    m2_index = headers['name'].index('w%d_Mixer_2' % OPA_index)
    wm_index = headers['name'].index('wm')
    zi_index = headers['kind'].index('channel')  # the first channel
    ws = headers['w%d points' % OPA_index]
    ws_len = len(ws)
    wm_len = len(headers['wm points'])
    m2_len = len(headers['w%d_Mixer_2 points' % OPA_index])
    # get arrays
    arr = np.genfromtxt(data_filepath).T
    wm = arr[wm_index]
    wm.shape = (ws_len, m2_len, wm_len)
    wm = wt_units.converter(wm, 'nm', 'wn')
    m2 = arr[m2_index]
    m2.shape = (ws_len, m2_len, wm_len)
    zi = arr[zi_index]
    zi.shape = (ws_len, m2_len, wm_len)
    # fit each mono slice
    outs = np.full((ws_len, m2_len, 4), np.nan)
    function = wt_fit.Gaussian()
    for idx in np.ndindex(ws_len, m2_len):
        xi = wm[idx]
        yi = zi[idx]
        out = function.fit(yi, xi)
        outs[idx] = out
    outs = outs.T
    cen, wid, amp, base = outs
    cen = wt_units.converter(cen, 'wn', 'nm')
    # remove points with amplitudes that are ridiculous
    amp[amp < 0.1] = np.nan
    amp[amp > 5] = np.nan
    # remove points with centers that are ridiculous
    cen[cen < 550] = np.nan
    cen[cen > 810] = np.nan
    # remove points with widths that are ridiculous
    wid[wid < 5] = np.nan
    wid[wid > 500] = np.nan
    # finish removal
    amp, cen, wid = wt_kit.share_nans([amp, cen, wid])
    # get ws, m2
    ws = np.array(headers['w%d points' % OPA_index])
    m2 = np.array(headers['w%d_Mixer_2 points' % OPA_index])
    # choose best mixer position by expectation value
    function = wt_fit.ExpectationValue()
    chosen_deltas = np.full(ws.size, np.nan)
    for i in range(ws.size):
        yi = amp[:, i]
        outs = function.fit(yi, m2)
        chosen_deltas[i] = outs[0]
    # calculate actual m2 motor positions (rather than just deltas)
    chosen_m2 = old_curve.motors[0].positions + chosen_deltas
    # find corresponding color through linear interpolation
    chosen_colors = np.full(ws.size, np.nan)
    for i in range(ws.size):
        xi = m2
        yi = cen[:, i]
        xi, yi = wt_kit.remove_nans_1D([xi, yi])
        if len(xi) > 1:
            interp = interp1d(xi, yi)
            chosen_colors[i] = interp(chosen_deltas[i])
        else:
            chosen_colors[i] = ws[i]
    # ensure smoothness with spline
    spline = UnivariateSpline(chosen_colors, chosen_m2, k=2, s=1000)
    chosen_m2 = spline(chosen_colors)
    # create new tuning curve
    curve = old_curve.copy()
    curve.colors = chosen_colors
    curve.motors[0].positions = chosen_m2
    curve.interpolate()
    setpoints = np.linspace(570, 810, 13)
    curve.map_colors(setpoints)
    # preapre for plot
    fig = plt.figure(figsize=[8, 6])
    cmap = wt_artists.colormaps['default']
    cmap.set_bad([0.75] * 3, 1.)
    cmap.set_under([0.75] * 3, 1.)
    gs = grd.GridSpec(2, 2, hspace=0.1, wspace=0.1, width_ratios=[20, 1])
    # lines
    ax = plt.subplot(gs[0, 0])
    ax.plot(old_curve.colors, old_curve.motors[0].positions, c='k', lw=1)
    ax.plot(curve.colors, curve.motors[0].positions, c='k', lw=5)
    ax.set_xlim(ws.min(), ws.max())
    ax.grid()
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('M2', fontsize=16)
    # pcolor
    cmap = wt_artists.colormaps['default']
    cmap.set_bad([0.75] * 3, 1.)
    cmap.set_under([0.75] * 3, 1.)
    X, Y, Z = wt_artists.pcolor_helper(ws, m2, amp)
    ax = plt.subplot(gs[1, 0])
    mappable = ax.pcolor(X, Y, Z, vmin=0, vmax=np.nanmax(amp), cmap=cmap)
    ax.set_xlim(ws.min(), ws.max())
    ax.set_ylim(m2.min(), m2.max())
    plt.axhline(c='k', lw=1)
    ax.plot(ws, chosen_deltas, c='grey', lw=5)
    old_curve.map_colors(setpoints)
    final_deltas = curve.motors[0].positions - old_curve.motors[0].positions
    ax.plot(setpoints, final_deltas, c='k', lw=5)
    ax.grid()
    ax.set_xlabel('setpoint (nm)', fontsize=16)
    ax.set_ylabel('$\mathsf{\Delta}$M2', fontsize=16)
    # colorbar
    cax = plt.subplot(gs[1, 1])
    plt.colorbar(mappable=mappable, cax=cax)
    cax.set_ylabel('intensity', fontsize=18)
    # finish plot
    title = os.path.basename(data_filepath).replace('.data', '')[-19:]  # extract timestamp
    #plt.suptitle(title, fontsize=20)
    # finish
    if save:
        directory = os.path.dirname(data_filepath)
        path = curve.save(save_directory=directory)
        image_path = data_filepath.replace('.data', '.png')
        plt.savefig(image_path, dpi=300, transparent=True)
        plt.close(fig)
    return curve


def process_SFS_motortune(OPA_index, data_filepath, curves, save=True):
    old_curve = wt_curve.from_TOPAS_crvs(curves, 'TOPAS-C', 'NON-NON-SF-Sig')
    # extract information from headers
    headers = wt_kit.read_headers(data_filepath)
    m2_index = headers['name'].index('w%d_Mixer_1' % OPA_index)
    wm_index = headers['name'].index('wm')
    zi_index = headers['kind'].index('channel')  # the first channel
    ws = headers['w%d points' % OPA_index]
    ws_len = len(ws)
    wm_len = len(headers['wm points'])
    m1_len = len(headers['w%d_Mixer_1 points' % OPA_index])
    # get arrays
    arr = np.genfromtxt(data_filepath).T
    wm = arr[wm_index]
    wm.shape = (ws_len, m1_len, wm_len)
    wm = wt_units.converter(wm, 'nm', 'wn')
    m2 = arr[m2_index]
    m2.shape = (ws_len, m1_len, wm_len)
    zi = arr[zi_index]
    zi.shape = (ws_len, m1_len, wm_len)
    # fit each mono slice
    outs = np.full((ws_len, m1_len, 4), np.nan)
    function = wt_fit.Gaussian()
    for idx in np.ndindex(ws_len, m1_len):
        xi = wm[idx]
        yi = zi[idx]
        out = function.fit(yi, xi)
        outs[idx] = out
    outs = outs.T
    cen, wid, amp, base = outs
    cen = wt_units.converter(cen, 'wn', 'nm')
    # remove points with amplitudes that are ridiculous
    amp[amp < 0.1] = np.nan
    amp[amp > 5] = np.nan
    # remove points with centers that are ridiculous
    cen[cen < 470] = np.nan
    cen[cen > 550] = np.nan
    # remove points with widths that are ridiculous
    wid[wid < 5] = np.nan
    wid[wid > 500] = np.nan
    # finish removal
    amp, cen, wid = wt_kit.share_nans([amp, cen, wid])
    # get ws, m1
    ws = np.array(headers['w%d points' % OPA_index])
    m1 = np.array(headers['w%d_Mixer_1 points' % OPA_index])
    # choose best mixer position by expectation value
    function = wt_fit.ExpectationValue()
    chosen_deltas = np.full(ws.size, np.nan)
    for i in range(ws.size):
        yi = amp[:, i]
        outs = function.fit(yi, m1)
        chosen_deltas[i] = outs[0]
    # calculate actual m1 motor positions (rather than just deltas)
    chosen_m1 = old_curve.motors[0].positions + chosen_deltas
    # find corresponding color through linear interpolation
    chosen_colors = np.full(ws.size, np.nan)
    for i in range(ws.size):
        xi = m1
        yi = cen[:, i]
        xi, yi = wt_kit.remove_nans_1D([xi, yi])
        if len(xi) > 1:
            interp = interp1d(xi, yi)
            chosen_colors[i] = interp(chosen_deltas[i])
        else:
            chosen_colors[i] = ws[i]
    # ensure smoothness with spline
    spline = UnivariateSpline(chosen_colors, chosen_m1, k=2, s=1000)
    chosen_m1 = spline(chosen_colors)
    # create new tuning curve
    curve = old_curve.copy()
    curve.colors = chosen_colors
    curve.motors[0].positions = chosen_m1
    curve.interpolate()
    setpoints = np.linspace(480, 540, 13)
    curve.map_colors(setpoints)
    # preapre for plot
    fig = plt.figure(figsize=[8, 6])
    cmap = wt_artists.colormaps['default']
    cmap.set_bad([0.75] * 3, 1.)
    cmap.set_under([0.75] * 3, 1.)
    gs = grd.GridSpec(2, 2, hspace=0.1, wspace=0.1, width_ratios=[20, 1])
    # lines
    ax = plt.subplot(gs[0, 0])
    ax.plot(old_curve.colors, old_curve.motors[0].positions, c='k', lw=1)
    ax.plot(curve.colors, curve.motors[0].positions, c='k', lw=5)
    ax.set_xlim(ws.min(), ws.max())
    ax.grid()
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('M1', fontsize=16)
    # pcolor
    cmap = wt_artists.colormaps['default']
    cmap.set_bad([0.75] * 3, 1.)
    cmap.set_under([0.75] * 3, 1.)
    X, Y, Z = wt_artists.pcolor_helper(ws, m1, amp)
    ax = plt.subplot(gs[1, 0])
    mappable = ax.pcolor(X, Y, Z, vmin=0, vmax=np.nanmax(amp), cmap=cmap)
    ax.set_xlim(ws.min(), ws.max())
    ax.set_ylim(m1.min(), m1.max())
    plt.axhline(c='k', lw=1)
    ax.plot(ws, chosen_deltas, c='grey', lw=5)
    old_curve.map_colors(setpoints)
    final_deltas = curve.motors[0].positions - old_curve.motors[0].positions
    ax.plot(setpoints, final_deltas, c='k', lw=5)
    ax.grid()
    ax.set_xlabel('setpoint (nm)', fontsize=16)
    ax.set_ylabel('$\mathsf{\Delta}$M1', fontsize=16)
    # colorbar
    cax = plt.subplot(gs[1, 1])
    plt.colorbar(mappable=mappable, cax=cax)
    # finish plot
    title = os.path.basename(data_filepath).replace('.data', '')[-19:]  # extract timestamp
    plt.suptitle(title, fontsize=20)
    # finish
    if save:
        directory = os.path.dirname(data_filepath)
        path = curve.save(save_directory=directory, old_filepaths=curves)
        image_path = data_filepath.replace('.data', '.png')
        plt.savefig(image_path, dpi=300, transparent=True)
        plt.close(fig)
    return curve
