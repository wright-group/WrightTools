# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from ._helpers import create_figure, plot_colorbar, savefig, add_sideplot
from ._colors import colormaps
from .. import kit as wt_kit
from .. import data as wt_data


# --- define --------------------------------------------------------------------------------------


__all__ = ["quick2D_interactive"]


def get_axes(data, axes):
    xaxis, yaxis = axes
    if type(xaxis) in [int, str]:
        xaxis = data.axes[xaxis]
    elif type(xaxis) != wt_data.Axis:
        raise TypeError('invalid xaxis type {0}'.format(type(xaxis)))
    if type(yaxis) in [int, str]:
        yaxis = data.axes[yaxis]
    elif type(yaxis) != wt_data.Axis:
        raise TypeError('invalid xaxis type {0}'.format(type(yaxis)))
    return xaxis, yaxis


def get_channel(data, channel):
    if type(channel) in [int, str]:
        channel = data.channels[channel]
    elif type(channel) != wt_data.Channel:
        raise TypeError('invalid channel type {0}'.format(type(channel)))
    return channel


def get_colormap(channel):
    if channel.signed:
        cmap = "signed"
    else:
        cmap = "default"
    cmap = colormaps[cmap]
    cmap.set_bad([0.75] * 3, 1.)
    cmap.set_under([0.75] * 3, 1.)
    return cmap


def get_levels(channel, dynamic_range):
    if channel.signed:
        levels = np.linspace(-channel.mag(), channel.mag(), 200)
    else:
        levels = np.linspace(0, channel.max(), 200)
    return levels    


def set_aspect(xaxis, yaxis):
    if xaxis.units == yaxis.units:
        xr = xaxis.max() - xaxis.min()
        yr = yaxis.max() - yaxis.min()
        aspect = np.abs(yr / xr)
        if 3 < aspect or aspect < 1 / 3.:
            raise Warning('units agree, but aspect {0} required for equal spacing is too' /
                          + 'narrow.'.format(aspect))
            aspect = np.clip(aspect, 1 / 3., 3.)
            print('using aspect {0} instead'.format(aspect))
    else:
        aspect = 1
    return aspect


def quick2D_interactive(data, channel=0, axes=[0, 1], dynamic_range=False):
    channel = get_channel(data, channel)
    xaxis, yaxis = get_axes(data, axes)
    # get channel, scale appropriately
    cmap = get_colormap(channel)

    xaxis = data.axes[0]
    yaxis = data.axes[1]
    aspect = set_aspect(xaxis, yaxis)

    # how many sliders?
    nsliders = data.ndim - 2
    if nsliders < 0:
        print('note enough dimensions')
        return
    elif nsliders > 0:
        aspects = [[[0, 0], aspect]] + [[[i + 1, 0], 0.1] for i in range(nsliders)]
    else:
        aspects = [[[0, 0], aspect]]

    levels = get_levels(channel, dynamic_range)

    fig, gs = create_figure(width='single', nrows=1 + nsliders,
                            aspects=aspects, hspace=0.5)
    # draw
    ax0 = plt.subplot(gs[0])
    ax0.patch.set_facecolor("w")
    ax0.grid(b=True)

    slices = []
    for axis in data.axes:
        if axis in [xaxis, yaxis]:
            slices.append(slice(None))
        else:
            slices.append(slice(0, 1))
    # initial xyz start are from zero indices of additional axes
    zi = channel[slices]
    zi = zi.squeeze()
    if wt_kit.get_index(data.axes, xaxis) < wt_kit.get_index(data.axes, yaxis):
        zi = zi.T.copy()
    # print(xaxis.points.shape, yaxis.points.shape, zi.shape)

    obj2D = ax0.pcolormesh(xaxis.points, yaxis.points, zi,
                           cmap=cmap, vmin=levels.min(), vmax=levels.max())

    sp_x = add_sideplot(ax0, 'x', pad=0.3)
    sp_y = add_sideplot(ax0, 'y', pad=0.3)
    sp_x.fill_between(xaxis.points, zi.sum(axis=0), 0, color='k', alpha=0.3)
    sp_y.fill_betweenx(yaxis.points, zi.sum(axis=1), 0, color='k', alpha=0.3)

    sliders = {}
    for axis in data.axes:
        if axis not in [xaxis, yaxis]:
            slider_axes = plt.subplot(gs[len(sliders) + 1]).axes
            slider = Slider(slider_axes, axis.label,
                            0, axis.points.size - 1,
                            valinit=0, valstep=1, valfmt='%i')
            sliders[axis.natural_name] = slider

    def get_slice(data, sliders):
        slices = []
        for axis in data.axes:
            if axis.natural_name in sliders.keys():
                sliders[axis.natural_name].val
                slices.append(slice(sliders[axis.natural_name].val))
            else:
                slices.append(slice())
        return slices

    def update(val):
        slices = []
        for axis in data.axes:
            if axis.natural_name in sliders.keys():
                this_val = int(sliders[axis.natural_name].val)
                print(axis.natural_name, sliders[axis.natural_name].val,
                      axis.points[this_val])
                slices.append(slice(this_val, this_val+1))
            else:
                slices.append(slice(None))
        #print(slices)
        arr = channel[slices].squeeze()
        # TODO: check whether yaxis index is smaller (transpose not necessary)
        arr = arr.T.copy()
        print(arr.shape)
        # TODO: why am I stripping array information?
        # cf. https://stackoverflow.com/questions/29009743/using-set-array-with-pyplot-pcolormesh-ruins-figure
        obj2D.set_array(arr[:-1, :-1].ravel())
        sp_x.collections.clear()
        sp_y.collections.clear()
        x_proj = arr.sum(axis=0)
        y_proj = arr.sum(axis=1)
        print(x_proj.shape)
        sp_x.fill_between(xaxis.points, x_proj / np.abs(x_proj).max(), 0, color='k', alpha=0.3)
        sp_y.fill_betweenx(yaxis.points, y_proj / np.abs(y_proj).max(), 0, color='k', alpha=0.3)

        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)

    return obj2D, sliders
