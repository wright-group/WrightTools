# --- import --------------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from ._helpers import create_figure, plot_colorbar, savefig, add_sideplot
from ._colors import colormaps
from .. import kit as wt_kit
from .. import data as wt_data


# --- define --------------------------------------------------------------------------------------


__all__ = ["quick2D_interactive"]


# http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/?in=user-97991
# used to keep track of vars useful to widgets
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__ = self


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

    current_state = Bunch()

    fig, gs = create_figure(width='single', nrows=1 + nsliders,
                            aspects=aspects, hspace=0.5)

    # draw
    ax0 = plt.subplot(gs[0])
    ax0.patch.set_facecolor("w")
    ax0.grid(b=True)

    sp_x = add_sideplot(ax0, 'x', pad=0.3)
    sp_y = add_sideplot(ax0, 'y', pad=0.3)
    line_sp_x = sp_x.plot([None], [None], visible=False, color='teal')[0]
    line_sp_y = sp_y.plot([None], [None], visible=False, color='coral')[0]
    crosshair_hline = ax0.plot([None], [None], visible=False, color='teal')[0]
    crosshair_vline = ax0.plot([None], [None], visible=False, color='coral')[0]

    current_state.xpos = crosshair_hline.get_ydata()[0]
    current_state.ypos = crosshair_vline.get_xdata()[0]

    sliders = {}
    for axis in data.axes:
        if axis not in [xaxis, yaxis]:
            slider_axes = plt.subplot(gs[len(sliders) + 1]).axes
            slider = Slider(slider_axes, axis.label,
                            0, axis.points.size - 1,
                            valinit=0, valstep=1, valfmt='%i')
            sliders[axis.natural_name] = slider

    def get_slices(sliders):
        slices = []
        for axis in data.axes:
            if axis.natural_name in sliders.keys():
                this_val = int(sliders[axis.natural_name].val)
                print(axis.natural_name, sliders[axis.natural_name].val,
                      axis.points[this_val])
                slices.append(slice(this_val, this_val + 1))
            else:
                slices.append(slice(None))
        return slices

    slices = get_slices(sliders)
    # initial xyz start are from zero indices of additional axes
    zi = channel[slices]
    zi = zi.squeeze()
    if wt_kit.get_index(data.axes, xaxis) < wt_kit.get_index(data.axes, yaxis):
        zi = zi.T.copy()
    current_state.slices = slices

    obj2D = ax0.pcolormesh(xaxis.points, yaxis.points, zi,
                           cmap=cmap, vmin=levels.min(), vmax=levels.max())

    sp_x.fill_between(xaxis.points, np.nansum(zi, axis=0), 0, color='k', alpha=0.3)
    sp_y.fill_betweenx(yaxis.points, np.nansum(zi, axis=1), 0, color='k', alpha=0.3)

    ax0.set_xlim(xaxis.points.min(), xaxis.points.max())
    ax0.set_ylim(yaxis.points.min(), yaxis.points.max())
    if channel.signed:
        sp_x.set_ylim(-1, 1)
        sp_y.set_xlim(-1, 1)

    def update_sideplot_slices():
        xlim = ax0.get_xlim()
        ylim = ax0.get_ylim()
        x0 = current_state.xpos
        y0 = current_state.ypos

        crosshair_hline.set_data(np.array([xlim, [y0, y0]]))
        crosshair_vline.set_data(np.array([[x0, x0], ylim]))

        arr = channel[current_state.slices].squeeze()

        x_temp = np.abs(data.axes[0].points - x0)
        x_index = np.argmin(x_temp)
        side_plot = arr[x_index]
        line_sp_y.set_data(side_plot / side_plot.max(), data.axes[1][:])

        y_temp = np.abs(data.axes[1].points - y0)
        y_index = np.argmin(y_temp)
        side_plot = arr[:, y_index]
        line_sp_x.set_data(data.axes[0][:, 0], side_plot / side_plot.max())

    def update(info):
        # is info a value?  then we have a slider
        # is info an object with xydata?  then we have an event
        # print(info)
        # print(type(info))
        slices = get_slices(sliders)
        #if type(info) in [int, float, np.float64]:
        if slices != current_state.slices: # a Slider moved; need to update all plots
            # sliders have changed
            arr = channel[slices].squeeze()
            current_state.slices = slices
            # TODO: check whether yaxis index is smaller (transpose not necessary)
            arr = arr.T.copy()
            # TODO: why am I stripping array information?
            # cf. https://stackoverflow.com/questions/29009743/using-set-array-with-pyplot-pcolormesh-ruins-figure
            obj2D.set_array(arr[:-1, :-1].ravel())
            sp_x.collections.clear()
            sp_y.collections.clear()
            x_proj = np.nansum(arr, axis=0)
            y_proj = np.nansum(arr, axis=1)
            sp_x.fill_between(xaxis.points, x_proj / np.abs(x_proj).max(), 0, color='k', alpha=0.3)
            sp_y.fill_betweenx(yaxis.points, y_proj / np.abs(y_proj).max(), 0, color='k', alpha=0.3)
            if line_sp_x.get_visible() and line_sp_y.get_visible():
                update_sideplot_slices()
                pass
        else:
            x0 = info.xdata
            y0 = info.ydata
            if x0 is None or y0 is None:
                print(info, info.xydata)
                raise AttributeError
            xlim = ax0.get_xlim()
            ylim = ax0.get_ylim()
            if x0 > xlim[0] and x0 < xlim[1] and y0 > ylim[0] and y0 < ylim[1]:
                current_state.xpos = info.xdata
                current_state.ypos = info.ydata
                if info.button == 1:  # left click
                    update_sideplot_slices()
                    line_sp_x.set_visible(True)
                    line_sp_y.set_visible(True)
                    crosshair_hline.set_visible(True)
                    crosshair_vline.set_visible(True)
                elif info.button == 3:  # right click
                    line_sp_x.set_visible(False)
                    line_sp_y.set_visible(False)
                    crosshair_hline.set_visible(False)
                    crosshair_vline.set_visible(False)
        fig.canvas.draw_idle()

    side_plotter = plt.matplotlib.widgets.AxesWidget(ax0)
    side_plotter.connect_event('button_release_event', update)

    for slider in sliders.values():
        slider.on_changed(update)

    return obj2D, sliders, side_plotter, crosshair_hline, crosshair_vline
