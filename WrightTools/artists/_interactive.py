"""Interactive (widget based) artists."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from ._helpers import create_figure, plot_colorbar, savefig, add_sideplot
from ._colors import colormaps
from .. import kit as wt_kit
from .. import data as wt_data

__all__ = ["interact2D"]


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
        raise TypeError("invalid xaxis type {0}".format(type(xaxis)))
    if type(yaxis) in [int, str]:
        yaxis = data.axes[yaxis]
    elif type(yaxis) != wt_data.Axis:
        raise TypeError("invalid xaxis type {0}".format(type(yaxis)))
    return xaxis, yaxis


def get_channel(data, channel):
    if type(channel) in [int, str]:
        channel = data.channels[channel]
    elif type(channel) != wt_data.Channel:
        raise TypeError("invalid channel type {0}".format(type(channel)))
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


def get_levels(channel, local):
    if local:
        raise NotImplementedError
    if channel.signed:
        levels = np.linspace(-channel.mag(), channel.mag(), 200)
    else:
        levels = np.linspace(0, channel.max(), 200)
    return levels


def get_slices(sliders, axes, verbose=False):
    slices = []
    for axis in axes:
        if axis.natural_name in sliders.keys():
            this_val = int(sliders[axis.natural_name].val)
            ndigits = int(infer_precision(axis))
            text = '{0}'.format(round(axis.points[this_val], ndigits))
            sliders[axis.natural_name].valtext.set_text(text)
            if verbose:
                print(axis.natural_name, sliders[axis.natural_name].val, text)
            slices.append(slice(this_val, this_val + 1))
        else:
            slices.append(slice(None))
    return slices


def infer_precision(axis):
    step = np.diff(axis.points).min()
    ordinal = np.log10(np.abs(step))
    return -np.floor(ordinal)


def norm(arr, signed, ignore_zero=True):
    if signed:
        norm = np.nanmax(np.abs(arr))
    else:
        norm = np.nanmax(arr)
    if norm != 0 and ignore_zero:
        arr /= norm
    return arr


def set_aspect(xaxis, yaxis):
    if xaxis.units == yaxis.units:
        xr = xaxis.max() - xaxis.min()
        yr = yaxis.max() - yaxis.min()
        aspect = np.abs(yr / xr)
        if 3 < aspect or aspect < 1 / 3.:
            raise Warning(
                "units agree, but aspect {0} required for equal spacing is too"
                / + "narrow.".format(aspect)
            )
            aspect = np.clip(aspect, 1 / 3., 3.)
            print("using aspect {0} instead".format(aspect))
    else:
        aspect = 1
    return aspect


def interact2D(data, xaxis=0, yaxis=1, channel=0, local=False, verbose=True):
    """ Interactive 2D plot of the dataset.
    Side plots show x and y projections of the slice (shaded gray).
    Left clicks on the main axes draw 1D slices on side plots at the coordinates selected.
    Right clicks remove the 1D slices.
    For 3+ dimensional data, sliders below the main axes are used to change which slice is viewed.

    Parameters
    ----------
    data : WrightTools.Data object
        Data to plot.
    axis : string or integer (optional)
        Expression or index of axis. Default is 0.
    channel : string or integer (optional)
        Name or index of channel to plot. Default is 0.
    local : boolean (optional)
        Toggle plotting locally. Default is False.
    verbose : boolean (optional)
        Toggle talkback. Default is True.
    """
    # unpack
    channel = get_channel(data, channel)
    xaxis, yaxis = get_axes(data, [xaxis, yaxis])
    cmap = get_colormap(channel)
    levels = get_levels(channel, local)
    current_state = Bunch()
    current_state.local = local
    # create figure
    nsliders = data.ndim - 2
    if nsliders < 0:
        print("note enough dimensions")
        return
    # TODO: implement aspect again; doesn't work because of our incorporation of colorbar
    fig, gs = create_figure(width="single", nrows=7 + nsliders, cols=[1, 1, 1, 1, 1, 'cbar'])
    # create axes
    ax0 = plt.subplot(gs[1:6, 0:5])
    ax0.patch.set_facecolor("w")
    ax0.grid(b=True)
    cax = plt.subplot(gs[1:6, -1])
    sp_x = add_sideplot(ax0, "x", pad=0.1)
    sp_y = add_sideplot(ax0, "y", pad=0.1)
    ax_local = plt.subplot(gs[0, 0])
    # NOTE: there are more axes here for more buttons / widgets in future plans
    # create lines
    line_sp_x = sp_x.plot([None], [None], visible=False, color="teal")[0]
    line_sp_y = sp_y.plot([None], [None], visible=False, color="coral")[0]
    crosshair_hline = ax0.plot([None], [None], visible=False, color="teal")[0]
    crosshair_vline = ax0.plot([None], [None], visible=False, color="coral")[0]
    current_state.xpos = crosshair_hline.get_ydata()[0]
    current_state.ypos = crosshair_vline.get_xdata()[0]
    # create buttons
    button_local = Button(ax_local, label="global")
    # create sliders
    sliders = {}
    for axis in data.axes:
        if axis not in [xaxis, yaxis]:
            slider_axes = plt.subplot(gs[~len(sliders), :]).axes
            slider = Slider(
                slider_axes, axis.label, 0, axis.points.size - 1, valinit=0, valstep=1, valfmt="%i"
            )
            sliders[axis.natural_name] = slider
    # initial xyz start are from zero indices of additional axes
    slices = get_slices(sliders, data.axes, verbose=verbose)
    zi = channel[slices]
    zi = zi.squeeze()
    if wt_kit.get_index(data.axes, xaxis) < wt_kit.get_index(data.axes, yaxis):
        zi = zi.T.copy()
    current_state.slices = slices
    # TODO: should we use pcolormesh or pcolor?
    obj2D = ax0.pcolormesh(
        xaxis.points, yaxis.points, zi, cmap=cmap, vmin=levels.min(), vmax=levels.max()
    )
    ax0.set_xlabel(xaxis.label)
    ax0.set_ylabel(yaxis.label)
    # colorbar
    plot_colorbar(cax, cmap=cmap, label=channel.label,
                  ticks=np.linspace(levels.min(), levels.max(), 11))

    def draw_sideplot_projections(arr):
        if channel.signed:
            colors = plt.cm.coolwarm(np.linspace(0,1,2))
            alpha = 0.2
            temp_arr = np.ma.masked_array(arr, np.isnan(arr), copy=True)
            temp_arr[temp_arr < 0] = 0
            x_proj_pos = np.nanmean(temp_arr, axis=0)
            y_proj_pos = np.nanmean(temp_arr, axis=1)

            temp_arr = np.ma.masked_array(arr, np.isnan(arr), copy=True)
            temp_arr[temp_arr > 0] = 0
            x_proj_neg = np.nanmean(temp_arr, axis=0)
            y_proj_neg = np.nanmean(temp_arr, axis=1)
            
            x_proj = np.nanmean(arr, axis=0)
            y_proj = np.nanmean(arr, axis=1)

            x_proj_norm = max(np.nanmax(x_proj_pos), np.nanmax(-x_proj_neg))
            x_proj_pos /= x_proj_norm
            x_proj_neg /= x_proj_norm
            x_proj /= x_proj_norm

            y_proj_norm = max(np.nanmax(y_proj_pos), np.nanmax(-y_proj_neg))
            y_proj_pos /= y_proj_norm
            y_proj_neg /= y_proj_norm
            y_proj /= y_proj_norm

            sp_x.fill_between(xaxis.points, x_proj_pos, 0, color='k', alpha=alpha)
            sp_x.fill_between(xaxis.points, 0, x_proj_neg, color='k', alpha=alpha)
            sp_x.fill_between(xaxis.points, x_proj, 0, color='k', alpha=0.3)

            sp_y.fill_betweenx(yaxis.points, y_proj_pos, 0, color='k', alpha=alpha)
            sp_y.fill_betweenx(yaxis.points, 0, y_proj_neg, color='k', alpha=alpha)
            sp_y.fill_betweenx(yaxis.points, y_proj, 0, color='k', alpha=alpha)

        else:
            x_proj = np.nansum(arr, axis=0)
            y_proj = np.nansum(arr, axis=1)
            x_proj = norm(x_proj, channel.signed)
            y_proj = norm(y_proj, channel.signed)
            sp_x.fill_between(xaxis.points, x_proj, 0, color="k", alpha=0.3)
            sp_y.fill_betweenx(yaxis.points, y_proj, 0, color="k", alpha=0.3)

    draw_sideplot_projections(zi)

    ax0.set_xlim(xaxis.points.min(), xaxis.points.max())
    ax0.set_ylim(yaxis.points.min(), yaxis.points.max())

    if channel.signed:
        sp_x.set_ylim(-1.1, 1.1)
        sp_y.set_xlim(-1.1, 1.1)

    def update_sideplot_slices():
        xlim = ax0.get_xlim()
        ylim = ax0.get_ylim()
        x0 = current_state.xpos
        y0 = current_state.ypos

        crosshair_hline.set_data(np.array([xlim, [y0, y0]]))
        crosshair_vline.set_data(np.array([[x0, x0], ylim]))

        arr = channel[current_state.slices].squeeze()
        if wt_kit.get_index(data.axes, xaxis) < wt_kit.get_index(data.axes, yaxis):
            arr = arr.T.copy()

        x_temp = np.abs(xaxis.points - x0)
        x_index = np.argmin(x_temp)
        side_plot = arr[:, x_index].copy()
        side_plot = norm(side_plot, channel.signed)
        line_sp_y.set_data(side_plot, yaxis.points)

        y_temp = np.abs(yaxis.points - y0)
        y_index = np.argmin(y_temp)
        side_plot = arr[y_index].copy()
        side_plot = norm(side_plot, channel.signed)
        line_sp_x.set_data(xaxis.points, side_plot)

    def update(info):
        # is info a value?  then we have a slider
        # is info an object with xydata?  then we have an event
        slices = get_slices(sliders, data.axes, verbose=verbose)
        if slices != current_state.slices:  # a Slider moved; need to update all plot objects
            arr = channel[slices].squeeze()
            current_state.slices = slices
            if wt_kit.get_index(data.axes, xaxis) < wt_kit.get_index(data.axes, yaxis):
                arr = arr.T.copy()
            # TODO: why am I stripping off array information?
            # cf. https://stackoverflow.com/questions/29009743
            obj2D.set_array(arr[:-1, :-1].ravel())
            sp_x.collections.clear()
            sp_y.collections.clear()
            draw_sideplot_projections(arr)
            if line_sp_x.get_visible() and line_sp_y.get_visible():
                update_sideplot_slices()
                pass
        elif info.inaxes == ax_local:
            if button_local.label.get_text() == "global":
                button_local.label.set_text("local")
            elif button_local.label.get_text() == "local":
                button_local.label.set_text("global")
        elif info.inaxes == ax0:  # crosshairs
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
    side_plotter.connect_event("button_release_event", update)

    button_local.connect_event("button_release_event", update)

    for slider in sliders.values():
        slider.on_changed(update)

    return obj2D, sliders, side_plotter, crosshair_hline, crosshair_vline
