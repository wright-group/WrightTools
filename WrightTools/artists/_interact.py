"""Interactive (widget based) artists."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

from ._helpers import create_figure, plot_colorbar, add_sideplot
from ._colors import colormaps
from ..exceptions import DimensionalityError
from .. import kit as wt_kit
from .. import data as wt_data

__all__ = ["interact2D"]


# http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/?in=user-97991
# used to keep track of vars useful to widgets
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__ = self


def _at_dict(data, sliders, xaxis, yaxis):
    return {
        a.natural_name: (a[:].flat[int(sliders[a.natural_name].val)], a.units)
        for a in data.axes
        if a not in [xaxis, yaxis]
    }


def get_axes(data, axes):
    xaxis, yaxis = axes
    if type(xaxis) in [int, str]:
        xaxis = wt_kit.get_index(data.axis_names, xaxis)
        xaxis = data.axes[xaxis]
    elif type(xaxis) != wt_data.Axis:
        raise TypeError("invalid xaxis type {0}".format(type(xaxis)))
    if type(yaxis) in [int, str]:
        yaxis = wt_kit.get_index(data.axis_names, yaxis)
        yaxis = data.axes[yaxis]
    elif type(yaxis) != wt_data.Axis:
        raise TypeError("invalid xaxis type {0}".format(type(yaxis)))
    return xaxis, yaxis


def get_channel(data, channel):
    if isinstance(channel, int):
        channel = data.channels[channel]
    elif isinstance(channel, str):
        channel = [ch for ch in data.channels if ch.natural_name == channel][0]
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


def get_clim(channel, current_state):
    if current_state.local:
        arr = current_state.dat[channel.natural_name][:]
        if channel.signed:
            mag = np.nanmax(np.abs(arr))
            clim = [-mag, mag]
        else:
            clim = [0, np.nanmax(arr)]
    else:
        if channel.signed:
            clim = [-channel.mag(), channel.mag()]
        else:
            clim = [0, channel.max()]
    return clim


def gen_ticklabels(points, signed=None):
    step = np.nanmin(np.diff(points))
    if step == 0:  # zeros everywhere
        ticklabels = ['' for i in range(11)]
        if signed:
            ticklabels[5] = '0'
        else:
            ticklabels[0] = '0'
        return ticklabels
    ordinal = np.log10(np.abs(step))
    ndigits = -int(np.floor(ordinal))
    if ndigits < 0:
        ndigits += 1
        fmt = "{0:0.0f}"
    else:
        fmt = "{" + "0:.{0}f".format(ndigits) + "}"
    ticklabels = [fmt.format(round(point, ndigits)) for point in points]
    return ticklabels


def norm(arr, signed, ignore_zero=True):
    if signed:
        norm = np.nanmax(np.abs(arr))
    else:
        norm = np.nanmax(arr)
    if norm != 0 and ignore_zero:
        arr /= norm
    return arr


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
    xaxis : string, integer, or data.Axis object (optional)
        Expression or index of x axis. Default is 0.
    yaxis : string, integer, or data.Axis object (optional)
        Expression or index of y axis. Default is 1.
    channel : string, integer, or data.Channel object (optional)
        Name or index of channel to plot. Default is 0.
    local : boolean (optional)
        Toggle plotting locally. Default is False.
    verbose : boolean (optional)
        Toggle talkback. Default is True.
    """
    # avoid changing passed data object
    data = data.copy()
    # unpack
    channel = get_channel(data, channel)
    for ch in data.channel_names:
        if ch != channel.natural_name:
            data.remove_channel(ch, verbose=False)
    varis = list(data.variable_names)
    for ax in data.axes:
        for v in ax.variables:
            varis.remove(v.natural_name)
    for v in varis:
        data.remove_variable(v, implied=False, verbose=False)
    xaxis, yaxis = get_axes(data, [xaxis, yaxis])
    cmap = get_colormap(channel)
    current_state = Bunch()
    # create figure
    nsliders = data.ndim - 2
    if nsliders < 0:
        raise DimensionalityError(">= 2", data.ndim)
    # TODO: implement aspect; doesn't work currently because of our incorporation of colorbar
    fig, gs = create_figure(width="single", nrows=7 + nsliders, cols=[1, 1, 1, 1, 1, "cbar"])
    # create axes
    ax0 = plt.subplot(gs[1:6, 0:5])
    ax0.patch.set_facecolor("w")
    cax = plt.subplot(gs[1:6, -1])
    sp_x = add_sideplot(ax0, "x", pad=0.1)
    sp_y = add_sideplot(ax0, "y", pad=0.1)
    ax_local = plt.subplot(gs[0, 0], aspect="equal", frameon=False)
    ax_title = plt.subplot(gs[0, 3], frameon=False)
    ax_title.text(
        0.5,
        0.5,
        data.natural_name,
        fontsize=18,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax_title.transAxes,
    )
    ax_title.set_axis_off()
    # NOTE: there are more axes here for more buttons / widgets in future plans
    # create lines
    x_color = "#00BFBF"  # cyan with saturation increased
    y_color = "coral"
    line_sp_x = sp_x.plot([None], [None], visible=False, color=x_color)[0]
    line_sp_y = sp_y.plot([None], [None], visible=False, color=y_color)[0]
    crosshair_hline = ax0.plot([None], [None], visible=False, color=x_color)[0]
    crosshair_vline = ax0.plot([None], [None], visible=False, color=y_color)[0]
    current_state.xpos = crosshair_hline.get_ydata()[0]
    current_state.ypos = crosshair_vline.get_xdata()[0]
    current_state.bin_vs_x = True
    current_state.bin_vs_y = True
    # create buttons
    current_state.local = local
    radio = RadioButtons(ax_local, (" global", " local"))
    if local:
        radio.set_active(1)
    else:
        radio.set_active(0)
    for circle in radio.circles:
        circle.set_radius(0.14)
    # create sliders
    sliders = {}
    for axis in data.axes:
        if axis not in [xaxis, yaxis]:
            if axis.size > np.prod(axis.shape):
                raise NotImplementedError("Cannot use multivariable axis as a slider")
            slider_axes = plt.subplot(gs[~len(sliders), :]).axes
            slider = Slider(slider_axes, axis.label, 0, axis.points.size - 1, valinit=0, valstep=1)
            sliders[axis.natural_name] = slider
            slider.ax.vlines(
                range(axis.points.size - 1),
                *slider.ax.get_ylim(),
                colors="k",
                linestyle=":",
                alpha=0.5
            )
            slider.valtext.set_text(gen_ticklabels(axis.points)[0])
    # initial xyz start are from zero indices of additional axes
    current_state.dat = data.chop(
        xaxis.natural_name,
        yaxis.natural_name,
        at=_at_dict(data, sliders, xaxis, yaxis),
        verbose=False,
    )[0]
    clim = get_clim(channel, current_state)
    ticklabels = gen_ticklabels(np.linspace(*clim, 11), channel.signed)
    if clim[0] == clim[1]:
        clim = [-1 if channel.signed else 0, 1]
    obj2D = ax0.pcolormesh(
        current_state.dat,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        ylabel=yaxis.label,
        xlabel=xaxis.label,
    )
    ax0.grid(b=True)
    # colorbar
    colorbar = plot_colorbar(
        cax, cmap=cmap, label=channel.natural_name, ticks=np.linspace(clim[0], clim[1], 11)
    )
    colorbar.set_ticklabels(ticklabels)
    fig.canvas.draw_idle()

    def draw_sideplot_projections():
        arr = current_state.dat[channel.natural_name][:]
        xind = list(
            np.array(
                current_state.dat.axes[
                    current_state.dat.axis_expressions.index(xaxis.expression)
                ].shape
            )
            > 1
        ).index(True)
        yind = list(
            np.array(
                current_state.dat.axes[
                    current_state.dat.axis_expressions.index(yaxis.expression)
                ].shape
            )
            > 1
        ).index(True)
        if channel.signed:
            temp_arr = np.ma.masked_array(arr, np.isnan(arr), copy=True)
            temp_arr[temp_arr < 0] = 0
            x_proj_pos = np.nanmean(temp_arr, axis=yind)
            y_proj_pos = np.nanmean(temp_arr, axis=xind)

            temp_arr = np.ma.masked_array(arr, np.isnan(arr), copy=True)
            temp_arr[temp_arr > 0] = 0
            x_proj_neg = np.nanmean(temp_arr, axis=yind)
            y_proj_neg = np.nanmean(temp_arr, axis=xind)

            x_proj = np.nanmean(arr, axis=yind)
            y_proj = np.nanmean(arr, axis=xind)

            alpha = 0.4
            blue = "#517799"  # start with #87C7FF and change saturation
            red = "#994C4C"  # start with #FF7F7F and change saturation
            if current_state.bin_vs_x:
                x_proj_norm = max(np.nanmax(x_proj_pos), np.nanmax(-x_proj_neg))
                if x_proj_norm != 0:
                    x_proj_pos /= x_proj_norm
                    x_proj_neg /= x_proj_norm
                    x_proj /= x_proj_norm
                try:
                    sp_x.fill_between(xaxis.points, x_proj_pos, 0, color=red, alpha=alpha)
                    sp_x.fill_between(xaxis.points, 0, x_proj_neg, color=blue, alpha=alpha)
                    sp_x.fill_between(xaxis.points, x_proj, 0, color="k", alpha=0.3)
                except ValueError:  # Input passed into argument is not 1-dimensional
                    current_state.bin_vs_x = False
                    sp_x.set_visible(False)
            if current_state.bin_vs_y:
                y_proj_norm = max(np.nanmax(y_proj_pos), np.nanmax(-y_proj_neg))
                if y_proj_norm != 0:
                    y_proj_pos /= y_proj_norm
                    y_proj_neg /= y_proj_norm
                    y_proj /= y_proj_norm
                try:
                    sp_y.fill_betweenx(yaxis.points, y_proj_pos, 0, color=red, alpha=alpha)
                    sp_y.fill_betweenx(yaxis.points, 0, y_proj_neg, color=blue, alpha=alpha)
                    sp_y.fill_betweenx(yaxis.points, y_proj, 0, color="k", alpha=0.3)
                except ValueError:
                    current_state.bin_vs_y = False
                    sp_y.set_visible(False)
        else:
            if current_state.bin_vs_x:
                x_proj = np.nanmean(arr, axis=yind)
                x_proj = norm(x_proj, channel.signed)
                try:
                    sp_x.fill_between(xaxis.points, x_proj, 0, color="k", alpha=0.3)
                except ValueError:
                    current_state.bin_vs_x = False
                    sp_x.set_visible(False)
            if current_state.bin_vs_y:
                y_proj = np.nanmean(arr, axis=xind)
                y_proj = norm(y_proj, channel.signed)
                try:
                    sp_y.fill_betweenx(yaxis.points, y_proj, 0, color="k", alpha=0.3)
                except ValueError:
                    current_state.bin_vs_y = False
                    sp_y.set_visible(False)

    draw_sideplot_projections()

    ax0.set_xlim(xaxis.points.min(), xaxis.points.max())
    ax0.set_ylim(yaxis.points.min(), yaxis.points.max())

    if channel.signed:
        sp_x.set_ylim(-1.1, 1.1)
        sp_y.set_xlim(-1.1, 1.1)

    def update_sideplot_slices():
        # TODO:  if bins is only available along one axis, slicing should be valid along the other
        #   e.g., if bin_vs_y =  True, then assemble slices vs x
        #   for now, just uniformly turn off slicing
        if (not current_state.bin_vs_x) or (not current_state.bin_vs_y):
            return
        xlim = ax0.get_xlim()
        ylim = ax0.get_ylim()
        x0 = current_state.xpos
        y0 = current_state.ypos

        crosshair_hline.set_data(np.array([xlim, [y0, y0]]))
        crosshair_vline.set_data(np.array([[x0, x0], ylim]))

        at_dict = _at_dict(data, sliders, xaxis, yaxis)
        at_dict[xaxis.natural_name] = (x0, xaxis.units)
        side_plot_data = data.chop(yaxis.natural_name, at=at_dict, verbose=False)
        side_plot = side_plot_data[0][channel.natural_name].points
        side_plot = norm(side_plot, channel.signed)
        line_sp_y.set_data(side_plot, yaxis.points)
        side_plot_data.close()

        at_dict = _at_dict(data, sliders, xaxis, yaxis)
        at_dict[yaxis.natural_name] = (y0, yaxis.units)
        side_plot_data = data.chop(xaxis.natural_name, at=at_dict, verbose=False)
        side_plot = side_plot_data[0][channel.natural_name].points
        side_plot = norm(side_plot, channel.signed)
        line_sp_x.set_data(xaxis.points, side_plot)
        side_plot_data.close()

    def update_local(index):
        if verbose:
            print("normalization:", index)
        current_state.local = radio.value_selected[1:] == "local"
        clim = get_clim(channel, current_state)
        ticklabels = gen_ticklabels(np.linspace(*clim, 11), channel.signed)
        colorbar.set_ticklabels(ticklabels)
        if clim[0] == clim[1]:
            clim = [-1 if channel.signed else 0, 1]
        obj2D.set_clim(*clim)
        fig.canvas.draw_idle()

    def update(info):
        if isinstance(info, (float, int)):
            current_state.dat.close()
            current_state.dat = data.chop(
                xaxis.natural_name,
                yaxis.natural_name,
                at={
                    a.natural_name: (a[:].flat[int(sliders[a.natural_name].val)], a.units)
                    for a in data.axes
                    if a not in [xaxis, yaxis]
                },
                verbose=False,
            )[0]
            for k, s in sliders.items():
                s.valtext.set_text(
                    gen_ticklabels(data.axes[data.axis_names.index(k)].points)[int(s.val)]
                )
            obj2D.set_array(current_state.dat[channel.natural_name][:].ravel())
            clim = get_clim(channel, current_state)
            ticklabels = gen_ticklabels(np.linspace(*clim, 11), channel.signed)
            if clim[0] == clim[1]:
                clim = [-1 if channel.signed else 0, 1]
            obj2D.set_clim(*clim)
            colorbar.set_ticklabels(ticklabels)
            sp_x.collections.clear()
            sp_y.collections.clear()
            draw_sideplot_projections()
            if line_sp_x.get_visible() and line_sp_y.get_visible():
                update_sideplot_slices()
        if isinstance(info, mpl.backend_bases.MouseEvent) and info.inaxes == ax0:  # crosshairs
            x0 = info.xdata
            y0 = info.ydata
            if x0 is None or y0 is None:
                raise TypeError(info)
            xlim = ax0.get_xlim()
            ylim = ax0.get_ylim()
            if x0 > xlim[0] and x0 < xlim[1] and y0 > ylim[0] and y0 < ylim[1]:
                current_state.xpos = info.xdata
                current_state.ypos = info.ydata
                if info.button == 1:  # left click
                    if verbose:
                        print(current_state.xpos, current_state.ypos)
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

    radio.on_clicked(update_local)

    for slider in sliders.values():
        slider.on_changed(update)

    return obj2D, sliders, side_plotter, crosshair_hline, crosshair_vline, radio, colorbar
