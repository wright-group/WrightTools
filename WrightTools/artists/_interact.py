"""Interactive (widget based) artists."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from types import SimpleNamespace

from ._helpers import create_figure, plot_colorbar, add_sideplot
from ._base import _order_for_imshow
from ._colors import colormaps
from ..exceptions import DimensionalityError
from .. import kit as wt_kit
from .. import data as wt_data

__all__ = ["interact2D"]


class Focus:
    def __init__(self, axes, sliders, linewidth=2):
        self.axes = axes
        self.sliders = sliders
        self.linewidth = linewidth
        ax = axes[0]
        for side in ["top", "bottom", "left", "right"]:
            ax.spines[side].set_linewidth(self.linewidth)
        self.focus_axis = ax

    def __call__(self, ax):
        if type(ax) == str:
            ind = self.axes.index(self.focus_axis)
            if ax == "next":
                ind -= 1
            elif ax == "previous":
                ind += 1
            ax = self.axes[ind % len(self.axes)]
        if self.focus_axis == ax or ax not in self.axes:
            return
        else:  # set new focus
            if self.focus_axis.get_gid() in self.sliders.keys():
                self.sliders[self.focus_axis.get_gid()].track.set_facecolor("lightgrey")
            if ax.get_gid() in self.sliders.keys():
                self.sliders[ax.get_gid()].track.set_facecolor("darkgrey")

            for spine in ["top", "bottom", "left", "right"]:
                self.focus_axis.spines[spine].set_linewidth(1)
                ax.spines[spine].set_linewidth(self.linewidth)
            self.focus_axis = ax


def _at_dict(data, sliders, xaxis, yaxis):
    return {
        a.natural_name: (a[:].flat[int(sliders[a.natural_name].val)], a.units)
        for a in data.axes
        if a not in [xaxis, yaxis]
    }


def create_local_global_radio(ax, local):
    if mpl.__version_info__ >= (3, 7):
        radio = RadioButtons(ax, (" global", " local"), radio_props={"s": 100})
    else:
        radio = RadioButtons(ax, (" global", " local"))
        for circle in radio.circles:
            circle.set_radius(0.14)
    if local:
        radio.set_active(1)
    else:
        radio.set_active(0)
    return radio


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


def get_colormap(signed):
    cmap = "signed" if signed else "default"
    cmap = colormaps[cmap]
    cmap.set_bad([0.75] * 3, 1.0)
    cmap.set_under([0.75] * 3, 1.0)
    return cmap


class Norm:
    def __init__(self, channel, current_state):
        self.current_state = current_state
        self.signed = channel.signed
        self.update(channel)

    def __call__(self, data):
        out = self.norm(data)
        return out

    def update(self, channel):
        if self.signed:
            if not self.current_state.local:
                norm = mpl.colors.CenteredNorm(vcenter=channel.null, halfrange=channel.mag())
            else:
                norm = mpl.colors.CenteredNorm(vcenter=channel.null)
                norm.autoscale_None(
                    np.ma.masked_invalid(self.current_state.dat[channel.natural_name][:])
                )
            if norm.halfrange == 0:
                norm.halfrange = 1
        else:
            if not self.current_state.local:
                norm = mpl.colors.Normalize(vmin=channel.null, vmax=np.nanmax(channel[:]))
            else:
                norm = mpl.colors.Normalize(vmin=channel.null)
                norm.autoscale_None(
                    np.ma.masked_invalid(self.current_state.dat[channel.natural_name][:])
                )
            if norm.vmax == norm.vmin:
                norm.vmax += 1
        self.norm = norm

    @property
    def ticks(self) -> np.array:
        if type(self.norm) == mpl.colors.CenteredNorm:
            vmin = self.norm.vcenter - self.norm.halfrange
            vmax = self.norm.vcenter + self.norm.halfrange
        else:  # mpl.colors.Normalize
            vmin = self.norm.vmin
            vmax = self.norm.vmax
        return np.linspace(vmin, vmax, 11)


def gen_ticklabels(points, signed=None):
    step = np.nanmin(np.diff(points))
    if step == 0:  # zeros everywhere
        ticklabels = ["" for i in range(11)]
        if signed:
            ticklabels[5] = "0"
        else:
            ticklabels[0] = "0"
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


def interact2D(
    data: wt_data.Data,
    xaxis=0,
    yaxis=1,
    channel=0,
    cmap=None,
    local=False,
    use_imshow=False,
    verbose=True,
):
    """Interactive 2D plot of the dataset.
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
    cmap : string or cm object (optional)
        Name of colormap, or explicit colormap object.  Defaults to channel default.
    local : boolean (optional)
        Toggle plotting locally. Default is False.
    use_imshow : boolean (optional)
        If True, matplotlib imshow is used to render the 2D slice.
        Can give better performance, but is only accurate for
        uniform grids.  Default is False.
    verbose : boolean (optional)
        Toggle talkback. Default is True.
    """
    # avoid changing passed data object
    data = data.copy()
    # unpack
    channel = get_channel(data, channel)
    xaxis, yaxis = get_axes(data, [xaxis, yaxis])
    data.prune(keep_channels=channel.natural_name, verbose=False)
    cmap = cmap if cmap is not None else get_colormap(channel.signed)
    current_state = SimpleNamespace()
    # create figure
    nsliders = data.ndim - 2
    if nsliders < 0:
        raise DimensionalityError(">= 2", data.ndim)
    # TODO: implement aspect; doesn't work currently because of our incorporation of colorbar
    fig, gs = create_figure(width="single", nrows=7 + nsliders, cols=[1, 1, 1, 1, 1, "cbar"])
    plt.get_current_fig_manager().set_window_title(f"interact2D: {data.natural_name}")
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
    x_color = "#00BFBF"  # cyan with increased saturation
    y_color = "coral"
    line_sp_x = sp_x.plot([None], [None], visible=False, color=x_color, linewidth=2)[0]
    line_sp_y = sp_y.plot([None], [None], visible=False, color=y_color, linewidth=2)[0]
    crosshair_hline = ax0.plot([None], [None], visible=False, color=x_color, linewidth=2)[0]
    crosshair_vline = ax0.plot([None], [None], visible=False, color=y_color, linewidth=2)[0]
    current_state.xarg = xaxis.points.flatten().size // 2
    current_state.yarg = yaxis.points.flatten().size // 2
    xdir = 1 if xaxis.points.flatten()[-1] - xaxis.points.flatten()[0] > 0 else -1
    ydir = 1 if yaxis.points.flatten()[-1] - yaxis.points.flatten()[0] > 0 else -1
    current_state.bin_vs_x = True
    current_state.bin_vs_y = True

    # create buttons
    current_state.local = local
    radio = create_local_global_radio(ax_local, local)

    # create sliders
    sliders = {}
    for axis in filter(lambda a: a not in [xaxis, yaxis], data.axes):
        if axis.size > np.prod(axis.shape):
            raise NotImplementedError("Cannot use multivariable axis as a slider")
        slider_axes = plt.subplot(gs[~len(sliders), :]).axes
        slider = Slider(
            slider_axes,
            axis.label,
            0,
            axis.points.size - 1,
            valinit=0,
            valstep=1,
            track_color="lightgrey",
        )
        sliders[axis.natural_name] = slider
        slider_axes.set_gid(axis.natural_name)
        slider.ax.vlines(
            range(axis.points.size - 1),
            *slider.ax.get_ylim(),
            colors="k",
            linestyle=":",
            alpha=0.5,
        )
        slider.valtext.set_text(gen_ticklabels(axis.points)[0])
    current_state.focus = Focus([ax0] + [slider.ax for slider in sliders.values()], sliders)
    # initial xyz start are from zero indices of additional axes
    current_state.dat = data.at(**_at_dict(data, sliders, xaxis, yaxis))
    current_state.dat.transform(xaxis.expression, yaxis.expression)
    current_state.norm = Norm(channel, current_state)

    gen_mesh = ax0.pcolormesh if not use_imshow else ax0.imshow
    obj2D = gen_mesh(
        current_state.dat,
        cmap=cmap,
        norm=current_state.norm.norm,
        ylabel=yaxis.label,
        xlabel=xaxis.label,
    )
    ax0.grid(True)
    # colorbar
    ticks = current_state.norm.ticks
    ticklabels = gen_ticklabels(ticks, channel.signed)
    colorbar = plot_colorbar(cax, cmap=cmap, label=channel.natural_name, ticks=ticks)
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

        norm = current_state.norm

        if channel.signed:
            temp_arr = np.ma.masked_array(arr, np.isnan(arr), copy=True)
            temp_arr[temp_arr < 0] = 0
            x_proj_pos = np.nanmax(temp_arr, axis=yind)
            y_proj_pos = np.nanmax(temp_arr, axis=xind)

            temp_arr = np.ma.masked_array(arr, np.isnan(arr), copy=True)
            temp_arr[temp_arr > 0] = 0
            x_proj_neg = np.nanmin(temp_arr, axis=yind)
            y_proj_neg = np.nanmin(temp_arr, axis=xind)

            x_proj = np.nanmean(arr, axis=yind)
            y_proj = np.nanmean(arr, axis=xind)

            alpha = 0.4
            blue = "#517799"  # start with #87C7FF and change saturation
            red = "#994C4C"  # start with #FF7F7F and change saturation

            if current_state.bin_vs_x:
                try:
                    sp_x.fill_between(xaxis.points, norm(x_proj_pos), 0.5, color=red, alpha=alpha)
                    sp_x.fill_between(xaxis.points, 0.5, norm(x_proj_neg), color=blue, alpha=alpha)
                    sp_x.fill_between(xaxis.points, norm(x_proj), 0.5, color="k", alpha=0.3)
                except ValueError:  # Input passed into argument is not 1-dimensional
                    current_state.bin_vs_x = False
                    sp_x.set_visible(False)
            if current_state.bin_vs_y:
                try:
                    sp_y.fill_betweenx(yaxis.points, norm(y_proj_pos), 0.5, color=red, alpha=alpha)
                    sp_y.fill_betweenx(
                        yaxis.points, 0.5, norm(y_proj_neg), color=blue, alpha=alpha
                    )
                    sp_y.fill_betweenx(yaxis.points, norm(y_proj), 0.5, color="k", alpha=0.3)
                except ValueError:
                    current_state.bin_vs_y = False
                    sp_y.set_visible(False)
        else:
            if current_state.bin_vs_x:
                x_proj = np.nanmax(arr, axis=yind)
                try:
                    sp_x.fill_between(xaxis.points, norm(x_proj), 0, color="k", alpha=0.3)
                except ValueError:
                    current_state.bin_vs_x = False
                    sp_x.set_visible(False)
            if current_state.bin_vs_y:
                y_proj = np.nanmax(arr, axis=xind)
                try:
                    sp_y.fill_betweenx(yaxis.points, norm(y_proj), 0, color="k", alpha=0.3)
                except ValueError:
                    current_state.bin_vs_y = False
                    sp_y.set_visible(False)

    draw_sideplot_projections()

    ax0.set_xlim(xaxis.points.min(), xaxis.points.max())
    ax0.set_ylim(yaxis.points.min(), yaxis.points.max())

    sp_x.set_ylim(0, 1)
    sp_y.set_xlim(0, 1)

    def update_sideplot_slices():
        # TODO:  if bins is only available along one axis, slicing should be valid along the other
        #   e.g., if bin_vs_y =  True, then assemble slices vs x
        #   for now, just uniformly turn off slicing
        if (not current_state.bin_vs_x) or (not current_state.bin_vs_y):
            return
        xlim = ax0.get_xlim()
        ylim = ax0.get_ylim()
        x0 = xaxis.points[current_state.xarg]
        y0 = yaxis.points[current_state.yarg]

        crosshair_hline.set_data(np.array([xlim, [y0, y0]]))
        crosshair_vline.set_data(np.array([[x0, x0], ylim]))

        at_dict = _at_dict(data, sliders, xaxis, yaxis)
        at_dict[xaxis.natural_name] = (x0, xaxis.units)
        side_plot_data = data.at(**at_dict)
        side_plot = side_plot_data[channel.natural_name].points
        side_plot = current_state.norm(side_plot)
        line_sp_y.set_data(side_plot, yaxis.points)
        side_plot_data.close()

        at_dict = _at_dict(data, sliders, xaxis, yaxis)
        at_dict[yaxis.natural_name] = (y0, yaxis.units)
        side_plot_data = data.at(**at_dict)
        side_plot = side_plot_data[channel.natural_name].points
        side_plot = current_state.norm(side_plot)
        line_sp_x.set_data(xaxis.points, side_plot)
        side_plot_data.close()

    def update_local(index):
        if verbose:
            print("normalization:", index)
        current_state.local = radio.value_selected[1:] == "local"
        current_state.norm.update(channel)
        obj2D.set_norm(current_state.norm.norm)
        ticklabels = gen_ticklabels(current_state.norm.ticks, channel.signed)
        colorbar.set_ticklabels(ticklabels)

        update_sideplots(sp_x, sp_y, line_sp_x, line_sp_y)

        fig.canvas.draw_idle()

    def update_slider(info, use_imshow=use_imshow):
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
        if use_imshow:
            transpose = _order_for_imshow(
                current_state.dat[xaxis.natural_name][:],
                current_state.dat[yaxis.natural_name][:],
            )
            obj2D.set_data(current_state.dat[channel.natural_name][:].transpose(transpose))
        else:
            obj2D.set_array(current_state.dat[channel.natural_name][:].ravel())
        current_state.norm.update(channel)
        obj2D.set_norm(current_state.norm.norm)

        ticks = current_state.norm.ticks
        ticklabels = gen_ticklabels(ticks, channel.signed)
        colorbar.set_ticklabels(ticklabels)

        update_sideplots(sp_x, sp_y, line_sp_x, line_sp_y)
        fig.canvas.draw_idle()

    def update_sideplots(sp_x, sp_y, line_sp_x, line_sp_y):
        [item.remove() for item in sp_x.collections]
        [item.remove() for item in sp_y.collections]
        if len(sp_x.collections) > 0:  # mpl < 3.7
            sp_x.collections.clear()
            sp_y.collections.clear()

        draw_sideplot_projections()
        if line_sp_x.get_visible() and line_sp_y.get_visible():
            update_sideplot_slices()

    def update_crosshairs(xarg, yarg, hide=False):
        # find closest x and y pts in dataset
        current_state.xarg = xarg
        current_state.yarg = yarg
        xedge = xarg in [0, xaxis.points.flatten().size - 1]
        yedge = yarg in [0, yaxis.points.flatten().size - 1]
        current_state.xpos = xaxis.points[xarg]
        current_state.ypos = yaxis.points[yarg]
        if not hide:  # update crosshairs and show
            if verbose:
                print(current_state.xpos, current_state.ypos)
            update_sideplot_slices()
            line_sp_x.set_visible(True)
            line_sp_y.set_visible(True)
            crosshair_hline.set_visible(True)
            crosshair_vline.set_visible(True)
            # thicker lines if on the axis edges
            crosshair_vline.set_linewidth(6 if xedge else 2)
            crosshair_hline.set_linewidth(6 if yedge else 2)
        else:  # do not update and hide crosshairs
            line_sp_x.set_visible(False)
            line_sp_y.set_visible(False)
            crosshair_hline.set_visible(False)
            crosshair_vline.set_visible(False)

    def update_button_release(info):
        # mouse button release
        current_state.focus(info.inaxes)
        if info.inaxes == ax0:
            xlim = ax0.get_xlim()
            ylim = ax0.get_ylim()
            x0, y0 = info.xdata, info.ydata
            if x0 > xlim[0] and x0 < xlim[1] and y0 > ylim[0] and y0 < ylim[1]:
                xarg = np.abs(xaxis.points - x0).argmin()
                yarg = np.abs(yaxis.points - y0).argmin()
                if info.button == 1 or info.button is None:  # left click
                    update_crosshairs(xarg, yarg)
                elif info.button == 3:  # right click
                    update_crosshairs(xarg, yarg, hide=True)
        fig.canvas.draw_idle()

    def update_key_press(info):
        if info.key in ["left", "right", "up", "down"]:
            if current_state.focus.focus_axis != ax0:  # sliders
                if info.key in ["up", "down"]:
                    return
                slider = [
                    slider
                    for slider in sliders.values()
                    if slider.ax == current_state.focus.focus_axis
                ][0]
                new_val = slider.val + 1 if info.key == "right" else slider.val - 1
                new_val %= slider.valmax + 1
                slider.set_val(new_val)
            else:  # crosshairs
                dx = dy = 0
                if info.key == "left":
                    dx -= 1
                elif info.key == "right":
                    dx += 1
                elif info.key == "up":
                    dy += 1
                elif info.key == "down":
                    dy -= 1
                update_crosshairs(
                    (current_state.xarg + dx * xdir) % xaxis.points.flatten().size,
                    (current_state.yarg + dy * ydir) % yaxis.points.flatten().size,
                )
        elif info.key == "tab":
            current_state.focus("next")
        elif info.key == "ctrl+tab":
            current_state.focus("previous")
        else:
            mpl.backend_bases.key_press_handler(info, fig.canvas, fig.canvas.toolbar)
        fig.canvas.draw_idle()

    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect("button_release_event", update_button_release)
    fig.canvas.mpl_connect("key_press_event", update_key_press)
    radio.on_clicked(update_local)

    for slider in sliders.values():
        slider.on_changed(update_slider)

    return obj2D, sliders, crosshair_hline, crosshair_vline, radio, colorbar
