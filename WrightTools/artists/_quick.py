"""Quick plotting."""

# --- import --------------------------------------------------------------------------------------


from contextlib import closing
from functools import reduce
import pathlib

import numpy as np

import matplotlib.pyplot as plt

from ._helpers import (
    _title,
    create_figure,
    plot_colorbar,
    savefig,
    set_ax_labels,
    norm_from_channel,
    ticks_from_norm
)
from ._base import _parse_cmap
from ._interact import Norm
from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


__all__ = ["quick1D", "quick2D", "QuicknD"]


# --- general purpose plotting functions ----------------------------------------------------------


class QuicknD:
    """class for keeping track of plotting through the chopped data"""

    def __init__(self, data, *axes, **kwargs):
        self.data = data
        self.axes = axes
        self.at = kwargs.get("at", {})
        self.nD = len(axes)

        # TODO: overload plot method and use these attrs elsewhere
        self.cmap = kwargs.get("cmap", None)
        self.contours = kwargs.get("contours", 0)
        self.contours_local = kwargs.get("contours_local", False)
        self.autosave = kwargs.get("autosave", False)
        self.dynamic_range = kwargs.get("dynamic_range", False)
        self.local = kwargs.get("local", False)
        self.pixelated = kwargs.get("pixelated", True)

        self.channel_index = wt_kit.get_index(data.channel_names, kwargs.get("channel", 0))
        self._global_limits = None
        shape = data.channels[self.channel_index].shape
        # remove dimensions that do not involve the channel
        self.channel_slice = [0 if size == 1 else slice(None) for size in shape]
        self.sliced_constants = [
            data.axis_expressions[i] for i in range(len(shape)) if not self.channel_slice[i]
        ]
        removed_shape = data._chop_prep(*self.axes, at=self.at)[0]

        len_chopped = reduce(int.__mul__, removed_shape) // reduce(int.__mul__, shape)
        if len_chopped > 10 and not self.autosave:
            print(f"expecting {len_chopped} figures.  Forcing autosave.")
            self.autosave = True
        if self.autosave:
            self.save_directory, self.filepath_seed = _filepath_seed(
                kwargs.get("save_directory", pathlib.Path.cwd()),
                kwargs.get("fname", data.natural_name),
                len_chopped,
                f"quick{self.nD}D",
            )
            pathlib.Path.mkdir(self.save_directory)

    def __call__(self, verbose=False) -> list:
        if self.nD == 1:
            plot = self.plot1D
        elif self.nD == 2:
            plot = self.plot2D
        out = list()
        with closing(self.data._from_slice(self.channel_slice)) as sliced:
            for constant in self.sliced_constants:
                sliced.remove_constant(constant)
            for i, fig in enumerate(map(plot, sliced.ichop(*self.axes, at=self.at))):
                if self.autosave:
                    filepath = self.filepath_seed.format(i)
                    savefig(filepath, fig=fig, facecolor="white")
                    plt.close(fig)
                    if verbose:
                        print("image saved at", str(filepath))
                    out.append(str(filepath))
                else:
                    out.append(fig)
        return out

    # plot functions can be overloaded to make for custom
    def plot2D(self, d):
        kwargs = {}
        if self.cmap is not None:
            kwargs["cmap"] = self.cmap
        # unpack data -------------------------------------------------------------------------
        xaxis = d.axes[0]
        yaxis = d.axes[1]
        channel = d.channels[self.channel_index]
        # create figure -----------------------------------------------------------------------
        if xaxis.units == yaxis.units:
            xr = xaxis.max() - xaxis.min()
            yr = yaxis.max() - yaxis.min()
            aspect = np.abs(yr / xr)
            if 3 < aspect or aspect < 1 / 3.0:
                # TODO: raise warning here
                aspect = np.clip(aspect, 1 / 3.0, 3.0)
        else:
            aspect = 1
        fig, gs = create_figure(
            width="single", nrows=1, cols=[1, "cbar"], aspects=[[[0, 0], aspect]]
        )
        ax = plt.subplot(gs[0])
        ax.patch.set_facecolor("w")
        # colors ------------------------------------------------------------------------------
        norm = norm_from_channel(channel if self.local else self.data.channels[self.channel_index])
        norm_ticks = ticks_from_norm(norm)
        if self.pixelated:
            img = ax.pcolor(
                d, channel=self.channel_index, norm=norm, **kwargs
            )
        else:
            img = ax.contourf(d, channel=self.channel_index, norm=norm, **kwargs)
        # contour lines -----------------------------------------------------------------------
        if self.contours:
            contour_levels = determine_contour_levels(
                channel, self.data.channels[self.channel_index], self.contours_local
            )
            ax.contour(d, channel=self.channel_index, levels=contour_levels)
        # decoration --------------------------------------------------------------------------
        self.decorate(ax, *d.axes)
        # constants: variable marker lines, title
        subtitle = self.annotate_constants(d)
        _title(fig, self.data.natural_name, subtitle=subtitle)
        # colorbar
        cax = plt.subplot(gs[1])
        plot_colorbar(
            cax=cax, cmap=img.get_cmap(), ticks=norm_ticks, label=channel.natural_name, **kwargs
        )
        plt.sca(ax)
        return fig

    def plot1D(self, d):
        # determine ymin and ymax for global axis scale
        # unpack data -------------------------------------------------------------------------
        axis = d.axes[0]
        channel = d.channels[self.channel_index]
        # create figure ------------------------------------------------------------------------
        aspects = [[[0, 0], 0.5]]
        fig, gs = create_figure(width="single", nrows=1, cols=[1], aspects=aspects)
        ax = plt.subplot(gs[0, 0])
        # plot --------------------------------------------------------------------------------
        ax.plot(axis.full, channel[:], lw=2)
        ax.scatter(axis.full, channel[:], color="grey", alpha=0.5, edgecolor="none")
        # decoration --------------------------------------------------------------------------
        self.decorate(ax, *d.axes)
        # constants: variable marker lines, title
        subtitle = self.annotate_constants(d)
        _title(fig, self.data.natural_name, subtitle=subtitle)
        return fig

    def annotate_constants(self, d):
        ls = []
        for c in d.constants:
            if c.units:
                ls.append(c.label)
                # x axis
                if d.axes[0].units_kind == c.units_kind:
                    c.convert(d.axes[0].units)
                    plt.axvline(c.value, color="k", linewidth=4, alpha=0.25)
                # y axis
                if self.nD == 2 and (d.axes[1].units_kind == c.units_kind):
                    c.convert(d.axes[1].units)
                    plt.axhline(c.value, color="k", linewidth=4, alpha=0.25)
        return ", ".join(ls)

    def global_limits(self):
        if self._global_limits is not None:
            return self._global_limits
        if self.nD == 1:
            data_channel = self.data.channels[self.channel_index]
            ymin, ymax = data_channel.min(), data_channel.max()
            dynamic_range = ymax - ymin
            ymin -= dynamic_range * 0.05
            ymax += dynamic_range * 0.05
            if np.sign(ymin) != np.sign(data_channel.min()):
                ymin = 0
            if np.sign(ymax) != np.sign(data_channel.max()):
                ymax = 0
            limits = [ymin, ymax]
        elif self.nD == 2:
            channel = self.data.channels[self.channel_index]
            if self.data.signed:
                if self.dynamic_range:
                    limit = min(
                        abs(channel.null - channel.min()),
                        abs(channel.null - channel.max())
                    )
                else:
                    limit = channel.mag()
                limits = [-limit + channel.null, limit + channel.null]
            else:
                if channel.max() < channel.null:
                    limits = [channel.min(), channel.null]
                else:
                    limits = [channel.null, channel.max()]
        return limits

    def decorate(self, ax, *axes):
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        ax.axvline(0, lw=2, c="k")
        ax.axhline(0, lw=2, c="k")
        ax.set_xlim(axes[0].min(), axes[0].max())
        ax.grid()
        if self.nD == 1:
            set_ax_labels(ax, xlabel=axes[0].label, ylabel=self.data.natural_name)
            if not self.local:
                ax.set_ylim(*self.global_limits())
        elif self.nD == 2:
            set_ax_labels(ax, xlabel=axes[0].label, ylabel=axes[1].label)
            ax.set_ylim(axes[1].min(), axes[1].max())


def quick1D(
    data,
    axis=0,
    at={},
    channel=0,
    *,
    local=False,
    autosave=False,
    save_directory=None,
    fname=None,
    verbose=True,
):
    """Quickly plot 1D slice(s) of data.

    Parameters
    ----------
    data : WrightTools.Data object
        Data to plot.
    axis : string or integer (optional)
        Expression or index of axis. Default is 0.
    at : dictionary (optional)
        Dictionary of parameters in non-plotted dimension(s). If not
        provided, plots will be made at each coordinate.
    channel : string or integer (optional)
        Name or index of channel to plot. Default is 0.
    local : boolean (optional)
        Toggle plotting locally. Default is False.
    autosave : boolean (optional)
         Toggle autosave. Default is False.
    save_directory : string (optional)
         Location to save image(s). Default is None (auto-generated).
    fname : string (optional)
         File name. If None, data name is used. Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    list
        if autosave, a list of saved image files (if any).
        if not, a list of Figures
    """
    return QuicknD(
        data,
        axis,
        at=at,
        channel=channel,
        local=local,
        autosave=autosave,
        save_directory=save_directory,
        fname=fname,
    )(verbose)


def quick2D(
    data,
    xaxis=0,
    yaxis=1,
    at={},
    channel=0,
    *,
    cmap=None,
    contours=0,
    pixelated=True,
    dynamic_range=False,
    local=False,
    contours_local=True,
    autosave=False,
    save_directory=None,
    fname=None,
    verbose=True,
):
    """Quickly plot 2D slice(s) of data.

    Parameters
    ----------
    data : WrightTools.Data object.
        Data to plot.
    xaxis : string or integer (optional)
        Expression or index of horizontal axis. Default is 0.
    yaxis : string or integer (optional)
        Expression or index of vertical axis. Default is 1.
    at : dictionary (optional)
        Dictionary of parameters in non-plotted dimension(s). If not
        provided, plots will be made at each coordinate.
    cmap : Colormap
        Colormap to use.  If None, will use "default" or "signed" depending on channel values.
    channel : string or integer (optional)
        Name or index of channel to plot. Default is 0.
    contours : integer (optional)
        The number of black contour lines to add to the plot. Default is 0.
    pixelated : boolean (optional)
        Toggle between pcolor and contourf (deulaney) plotting backends.
        Default is True (pcolor).
    dynamic_range : boolean (optional)
        Force the colorbar to use all of its colors. Only changes behavior
        for signed channels. Default is False.
    local : boolean (optional)
        Toggle plotting locally. Default is False.
    contours_local : boolean (optional)
        Toggle plotting black contour lines locally. Default is True.
    autosave : boolean (optional)
        Toggle autosave. Default is False when the number of plots is 10 or less.
        When the number of plots is greater than 10, saving is forced.
    save_directory : string (optional)
         Location to save image(s). Default is None (auto-generated).
    fname : string (optional)
         File name. If None, data name is used. Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    list
        if autosave, a list of saved image files (if any).
        if not, a list of Figures
    """
    return QuicknD(
        data,
        xaxis,
        yaxis,
        at=at,
        channel=channel,
        cmap=cmap,
        contours=contours,
        pixelated=pixelated,
        dynamic_range=dynamic_range,
        local=local,
        contours_local=contours_local,
        autosave=autosave,
        save_directory=save_directory,
        fname=fname,
    )(verbose)


# TODO: norm_from_channel

def limits_from_channel(channel):
    """
    arguments
    ---------
    channel: WrightTools.Channel
        channel from which to calculate limits
    respect_null: bool (Optional)
        When true and not signed, limits will clip data that is invalid (i.e. below null)
    dynamic_range: bool (Optional)
    """
    # TODO: add dynamic range
    if channel.signed:
        limit = channel.mag()
        limits = [-limit + channel.null, limit + channel.null]
    else:
        limit = channel.max()
    return limits


def determine_contour_levels(local_channel, global_channel, contours, local):
    # force top and bottom contour to be data range then clip them out
    if local_channel.signed:
        if local:
            limit = local_channel.mag()
        else:
            limit = global_channel.mag()
        levels = np.linspace(
            -limit + local_channel.null, limit + local_channel.null, contours + 2
        )[1:-1]
    else:
        if local:
            limit = local_channel.max()
        else:
            limit = global_channel.max()
        levels = np.linspace(local_channel.null, limit, contours + 2)[1:-1]
    return levels


def _filepath_seed(save_directory, fname, nchops, artist):
    """the big ugly logic block to determine the autosave filepaths"""
    if isinstance(save_directory, str):
        save_directory = pathlib.Path(save_directory)
    elif save_directory is None:
        save_directory = pathlib.Path.cwd()
    # create a folder if multiple images
    if nchops > 1:
        save_directory = save_directory / f"{artist} {wt_kit.TimeStamp().path}"
        pathlib.Path.mkdir(save_directory)
    return save_directory, fname + " {0:0>3}.png"
