"""Quick plotting."""

# --- import --------------------------------------------------------------------------------------


from contextlib import closing
from functools import reduce
from typing import Tuple, List, Union
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from ._helpers import (
    _title,
    create_figure,
    plot_colorbar,
    savefig,
    norm_from_channel,
    ticks_from_norm,
)
from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


__all__ = ["quick1D", "quick2D", "ChopHandler"]


# --- general purpose plotting functions ----------------------------------------------------------


class ChopHandler:
    """class for keeping track of plotting through the chopped data"""

    max_figures = 10  # value determines when interactive plotting is truncated

    def __init__(self, data, *axes, **kwargs):
        self.data = data
        self.axes = axes
        self.at = kwargs.get("at", {})
        self.nD = len(axes)

        self.autosave = kwargs.get("autosave", False)

        self.channel_index = wt_kit.get_index(data.channel_names, kwargs.get("channel", 0))
        shape = data.channels[self.channel_index].shape
        # identify dimensions that do not involve the channel
        self.channel_slice = [0 if size == 1 else slice(None) for size in shape]
        self.sliced_constants = [
            data.axis_expressions[i] for i in range(len(shape)) if not self.channel_slice[i]
        ]
        # pre-calculate the number of plots to decide whether to make a folder
        uninvolved_shape = (
            size if self.channel_slice[i] == 0 else 1 for i, size in enumerate(shape)
        )
        removed_shape = data._chop_prep(*self.axes, at=self.at)[0]
        self.nfigs = reduce(int.__mul__, removed_shape) // reduce(int.__mul__, uninvolved_shape)
        if self.nfigs > 10 and not self.autosave:
            print(
                f"number of expected figures ({self.nfigs}) is greater than the limit"
                + f"({self.max_figures}).  Only the first {self.max_figures} figures will be processed."
            )
        if self.autosave:
            self.save_directory, self.filepath_seed = _filepath_seed(
                kwargs.get("save_directory", pathlib.Path.cwd()),
                kwargs.get("fname", data.natural_name),
                self.nfigs,
                f"quick{self.nD}D",
            )

    def __call__(self, verbose=False) -> List[Union[str, plt.Figure]]:
        out = list()
        if self.autosave:
            self.save_directory.mkdir(exist_ok=True)
        with closing(self.data._from_slice(self.channel_slice)) as sliced:
            for constant in self.sliced_constants:
                sliced.remove_constant(constant, verbose=False)
            for i, fig in enumerate(map(self.plot, sliced.ichop(*self.axes, at=self.at))):
                if self.autosave:
                    filepath = self.save_directory / self.filepath_seed.format(i)
                    savefig(filepath, fig=fig, facecolor="white", close=True)
                    if verbose:
                        print("image saved at", str(filepath))
                    out.append(str(filepath))
                elif i == self.max_figures:
                    print(
                        "The maximum allowed number of figures"
                        + f"({self.max_figures}) is plotted. Stopping..."
                    )
                    break
                else:
                    out.append(fig)
        return out

    def plot(self, d):
        """To be defined in specific handlers.
        `d` is a WrightTools.Data object to be plotted
        This function should return a figure instance.
        """
        raise NotImplementedError

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

    def decorate(self, ax, *axes):
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        ax.axvline(0, lw=2, c="k")
        ax.set_xlim(axes[0].min(), axes[0].max())
        ax.grid(ls="--", color="grey", lw=0.5)
        if self.nD == 1:
            ax.axhline(self.data.channels[self.channel_index].null, lw=2, c="k")
        elif self.nD == 2:
            ax.axhline(0, lw=2, c="k")
            ax.set_ylim(axes[1].min(), axes[1].max())


def quick1D(data, *args, **kwargs):
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
        Toggle saving plots (True) as files or diplaying interactive (False).
        Default is False. When autosave is False, the number of plots is truncated by
        `ChopHandler.max_figures`.
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
    verbose = kwargs.pop("verbose", True)
    handler = _quick1D(data, *args, **kwargs)
    return handler(verbose)


def _quick1D(
    data,
    axis=0,
    at={},
    channel=0,
    *,
    local=False,
    autosave=False,
    save_directory=None,
    fname=None,
):
    """
    `quick1D` worker; factored out for testing purposes
    returns Quick1D handler object
    """

    class Quick1D(ChopHandler):
        def __init__(self, *args, **kwargs):
            self._global_limits = None
            super().__init__(*args, **kwargs)

        def plot(self, d):
            # unpack data -------------------------------------------------------------------------
            axis = d.axes[0]
            channel = d.channels[self.channel_index]
            # create figure ------------------------------------------------------------------------
            aspects = [[[0, 0], 0.5]]
            fig, gs = create_figure(width="single", nrows=1, cols=[1], aspects=aspects)
            ax = plt.subplot(gs[0, 0])
            # plot --------------------------------------------------------------------------------
            ax.plot(d, channel=self.channel_index, lw=2, autolabel=True)
            ax.scatter(axis.full, channel[:], color="grey", alpha=0.5, edgecolor="none")
            # decoration --------------------------------------------------------------------------
            if not local:
                ax.set_ylim(*self.global_limits)
            self.decorate(ax, *d.axes)
            # constants: variable marker lines, title
            _title(fig, self.data.natural_name, subtitle=self.annotate_constants(d))
            return fig

        @property
        def global_limits(self):
            if self._global_limits is None:
                data_channel = self.data.channels[self.channel_index]
                cmin, cmax = data_channel.min(), data_channel.max()
                buffer = (cmax - cmin) * 0.05
                limits = [cmin - buffer, cmax + buffer]
                if np.sign(limits[0]) != np.sign(cmin):
                    limits[0] = 0
                if np.sign(limits[1]) != np.sign(cmax):
                    limits[1] = 0
                self._global_limits = limits
            return self._global_limits

    return Quick1D(
        data,
        axis,
        at=at,
        channel=channel,
        autosave=autosave,
        save_directory=save_directory,
        fname=fname,
    )


def quick2D(data, *args, **kwargs):
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
        Toggle saving plots (True) as files or diplaying interactive (False).
        Default is False. When autosave is False, the number of plots is truncated by
        `ChopHandler.max_figures`.
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
    verbose = kwargs.pop("verbose", True)
    handler = _quick2D(data, *args, **kwargs)
    return handler(verbose)


def _quick2D(
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
):

    def determine_contour_levels(local_channel, global_channel, contours, local):
        # force top and bottom contour to be data range then clip them out
        null = local_channel.null
        if local_channel.signed:
            limit = local_channel.mag() if local else global_channel.mag()
            levels = np.linspace(-limit + null, limit + null, contours + 2)[1:-1]
        else:
            limit = local_channel.max() if local else global_channel.max()
            levels = np.linspace(null, limit, contours + 2)[1:-1]
        return levels

    class Quick2D(ChopHandler):
        kwargs = {"autolabel": "both"}
        if cmap is not None:
            kwargs["cmap"] = cmap

        def plot(self, d):
            # unpack data -------------------------------------------------------------------------
            xaxis = d.axes[0]
            yaxis = d.axes[1]
            channel = d.channels[self.channel_index]
            # create figure -----------------------------------------------------------------------
            if xaxis.units == yaxis.units:
                xr = xaxis.max() - xaxis.min()
                yr = yaxis.max() - yaxis.min()
                aspect = np.abs(yr / xr)
                aspect = np.clip(aspect, 1 / 3.0, 3.0)
            else:
                aspect = 1
            fig, gs = create_figure(
                width="single", nrows=1, cols=[1, "cbar"], aspects=[[[0, 0], aspect]]
            )
            ax = plt.subplot(gs[0])
            ax.patch.set_facecolor("w")
            # colors ------------------------------------------------------------------------------
            norm = norm_from_channel(
                channel if local else self.data.channels[self.channel_index],
                dynamic_range=dynamic_range,
            )
            norm_ticks = ticks_from_norm(norm)
            if pixelated:
                img = ax.pcolormesh(d, channel=self.channel_index, norm=norm, **self.kwargs)
            else:
                img = ax.contourf(d, channel=self.channel_index, norm=norm, **self.kwargs)
            # contour lines -----------------------------------------------------------------------
            if contours:
                contour_levels = determine_contour_levels(
                    channel, self.data.channels[self.channel_index], contours_local
                )
                ax.contour(d, channel=self.channel_index, levels=contour_levels)
            # decoration --------------------------------------------------------------------------
            self.decorate(ax, *d.axes)
            _title(fig, self.data.natural_name, subtitle=self.annotate_constants(d))
            # colorbar
            cax = plt.subplot(gs[1])
            plot_colorbar(
                cax=cax, cmap=img.get_cmap(), ticks=norm_ticks, label=channel.natural_name
            )
            plt.sca(ax)
            return fig

    return Quick2D(
        data,
        xaxis,
        yaxis,
        at=at,
        channel=channel,
        autosave=autosave,
        save_directory=save_directory,
        fname=fname,
    )


def _filepath_seed(save_directory, fname, nchops, artist) -> Tuple[pathlib.Path, str]:
    """determine the autosave filepaths"""
    if isinstance(save_directory, str):
        save_directory = pathlib.Path(save_directory)
    elif save_directory is None:
        save_directory = pathlib.Path.cwd()
    # create a folder if multiple images
    if nchops > 1:
        save_directory = save_directory / f"{artist} {wt_kit.TimeStamp().path}"
    return save_directory, ("" if fname is None else fname + " ") + "{0:0>3}.png"
