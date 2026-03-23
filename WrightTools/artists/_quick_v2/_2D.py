"""quick2D"""

from typing import Iterator

import numpy as np
import matplotlib.pyplot as plt

from .._helpers import (
    _title,
    create_figure,
    norm_from_channel,
    ticks_from_norm,
)
from ._util import ChopIteratorBase, annotate_constants

# --- define --------------------------------------------------------------------------------------


__all__ = ["quick2Ds"]


# --- general purpose plotting functions ----------------------------------------------------------


def determine_contour_levels(local_channel, global_channel, local: bool, contours: int):
    # force top and bottom contour to be data lims then clip them out
    null = local_channel.null
    if local_channel.signed:
        limit = local_channel.mag() if local else global_channel.mag()
        levels = np.linspace(-limit + null, limit + null, contours + 2)[1:-1]
    else:
        limit = local_channel.max() if local else global_channel.max()
        levels = np.linspace(null, limit, contours + 2)[1:-1]
    return levels


class Quick2DIterator(ChopIteratorBase):
    """Quick2D that creates a single figure, refreshing the content on each iteration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "autolabel" not in self.kwargs:
            self.kwargs["autolabel"] = "both"
        if self.kwargs["cmap"] is None:
            self.kwargs.pop("cmap")
        self.local = self.kwargs.pop("local")
        self.dynamic_range = self.kwargs.pop("dynamic_range")
        self.pixelated = self.kwargs.pop("pixelated")
        self.contours = self.kwargs.pop("contours")
        self.contours_local = self.kwargs.pop("contours_local")
        self.draw_figure()
        assert isinstance(self.fig, plt.Figure)

    def draw_figure(self):
        """initialize figure and create object attrs that will be used to update"""
        xaxis = self.data.get_axis(self.axes[0])
        yaxis = self.data.get_axis(self.axes[1])
        if xaxis.units == yaxis.units:
            xr = xaxis.max() - xaxis.min()
            yr = yaxis.max() - yaxis.min()
            aspect = np.abs(yr / xr)
            aspect = np.clip(aspect, 1 / 3.0, 3.0)
        else:
            aspect = 1
        self.fig, gs = create_figure(
            width="single", nrows=1, cols=[1, "cbar"], aspects=[[[0, 0], aspect]]
        )
        self.ax = plt.subplot(gs[0])
        self.ax.patch.set_facecolor("w")
        self.cax = plt.subplot(gs[1])
        self.subtitle = _title(self.fig, "", subtitle="")
        self.decorate(self.ax, xaxis, yaxis)
        self.colorbar = None
        self.img = None

    def update_figure(self, d):
        # unpack data -------------------------------------------------------------------------
        channel = d.channels[self.channel_index]
        # colors ------------------------------------------------------------------------------
        norm = norm_from_channel(
            channel if self.local else self.data.channels[self.channel_index],
            dynamic_range=self.dynamic_range,
        )
        norm_ticks = ticks_from_norm(norm)
        if self.img is None:
            if self.pixelated:
                self.img = self.ax.pcolormesh(
                    d, channel=self.channel_index, norm=norm, **self.kwargs
                )
            else:
                self.img = self.ax.contourf(
                    d, channel=self.channel_index, norm=norm, **self.kwargs
                )
        else:
            self.img.set_array(d.channels[self.channel_index])
        # contour lines -----------------------------------------------------------------------
        if self.contours:
            contour_levels = determine_contour_levels(
                channel, self.data.channels[self.channel_index], self.contours_local, self.contours
            )
            self.ax.contour(d, channel=self.channel_index, levels=contour_levels)
        # decoration --------------------------------------------------------------------------
        self.fig.suptitle(self.data.natural_name)
        self.subtitle.set_text(annotate_constants(d, self.ax))
        # colorbar
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(self.img, cax=self.cax)
            self.colorbar.set_label(label=channel.natural_name)
            self.colorbar.set_ticks(norm_ticks)
        self.fig.canvas.draw_idle()
        plt.sca(self.ax)
        return self.fig


# a wrapper for instance generation with explicit kwargs, docstring, typing
def quick2Ds(
    data,
    xaxis: int | str = 0,
    yaxis: int | str = 1,
    at: dict = {},
    channel: int | str = 0,
    cmap=None,
    contours: int = 0,
    pixelated: bool = True,
    dynamic_range: bool = False,
    local: bool = False,
    contours_local: bool = True,
    autosave: bool = False,
    save_directory=None,
    fname=None,
) -> Iterator[plt.Figure]:
    """Quickly generator of 2D image frames. Wraps class `Quick2D` class with explicit kwargs

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
    Iterable
        an iterator that will update the plot with the next data instance with each iteration

    Usage
    -----
    ```
    for frame in quick2D(data, autosave=True):
        plt.show()  # save and show member interactively
    ```

    """
    return Quick2DIterator(
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
    )

