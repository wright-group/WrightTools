"""quick1D"""

import numpy as np
import matplotlib.pyplot as plt

from .._helpers import (
    _title,
    create_figure,
)
from ._util import ChopIteratorBase, legacy_quick_class, annotate_constants

__all__ = ["Quick1Ds", "Quick1D", "Quick1DIterator", "Quick1DLegacy"]


class Quick1DIterator(ChopIteratorBase):
    """1D plot iterator that creates a single figure, refreshing the content on each iteration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "autolabel" not in self.kwargs:
            self.kwargs["autolabel"] = "both"
        self.local = self.kwargs.pop("local")
        self.draw_figure()

    def draw_figure(self):
        """initialize figure and create object attrs that will be used to update"""
        xaxis = self.data.get_axis(self.axes[0])
        aspect = 1
        self.fig, gs = create_figure(width="single", nrows=1, cols=[1], aspects=[[[0, 0], aspect]])
        self.ax = plt.subplot(gs[0])
        self.ax.patch.set_facecolor("w")
        self.cax = plt.subplot(gs[1])
        self.subtitle = _title(self.fig, "", subtitle="")
        self.decorate(self.ax, xaxis)
        self.colorbar = None
        self.img = None

    def update_figure(self, d) -> plt.Figure:
        # unpack data -------------------------------------------------------------------------
        channel = d.channels[self.channel_index]
        # decoration --------------------------------------------------------------------------
        self.fig.suptitle(self.data.natural_name)
        self.subtitle.set_text(annotate_constants(d, self.ax))
        self.fig.canvas.draw_idle()
        plt.sca(self.ax)
        return self.fig


Quick1DLegacy = legacy_quick_class(Quick1DIterator)


def quick1Ds(
    data,
    xaxis: int | str = 0,
    at: dict = {},
    channel: int | str = 0,
    local: bool = False,
    autosave: bool = False,
    save_directory=None,
    fname=None,
):
    """
    Quick generator of 1D image frames. Wraps `Quick1D` class with explicit kwargs

    Parameters
    ----------
    data : WrightTools.Data object.
        Data to plot.
    xaxis : string or integer (optional)
        Expression or index of horizontal axis. Default is 0.
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

    Returns
    -------
    Iterable
        an iterator that will update the plot with the next data instance with each iteration

    Usage
    -----
    ```
    for frame in quick2D(data, autosave=True):
        plt.show()  # save and show member interactively

    """
    ...


def _quick1D():
    """wrapper of Quick1DLegacy to supply kwarg arguments"""
    ...


def quick1D(): ...
