"""quick1D"""

import matplotlib.pyplot as plt

from .._helpers import (
    _title,
    create_figure,
)
from ._util import ChopIteratorBase, legacy_quick_class, annotate_constants

__all__ = ["quick1Ds", "quick1D", "Quick1DIterator", "Quick1DLegacy"]


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
        if self.img is None:
            self.img = self.ax.plot(d, channel=self.channel_index, **self.kwargs)
        else:
            self.img.set_data(d.axis[:].squeeze(), d.channels[self.channel_index].squeeze())
        self.fig.canvas.draw_idle()
        plt.sca(self.ax)
        return self.fig


Quick1DLegacy = legacy_quick_class(Quick1DIterator)


def quick1Ds(
    data,
    axis: int | str = 0,
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
    axis : string or integer (optional)
        Expression or index of horizontal axis. Default is 0.
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
    return Quick1DIterator(
        data,
        axis,
        at=at,
        channel=channel,
        local=local,
        autosave=autosave,
        save_directory=save_directory,
        fname=fname,
    )


def _quick1D(
    data,
    axis: int | str = 0,
    at: dict = {},
    channel: int | str = 0,
    local: bool = False,
    autosave: bool = False,
    save_directory=None,
    fname=None,
):
    """wrapper of Quick1DLegacy to supply kwarg arguments"""
    return Quick1DLegacy(
        data,
        axis,
        at=at,
        channel=channel,
        local=local,
        autosave=autosave,
        save_directory=save_directory,
        fname=fname,
    )


def quick1D(
    data,
    axis: int | str = 0,
    at: dict = {},
    channel: int | str = 0,
    local: bool = False,
    autosave: bool = False,
    save_directory=None,
    fname=None,
):
    return _quick1D(
        data,
        axis,
        at=at,
        channel=channel,
        local=local,
        autosave=autosave,
        save_directory=save_directory,
        fname=fname,
    )()
