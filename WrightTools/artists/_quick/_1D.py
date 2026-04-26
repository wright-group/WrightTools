"""quick1D"""

import matplotlib.pyplot as plt
import numpy as np

from .._helpers import (
    _title,
    create_figure,
)
from ._util import ChopIteratorBase, legacy_quick_class, annotate_constants

__all__ = ["quick1Ds", "quick1D", "Quick1DIterator", "Quick1DLegacy"]


class Quick1DIterator(ChopIteratorBase):
    """
    Iterator of 1D plots for 1D chops of data.
    Creates a single figure, refreshing the content on each iteration.
    """

    defaults = dict(lw=2, marker="o", markeredgewidth=0, autolabel="both")

    def __init__(self, *args, **kwargs):
        self._global_limits = None
        super().__init__(*args, **kwargs)
        self.local = self.kwargs.pop("local")
        self.draw_figure()
        for k, v in self.defaults.items():
            if k not in self.kwargs:
                self.kwargs[k] = v

    def draw_figure(self):
        """initialize figure and create object attrs that will be used to update"""
        xaxis = self.data.get_axis(self.axes[0])
        aspect = self.kwargs.get("aspect", 0.5)
        self.fig, gs = create_figure(width="single", nrows=1, cols=[1], aspects=[[[0, 0], aspect]])
        self.ax = plt.subplot(gs[0])
        self.ax.patch.set_facecolor("w")
        self.subtitle = _title(self.fig, "", subtitle="")
        self.decorate(self.ax, xaxis)
        self.colorbar = None
        self.line = None

    def update_figure(self, d) -> plt.Figure:
        self.fig.suptitle(self.data.natural_name)
        self.subtitle.set_text(annotate_constants(d, self.ax))
        if self.line is None:
            (self.line,) = self.ax.plot(d, channel=self.channel_index, **self.kwargs)
        else:
            self.line.set_data(d.axes[0][:].squeeze(), d.channels[self.channel_index][:].squeeze())
        if self.local:
            self.ax.autoscale(axis="y")
        else:
            self.ax.set_ylim(*self.global_limits)
        self.fig.canvas.draw_idle()
        plt.sca(self.ax)
        return self.fig

    @property
    def global_limits(self):
        """yaxis limits logic"""
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
) -> Quick1DIterator:
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
    for frame in quick1D(data, autosave=True):
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


def quick1D(
    data,
    axis: int | str = 0,
    at: dict = {},
    channel: int | str = 0,
    local: bool = False,
    autosave: bool = False,
    save_directory=None,
    fname=None,
    verbose=False,
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
        Toggle saving plots (True) as files or diplaying interactive (False).
        Default is False. When autosave is False, the number of plots is truncated by
        `ChopHandler.max_figures`.
    save_directory : string (optional)
         Location to save image(s). Default is None (auto-generated).
    fname : string (optional)
         File name. If None, data name is used. Default is None.
    verbose : boolean (optional)
        Deprecated option.  Use logging config to customize code feedback.

    Returns
    -------
    list
        if autosave, a list of saved image files (if any).
        if not, a list of Figures

    See Also
    --------
    ``artists.quick1Ds`` : Iterator implementation of quick1D
    """
    return Quick1DLegacy(
        data,
        axis,
        at=at,
        channel=channel,
        local=local,
        autosave=autosave,
        save_directory=save_directory,
        fname=fname,
    )()
