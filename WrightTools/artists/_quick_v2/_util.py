from contextlib import closing
from abc import abstractmethod
import pathlib

import matplotlib.pyplot as plt

from .._helpers import savefig
from ... import kit as wt_kit
from ...data import Data


class QuickIteratorBase:
    """base class for iterating through chopped data for figure generation"""

    def __init__(self, data: Data, *axes, **kwargs):
        """base class for looping and plotting
        v2:  generator-based, single figure, no figure limits
        subclasses need to supply artists objects that are manipulated
        and methods that define how those manipulations evolve
        and how they change with each iteration of chop

        args
        ----
        data: WrightTools.Data

        axes:  int | str | Axis
            define the x and y axes

        kwargs
        ------

        at: dict
            applies `at` before chopping, reducing dimensionality

        autosave: bool (default False)
            when true, sets up an automatic file write of each frame

        """
        self.data = data
        self.axes = [data.get_axis(a) for a in axes]
        self.at = kwargs.get("at", {})
        self.nD = len(axes)
        self.fig = None

        self.autosave = kwargs.get("autosave", False)

        self.channel_index = wt_kit.get_index(data.channel_names, kwargs.get("channel", 0))
        shape = data.channels[self.channel_index].shape
        # identify dimensions that do not involve the channel
        self.channel_slice = [0 if size == 1 else slice(None) for size in shape]
        self.sliced_constants = [
            data.axis_expressions[i] for i in range(len(shape)) if not self.channel_slice[i]
        ]
        if self.autosave:
            self.save_directory, self.filepath_seed = _filepath_seed(
                kwargs.get("save_directory", pathlib.Path.cwd()),
                kwargs.get("fname", data.natural_name),
                self.nfigs,
                f"quick{self.nD}D",
            )
        # a subclass will define this method, which initializes self.fig
        self.draw_figure()
        assert isinstance(self.fig, plt.Figure)

    def __iter__(self, verbose=False):
        if self.autosave:
            self.save_directory.mkdir(exist_ok=True)
        with closing(self.data._from_slice(self.channel_slice)) as sliced:
            for constant in self.sliced_constants:
                sliced.remove_constant(constant, verbose=False)
            for i, dati in enumerate(sliced.ichop(*[a.expression for a in self.axes], at=self.at)):
                self.update_figure(dati)
                if self.autosave:
                    filepath = self.save_directory / self.filepath_seed.format(i)
                    savefig(filepath, fig=self.fig, facecolor="white", close=True)
                    if verbose:
                        print("image saved at", str(filepath))
                self.fig.canvas.draw_idle()
                yield self.fig

    @abstractmethod
    def draw_figure(self) -> plt.Figure:
        """initialize figure"""
        ...

    @abstractmethod
    def update_figure(self, d):
        """To be defined in specific handlers.
        `d` is a WrightTools.Data object to be plotted
        This function should return a figure instance.
        If fig is provided, it should decorate the provided figure
        """
        ...

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


def _filepath_seed(save_directory, fname, nchops, artist) -> tuple[pathlib.Path, str]:
    """determine the autosave filepaths"""
    if isinstance(save_directory, str):
        save_directory = pathlib.Path(save_directory)
    elif save_directory is None:
        save_directory = pathlib.Path.cwd()
    # create a folder if multiple images
    if nchops > 1:
        save_directory = save_directory / f"{artist} {wt_kit.TimeStamp().path}"
    return save_directory, ("" if fname is None else fname + " ") + "{0:0>3}.png"
