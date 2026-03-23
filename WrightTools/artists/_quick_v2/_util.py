from contextlib import closing
from abc import abstractmethod
import pathlib
import logging

import matplotlib.pyplot as plt

from .._helpers import savefig
from ... import kit as wt_kit
from ...data import Data


__all__ = ["ChopIteratorBase"]


class ChopIteratorBase:
    """base class for iterating through data for figure generation"""

    logger = logging.getLogger("ChopIteratorBase")

    def __init__(self, data: Data, *axes, **kwargs):
        """base class for looping and plotting
        v2:  iterator-based, no figure limit checks
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

        channel: str or int
            channel index or name

        autosave: bool (default False)
            when true, sets up an automatic file write of each frame

        save_directory: path-like
            directory to save the files.  only applies for autosave=True

        fname: str
            stub to append to for saved frame filenames

        **kwargs
            additional kwargs are stored as a kwargs attr, which can be accessed by subclasses

        """
        self.data = data
        self.axes = [data.get_axis(a) for a in axes]
        self.at = kwargs.pop("at", {})
        save_directory = kwargs.pop("save_directory", pathlib.Path.cwd())
        fname = kwargs.pop("fname", data.natural_name)

        self.nD = len(axes)
        self.fig = None

        self.autosave = kwargs.pop("autosave", False)

        self.channel_index = wt_kit.get_index(data.channel_names, kwargs.pop("channel", 0))
        shape = data.channels[self.channel_index].shape
        # identify dimensions that do not involve the channel
        self.channel_slice = [0 if size == 1 else slice(None) for size in shape]
        self.sliced_constants = [
            data.axis_expressions[i] for i in range(len(shape)) if not self.channel_slice[i]
        ]
        if self.autosave:
            self.save_directory, self.filepath_seed = _filepath_seed(
                save_directory,
                fname,
                f"quick{self.nD}D",
            )
        self.kwargs = kwargs
        self.logger.info(f"{self.kwargs=}")

    def __iter__(self):
        if self.autosave:
            self.save_directory.mkdir(exist_ok=True)
        with closing(self.data._from_slice(self.channel_slice)) as sliced:
            for constant in self.sliced_constants:
                sliced.remove_constant(constant, verbose=False)
            for i, figi in enumerate(
                map(
                    self.update_figure,
                    sliced.ichop(*[a.expression for a in self.axes], at=self.at),
                )
            ):
                if self.autosave:
                    filepath = self.save_directory / self.filepath_seed.format(i)
                    self.savefig(figi, filepath)
                    self.logger.info(f"image {i} saved at {filepath}")
                    self.filepath = filepath
                yield figi

    def savefig(self, fig, filepath):
        """factored here so that it can be overloaded if needed"""
        savefig(filepath, fig=fig, facecolor="white", close=False)

    @abstractmethod
    def update_figure(self, d) -> plt.Figure:
        """To be defined in specific handlers.
        `d` is a WrightTools.Data object to be plotted
        This function should return a figure instance.
        If fig is provided, it should decorate the provided figure
        """
        ...

    def decorate(self, ax, *axes):
        ax.tick_params(axis="x", labelrotation=45, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.axvline(0, lw=2, c="k")
        ax.set_xlim(axes[0].min(), axes[0].max())
        ax.grid(ls="--", color="grey", lw=0.5)
        if self.nD == 1:
            ax.axhline(self.data.channels[self.channel_index].null, lw=2, c="k")
        elif self.nD == 2:
            ax.axhline(0, lw=2, c="k")
            ax.set_ylim(axes[1].min(), axes[1].max())


def _filepath_seed(save_directory, fname, artist) -> tuple[pathlib.Path, str]:
    """determine the autosave filepaths"""
    if isinstance(save_directory, str):
        save_directory = pathlib.Path(save_directory)
    elif save_directory is None:
        save_directory = pathlib.Path.cwd()
    # create a folder
    save_directory = save_directory / f"{artist} {wt_kit.TimeStamp().path}"
    return save_directory, ("" if fname is None else fname + " ") + "{0:0>3}.png"


def annotate_constants(d, ax):
    """generate constant string and mark constants on axes"""
    ls = []
    for c in d.constants:
        if c.units:
            ls.append(c.label)
            # x axis
            if d.axes[0].units_kind == c.units_kind:
                c.convert(d.axes[0].units)
                ax.axvline(c.value, color="k", linewidth=4, alpha=0.25)
            # y axis
            if (len(d.axes) == 2) and (d.axes[1].units_kind == c.units_kind):
                c.convert(d.axes[1].units)
                ax.axhline(c.value, color="k", linewidth=4, alpha=0.25)
    return ", ".join(ls)
