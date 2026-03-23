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
        fname = kwargs.pop("fname", data.natural_name),

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
                self.nfigs,
                f"quick{self.nD}D",
            )
        self.kwargs = kwargs
        self.logger.info(f"{self.kwargs=}")
        # a subclass will define this method, which initializes self.fig
        self.draw_figure()
        assert isinstance(self.fig, plt.Figure)

    def __iter__(self):
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
                    self.logger.info("image {i} saved at", str(filepath))
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


    def determine_contour_levels(local_channel, global_channel, local):
        # force top and bottom contour to be data range then clip them out
        null = local_channel.null
        if local_channel.signed:
            limit = local_channel.mag() if local else global_channel.mag()
            levels = np.linspace(-limit + null, limit + null, contours + 2)[1:-1]
        else:
            limit = local_channel.max() if local else global_channel.max()
            levels = np.linspace(null, limit, contours + 2)[1:-1]
        return levels

class Quick2D(ChopIteratorBase):

    def __init__(self, *args, **kwargs):
        # set some defaults
        if "autolabel" not in kwargs:
            kwargs["autolabel"] = "both"
        if kwargs["cmap"] is None:
            kwargs.pop("cmap")
        super().__init__(*args, **kwargs)

    def draw_figure(self):
        """initialize figure, create object attrs that will be used to update"""
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
            channel if local else self.data.channels[self.channel_index],
            dynamic_range=dynamic_range,
        )
        norm_ticks = ticks_from_norm(norm)
        if self.img is None:
            if pixelated:
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
        if contours:
            contour_levels = determine_contour_levels(
                channel, self.data.channels[self.channel_index], contours_local
            )
            self.ax.contour(d, channel=self.channel_index, levels=contour_levels)
        # decoration --------------------------------------------------------------------------
        self.fig.suptitle(self.data.natural_name)
        self.subtitle.set_text(self.annotate_constants(d))
        # colorbar
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(self.img, cax=self.cax)
            self.colorbar.set_label(label=channel.natural_name)
            self.colorbar.set_ticks(norm_ticks)
        plt.sca(self.ax)
