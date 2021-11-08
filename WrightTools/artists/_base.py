"""Base tools for visualizing data."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import matplotlib
from matplotlib.projections import register_projection
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .. import exceptions as wt_exceptions
from .. import kit as wt_kit
from ..data import Data
from ._colors import colormaps


# --- define -------------------------------------------------------------------------------------


__all__ = ["Axes", "Figure", "GridSpec", "apply_rcparams"]


# --- classes -------------------------------------------------------------------------------------


class Axes(matplotlib.axes.Axes):
    """Axes."""

    name = "wright"

    transposed = False
    is_sideplot = False

    def _parse_cmap(self, data=None, channel_index=None, **kwargs):
        if "cmap" in kwargs.keys():
            if isinstance(kwargs["cmap"], str):
                kwargs["cmap"] = colormaps[kwargs["cmap"]]
        elif data:
            if data.channels[channel_index].signed:
                kwargs["cmap"] = colormaps["signed"]
                return kwargs
            kwargs["cmap"] = colormaps["default"]
        return kwargs

    def _apply_labels(
        self, autolabel="none", xlabel=None, ylabel=None, data=None, channel_index=0
    ):
        """Apply x and y labels to axes.

        Parameters
        ----------
        autolabel : {'none', 'both', 'x', 'y'} (optional)
            Label(s) to apply from data. Default is none.
        xlabel : string (optional)
            x label. Default is None.
        ylabel : string (optional)
            y label. Default is None.
        data : WrightTools.data.Data object (optional)
            data to read labels from. Default is None.
        channel_index : integer (optional)
            Channel index. Default is 0.
        """
        # read from data
        if autolabel in ["xy", "both", "x"] and not xlabel:
            xlabel = data.axes[0].label
        if autolabel in ["xy", "both", "y"] and not ylabel:
            if data.ndim == 1:
                ylabel = data.channels[channel_index].label
            elif data.ndim == 2:
                ylabel = data.axes[1].label
        # apply
        if xlabel:
            if isinstance(xlabel, bool):
                xlabel = data.axes[0].label
            self.set_xlabel(xlabel, fontsize=18)
        if ylabel:
            if isinstance(ylabel, bool):
                ylabel = data.axes[1].label
            self.set_ylabel(ylabel, fontsize=18)

    def _parse_limits(self, zi=None, data=None, channel_index=None, dynamic_range=False, **kwargs):
        if "norm" in kwargs:
            return kwargs
        if zi is not None:
            if "levels" in kwargs.keys():
                levels = kwargs["levels"]
                vmin = levels.min()
                vmax = levels.max()
            else:
                vmin = np.nanmin(zi)
                vmax = np.nanmax(zi)
        elif data is not None:
            signed = data.channels[channel_index].signed
            null = data.channels[channel_index].null
            if signed and dynamic_range:
                vmin = -data.channels[channel_index].minor_extent + null
                vmax = +data.channels[channel_index].minor_extent + null
            elif signed and not dynamic_range:
                vmin = -data.channels[channel_index].major_extent + null
                vmax = +data.channels[channel_index].major_extent + null
            else:
                vmin = null
                vmax = data.channels[channel_index].max()
        # don't overwrite
        if "vmin" not in kwargs.keys():
            kwargs["vmin"] = vmin
        if "vmax" not in kwargs.keys():
            kwargs["vmax"] = vmax
        return kwargs

    def _parse_plot_args(self, *args, **kwargs):
        plot_type = kwargs.pop("plot_type")
        if plot_type not in ["pcolor", "pcolormesh", "contourf", "contour", "imshow"]:
            raise NotImplementedError
        args = list(args)  # offer pop, append etc
        dynamic_range = kwargs.pop("dynamic_range", False)
        if isinstance(args[0], Data):
            data = args.pop(0)
            channel = kwargs.pop("channel", 0)
            channel_index = wt_kit.get_index(data.channel_names, channel)
            squeeze = np.array(data.channels[channel_index].shape) == 1
            xa = data.axes[0]
            ya = data.axes[1]
            for sq, xs, ys in zip(squeeze, xa.shape, ya.shape):
                if sq and (xs != 1 or ys != 1):
                    raise wt_exceptions.ValueError("Cannot squeeze axis to fit channel")
            zi = data.channels[channel_index].points
            if not zi.ndim == 2:
                raise wt_exceptions.DimensionalityError(ndim, data.ndim)
            squeeze = tuple([0 if i else slice(None) for i in squeeze])
            if plot_type == "imshow":
                if "aspect" not in kwargs.keys():
                    kwargs["aspect"] = "auto"
                if "origin" not in kwargs.keys():
                    kwargs["origin"] = "lower"
                if "interpolation" not in kwargs.keys():
                    if max(zi.shape) < 10 ** 3:  # TODO: better decision logic
                        kwargs["interpolation"] = "nearest"
                    else:
                        kwargs["interpolation"] = "antialiased"
                xi = xa[:][squeeze]
                yi = ya[:][squeeze]

                zi = zi.transpose(_order_for_imshow(xi, yi))
                # extract extent
                if "extent" not in kwargs.keys():
                    xlim = [xi[0, 0], xi[-1, -1]]
                    ylim = [yi[0, 0], yi[-1, -1]]
                    xstep = (xlim[1] - xlim[0]) / (2 * xi.size)
                    ystep = (ylim[1] - ylim[0]) / (2 * yi.size)
                    x_extent = [xlim[0] - xstep, xlim[1] + xstep]
                    y_extent = [ylim[0] - ystep, ylim[1] + ystep]
                    extent = [*x_extent, *y_extent]
                    kwargs["extent"] = extent
                args = [zi] + args
            else:
                xi = xa.full[squeeze]
                yi = ya.full[squeeze]
                args = [xi, yi, zi] + args
            # limits
            kwargs = self._parse_limits(
                data=data, channel_index=channel_index, dynamic_range=dynamic_range, **kwargs
            )
            if plot_type == "contourf":
                if "levels" not in kwargs.keys():
                    kwargs["levels"] = np.linspace(kwargs["vmin"], kwargs["vmax"], 256)
            elif plot_type == "contour":
                if "levels" not in kwargs.keys():
                    if data.channels[channel_index].signed:
                        n = 11
                    else:
                        n = 6
                    kwargs["levels"] = np.linspace(kwargs.pop("vmin"), kwargs.pop("vmax"), n)[1:-1]
                # colors
                if "colors" not in kwargs.keys() and "cmap" not in kwargs.keys():
                    kwargs["colors"] = "k"
                if "alpha" not in kwargs.keys():
                    kwargs["alpha"] = 0.5
            if plot_type in ["pcolor", "pcolormesh", "contourf", "imshow"]:
                kwargs = self._parse_cmap(data=data, channel_index=channel_index, **kwargs)
        else:
            if plot_type == "imshow":
                kwargs = self._parse_limits(zi=args[0], **kwargs)
            else:
                kwargs = self._parse_limits(zi=args[2], **kwargs)
            data = None
            channel_index = 0
            if plot_type == "contourf":
                if "levels" not in kwargs.keys():
                    kwargs["levels"] = np.linspace(kwargs["vmin"], kwargs["vmax"], 256)
            if plot_type in ["pcolor", "pcolormesh", "contourf", "imshow"]:
                kwargs = self._parse_cmap(**kwargs)
        # labels
        self._apply_labels(
            autolabel=kwargs.pop("autolabel", False),
            xlabel=kwargs.pop("xlabel", None),
            ylabel=kwargs.pop("ylabel", None),
            data=data,
            channel_index=channel_index,
        )

        if plot_type != "contour":
            self.set_facecolor([0.75] * 3)
        if plot_type.startswith("pcolor"):
            kwargs["shading"] = kwargs.get("shading", "auto")

        return args, kwargs

    def add_sideplot(self, along, pad=0, height=0.75, ymin=0, ymax=1.1):
        """Add a side axis.

        Parameters
        ----------
        along : {'x', 'y'}
            Axis to add along.
        pad : float (optional)
            Side axis pad. Default is 0.
        height : float (optional)
            Side axis height. Default is 0.
        """
        # divider should only be created once
        if hasattr(self, "divider"):
            divider = self.divider
        else:
            divider = make_axes_locatable(self)
            setattr(self, "divider", divider)
        # create
        if along == "x":
            ax = self.sidex = divider.append_axes("top", height, pad=pad, sharex=self)
        elif along == "y":
            ax = self.sidey = divider.append_axes("right", height, pad=pad, sharey=self)
            ax.transposed = True
        # beautify
        if along == "x":
            ax.set_ylim(ymin, ymax)
        elif along == "y":
            ax.set_xlim(ymin, ymax)
        ax.autoscale(enable=False)
        ax.set_adjustable("box")
        ax.is_sideplot = True
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="both", which="both", length=0)
        return ax

    def contour(self, *args, **kwargs):
        """Plot contours.

        If a 3D or higher Data object is passed, a lower dimensional
        channel can be plotted, provided the ``squeeze`` of the channel
        has ``ndim==2`` and the first two axes do not span dimensions
        other than those spanned by that channel.

        Parameters
        ----------
        data : 2D WrightTools.data.Data object
            Data to plot.
        channel : int or string (optional)
            Channel index or name. Default is 0.
        dynamic_range : boolean (optional)
            Force plotting of all contours, overloading for major extent. Only applies to signed
            data. Default is False.
        autolabel : {'none', 'both', 'x', 'y'}  (optional)
            Parameterize application of labels directly from data object. Default is none.
        xlabel : string (optional)
            xlabel. Default is None.
        ylabel : string (optional)
            ylabel. Default is None.
        **kwargs
            matplotlib.axes.Axes.contour__ optional keyword arguments.

            __ https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.contour.html

        Returns
        -------
        matplotlib.contour.QuadContourSet
        """
        args, kwargs = self._parse_plot_args(*args, **kwargs, plot_type="contour")
        return super().contour(*args, **kwargs)

    def contourf(self, *args, **kwargs):
        """Plot contours.

        If a 3D or higher Data object is passed, a lower dimensional
        channel can be plotted, provided the ``squeeze`` of the channel
        has ``ndim==2`` and the first two axes do not span dimensions
        other than those spanned by that channel.

        Parameters
        ----------
        data : 2D WrightTools.data.Data object
            Data to plot.
        channel : int or string (optional)
            Channel index or name. Default is 0.
        dynamic_range : boolean (optional)
            Force plotting of all contours, overloading for major extent. Only applies to signed
            data. Default is False.
        autolabel : {'none', 'both', 'x', 'y'}  (optional)
            Parameterize application of labels directly from data object. Default is none.
        xlabel : string (optional)
            xlabel. Default is None.
        ylabel : string (optional)
            ylabel. Default is None.
        **kwargs
            matplotlib.axes.Axes.contourf__ optional keyword arguments.

            __ https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.contourf.html

        Returns
        -------
        matplotlib.contour.QuadContourSet
        """
        args, kwargs = self._parse_plot_args(*args, **kwargs, plot_type="contourf")
        # Overloading contourf in an attempt to fix aliasing problems when saving vector graphics
        # see https://stackoverflow.com/questions/15822159
        # also see https://stackoverflow.com/a/32911283
        # set_edgecolor('face') does indeed remove all of the aliasing problems
        # unfortunately, it also seems to distort the plot in a subtle but important way
        # it shifts the entire colorbar down w.r.t. the data (by one contour? not clear)
        # so for now, I am trying to fix the problem by adding contour just below contourf
        # this does not perfectly get rid of the aliasing, but it doesn't distort the data
        # which is more important
        # I anticipate that this method will be tinkered with in the future
        # so I've left the things I have tried and abandoned as comments---good luck!
        # ---Blaise 2017-07-30
        kwargs["antialiased"] = False
        kwargs["extend"] = "both"
        contours = super().contourf(*args, **kwargs)
        # fill lines
        zorder = contours.collections[0].zorder - 0.1
        levels = (contours.levels[1:] + contours.levels[:-1]) / 2
        matplotlib.axes.Axes.contour(
            self, *args[:3], levels=levels, cmap=contours.cmap, zorder=zorder
        )
        # decoration
        self.set_facecolor([0.75] * 3)
        # PathCollection modifications
        for c in contours.collections:
            pass
            # c.set_rasterized(True)
            # c.set_edgecolor('face')
        return contours

    def legend(self, *args, **kwargs):
        """Add a legend.

        Parameters
        ----------
        *args
            matplotlib legend args.
        *kwargs
            matplotlib legend kwargs.

        Returns
        -------
        legend
        """
        if "fancybox" not in kwargs.keys():
            kwargs["fancybox"] = False
        if "framealpha" not in kwargs.keys():
            kwargs["framealpha"] = 1.0
        return super().legend(*args, **kwargs)

    def pcolor(self, *args, **kwargs):
        """Create a pseudocolor plot of a 2-D array.

        If a 3D or higher Data object is passed, a lower dimensional
        channel can be plotted, provided the ``squeeze`` of the channel
        has ``ndim==2`` and the first two axes do not span dimensions
        other than those spanned by that channel.

        Defaults to ``shading="auto"`` to ensure that color boundaries
        are drawn bisecting point positions, when possible.

        Parameters
        ----------
        data : 2D WrightTools.data.Data object
            Data to plot.
        channel : int or string (optional)
            Channel index or name. Default is 0.
        dynamic_range : boolean (optional)
            Force plotting of all contours, overloading for major extent. Only applies to signed
            data. Default is False.
        autolabel : {'none', 'both', 'x', 'y'}  (optional)
            Parameterize application of labels directly from data object. Default is none.
        xlabel : string (optional)
            xlabel. Default is None.
        ylabel : string (optional)
            ylabel. Default is None.
        **kwargs
            matplotlib.axes.Axes.pcolor__ optional keyword arguments.

            __ https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.pcolor.html

        Returns
        -------
        matplotlib.collections.PolyCollection
        """
        args, kwargs = self._parse_plot_args(*args, **kwargs, plot_type="pcolor")
        return super().pcolor(*args, **kwargs)

    def imshow(self, *args, **kwargs):
        """Create a pseudocolor plot of a 2-D array.  The array is plotted
        with uniform spacing.  Quicker than pcolor, pcolormesh.

        **Requires that the plotted axes are grid aligned (i.e. the `squeeze`
         of each axis has ``ndim==1``).**

        If a 3D or higher Data object is passed, a lower dimensional
        channel can be plotted, provided the ``squeeze`` of the channel
        has ``ndim==2``.

        Defaults to ``aspect="auto"`` (pixels are stretched to fit the
        subplot axes)

        If `interpolation` method is not specified, defaults to either
        "antialiased" (for large images) or "nearest" (for small arrays).

        `extent` defaults to ensure that pixels are drawn bisecting point
        positions.

        Parameters
        ----------
        data : 2D WrightTools.data.Data object
            Data to plot.
        channel : int or string (optional)
            Channel index or name. Default is 0.
        dynamic_range : boolean (optional)
            Force plotting of all contours, overloading for major extent. Only applies to signed
            data. Default is False.
        autolabel : {'none', 'both', 'x', 'y'}  (optional)
            Parameterize application of labels directly from data object. Default is none.
        xlabel : string (optional)
            xlabel. Default is None.
        ylabel : string (optional)
            ylabel. Default is None.
        **kwargs
            matplotlib.axes.Axes.imshow__ optional keyword arguments.

            __ https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html

        Returns
        -------
        matplotlib.image.AxesImage
        """
        xlim, ylim = super().get_xlim(), super().get_ylim()
        old_signs = list(map(lambda x: (x[1] - x[0]) > 0, [xlim, ylim]))

        args, kwargs = self._parse_plot_args(*args, **kwargs, plot_type="imshow")
        out = super().imshow(*args, **kwargs)

        # undo axis order if it was flipped
        xlim, ylim = super().get_xlim(), super().get_ylim()
        new_signs = list(map(lambda x: (x[1] - x[0]) > 0, [xlim, ylim]))
        if old_signs[0] != new_signs[0]:
            super().invert_xaxis()
        if old_signs[1] != new_signs[1]:
            super().invert_yaxis()
        return out

    def pcolormesh(self, *args, **kwargs):
        """Create a pseudocolor plot of a 2-D array.

        If a 3D or higher Data object is passed, a lower dimensional
        channel can be plotted, provided the ``squeeze`` of the channel
        has ``ndim==2`` and the first two axes do not span dimensions
        other than those spanned by that channel.

        Defaults to ``shading="auto"`` to ensure that color boundaries
        are drawn bisecting point positions, when possible.

        Quicker than pcolor

        Parameters
        ----------
        data : 2D WrightTools.data.Data object
            Data to plot.
        channel : int or string (optional)
            Channel index or name. Default is 0.
        dynamic_range : boolean (optional)
            Force plotting of all contours, overloading for major extent. Only applies to signed
            data. Default is False.
        autolabel : {'none', 'both', 'x', 'y'}  (optional)
            Parameterize application of labels directly from data object. Default is none.
        xlabel : string (optional)
            xlabel. Default is None.
        ylabel : string (optional)
            ylabel. Default is None.
        **kwargs
            matplotlib.axes.Axes.pcolormesh__ optional keyword arguments.

            __ https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolormesh.html

        Returns
        -------
        matplotlib.collections.QuadMesh
        """
        args, kwargs = self._parse_plot_args(*args, **kwargs, plot_type="pcolormesh")
        return super().pcolormesh(*args, **kwargs)

    def plot(self, *args, **kwargs):
        """Plot lines and/or markers.

        If a 2D or higher Data object is passed, a lower dimensional
        channel can be plotted, provided the ``squeeze`` of the channel
        has ``ndim==1`` and the first axis does not span dimensions
        other than that spanned by the channel.

        Parameters
        ----------
        data : 1D WrightTools.data.Data object
            Data to plot.
        channel : int or string (optional)
            Channel index or name. Default is 0.
        dynamic_range : boolean (optional)
            Force plotting of all contours, overloading for major extent. Only applies to signed
            data. Default is False.
        autolabel : {'none', 'both', 'x', 'y'}  (optional)
            Parameterize application of labels directly from data object. Default is none.
        xlabel : string (optional)
            xlabel. Default is None.
        ylabel : string (optional)
            ylabel. Default is None.
        **kwargs
            matplotlib.axes.Axes.plot__ optional keyword arguments.

            __ https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html

        Returns
        -------
        list
            list of matplotlib.lines.line2D objects
        """
        args = list(args)  # offer pop, append etc
        # unpack data object, if given
        if isinstance(args[0], Data):
            data = args.pop(0)
            channel = kwargs.pop("channel", 0)
            channel_index = wt_kit.get_index(data.channel_names, channel)
            squeeze = np.array(data.channels[channel_index].shape) == 1
            xa = data.axes[0]
            for sq, xs in zip(squeeze, xa.shape):
                if sq and xs != 1:
                    raise wt_exceptions.ValueError("Cannot squeeze axis to fit channel")
            squeeze = tuple([0 if i else slice(None) for i in squeeze])
            zi = data.channels[channel_index].points
            xi = xa[squeeze]
            if not zi.ndim == 1:
                raise wt_exceptions.DimensionalityError(1, data.ndim)
            args = [xi, zi] + args
        else:
            data = None
            channel_index = 0
        # labels
        self._apply_labels(
            autolabel=kwargs.pop("autolabel", False),
            xlabel=kwargs.pop("xlabel", None),
            ylabel=kwargs.pop("ylabel", None),
            data=data,
            channel_index=channel_index,
        )
        # call parent
        return super().plot(*args, **kwargs)


register_projection(Axes)


class Figure(matplotlib.figure.Figure):
    """Figure."""

    def add_subplot(self, *args, **kwargs):
        """Add a subplot to the figure.

        Parameters
        ----------
        *args
        **kwargs

        Returns
        -------
        WrightTools.artists.Axes object
        """
        kwargs.setdefault("projection", "wright")
        return super().add_subplot(*args, **kwargs)


class GridSpec(matplotlib.gridspec.GridSpec):
    """GridSpec."""

    pass


# --- artist helpers ------------------------------------------------------------------------------


def apply_rcparams(kind="fast"):
    """Quickly apply rcparams for given purposes.

    Parameters
    ----------
    kind: {'default', 'fast', 'publication'} (optional)
        Settings to use. Default is 'fast'.
    """
    if kind == "default":
        matplotlib.rcdefaults()
    elif kind == "fast":
        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.size"] = 14
        matplotlib.rcParams["legend.edgecolor"] = "grey"
        matplotlib.rcParams["contour.negative_linestyle"] = "solid"
    elif kind == "publication":
        matplotlib.rcParams["text.usetex"] = True
        preamble = "\\usepackage[cm]{sfmath}\\usepackage{amssymb}"
        matplotlib.rcParams["text.latex.preamble"] = preamble
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.serif"] = "cm"
        matplotlib.rcParams["font.sans-serif"] = "cm"
        matplotlib.rcParams["font.size"] = 14
        matplotlib.rcParams["legend.edgecolor"] = "grey"
        matplotlib.rcParams["contour.negative_linestyle"] = "solid"


def _order_for_imshow(xi, yi):
    """
    looks at x and y axis shape to determine order of zi axes
    **requires orthogonal, 1D axes**
    returns 2-ple:  the transpose order to apply to zi
    """
    sx = np.array(xi.shape)
    sy = np.array(yi.shape)
    # check that each axis is 1D (i.e. for ndim, number of axes with size 1 is >= ndim - 1 )
    if (sx.prod() == xi.size) and (sy.prod() == yi.size):
        # check that axes are orthogonal and orient z accordingly
        # determine index of x and y axes
        if (sx[0] == 1) and (sy[1] == 1):
            # zi[y,x]
            return (0, 1)
        elif (sx[1] == 1) and (sy[0] == 1):
            # zi[x,y]; imshow expects zi[rows, cols]
            return (1, 0)
        else:
            raise TypeError(f"x and y must be orthogonal; shapes are: {xi.shape}, {yi.shape}")
    else:
        raise TypeError(f"Axes are not 1D: {xi.shape}, {yi.shape}")
