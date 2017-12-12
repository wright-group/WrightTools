"""Tools for visualizing data.

.. _gridspec: http://matplotlib.org/users/gridspec.html#gridspec-and-subplotspec
.. _grayify: https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/
.. _cubehelix: http://arxiv.org/abs/1108.5083.
.. _colormap: http://nbviewer.ipython.org/gist/anonymous/a4fa0adb08f9e9ea4f94
.. _nmtorgb: http://www.physics.sfasu.edu/astro/color/spectra.html
"""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import warnings

import numpy as np

import matplotlib
from matplotlib.axes import SubplotBase, subplot_class_factory
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as grd
import matplotlib.patheffects as PathEffects

import imageio

from .. import exceptions as wt_exceptions
from .. import kit as wt_kit
from ..data import Data
from ._colors import colormaps


# --- define --------------------------------------------------------------------------------------


# string types
if sys.version[0] == '2':
    # recognize unicode and string types
    string_type = basestring  # noqa: F821
else:
    string_type = str  # newer versions of python don't have unicode type


# --- classes -------------------------------------------------------------------------------------


class Axes(matplotlib.axes.Axes):
    """Axes."""

    transposed = False
    is_sideplot = False

    def _parse_cmap(self, data=None, channel_index=None, **kwargs):
        if 'cmap' in kwargs.keys():
            if isinstance(kwargs['cmap'], string_type):
                kwargs['cmap'] = colormaps[kwargs['cmap']]
        elif data:
            if data.channels[channel_index].signed:
                kwargs['cmap'] = colormaps['signed']
                return kwargs
            kwargs['cmap'] = colormaps['default']
        return kwargs

    def _apply_labels(self, autolabel='none', xlabel=None, ylabel=None, data=None, channel_index=0):
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
        if autolabel in ['xy', 'both', 'x'] and not xlabel:
            xlabel = data.axes[0].label
        if autolabel in ['xy', 'both', 'y'] and not ylabel:
            if data.dimensionality == 1:
                ylabel = data.channels[channel_index].label
            elif data.dimensionality == 2:
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
        if zi is not None:
            if 'levels' in kwargs.keys():
                levels = kwargs['levels']
                vmin = levels.min()
                vmax = levels.max()
            else:
                vmin = np.nanmin(zi)
                vmax = np.nanmax(zi)
        elif data is not None:
            signed = data.channels[channel_index].signed
            if signed and dynamic_range:
                vmin = -data.channels[channel_index].minor_extent
                vmax = +data.channels[channel_index].minor_extent
            elif signed and not dynamic_range:
                vmin = -data.channels[channel_index].major_extent
                vmax = +data.channels[channel_index].major_extent
            else:
                vmin = data.channels[channel_index].null
                vmax = data.channels[channel_index].max()
            # don't overwrite
            if 'vmin' not in kwargs.keys():
                kwargs['vmin'] = vmin
            if 'vmax' not in kwargs.keys():
                kwargs['vmax'] = vmax
        return kwargs

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
        if hasattr(self, 'divider'):
            divider = self.divider
        else:
            divider = make_axes_locatable(self)
            setattr(self, 'divider', divider)
        # create
        if along == 'x':
            ax = self.sidex = divider.append_axes('top', height, pad=pad, sharex=self)
        elif along == 'y':
            ax = self.sidey = divider.append_axes('right', height, pad=pad, sharey=self)
            ax.transposed = True
        # beautify
        if along == 'x':
            ax.set_ylim(ymin, ymax)
        elif along == 'y':
            ax.set_xlim(ymin, ymax)
        ax.autoscale(enable=False)
        ax.set_adjustable('box-forced')
        ax.is_sideplot = True
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        return ax

    def contour(self, *args, **kwargs):
        """Plot contours.

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
        args = list(args)  # offer pop, append etc
        channel = kwargs.pop('channel', 0)
        dynamic_range = kwargs.pop('dynamic_range', False)
        # unpack data object, if given
        if isinstance(args[0], Data):
            data = args.pop(0)
            if not data.dimensionality == 2:
                raise wt_exceptions.DimensionalityError(2, data.dimensionality)
            # arrays
            channel_index = wt_kit.get_index(data.channel_names, channel)
            signed = data.channels[channel_index].signed
            xi = data.axes[0][:]
            yi = data.axes[1][:]
            zi = data.channels[channel_index][:].T
            args = [xi, yi, zi] + args
            # limits
            kwargs = self._parse_limits(data=data, channel_index=channel_index,
                                        dynamic_range=dynamic_range, **kwargs)
            # levels
            if 'levels' not in kwargs.keys():
                if signed:
                    n = 11
                else:
                    n = 6
                kwargs['levels'] = np.linspace(kwargs.pop('vmin'), kwargs.pop('vmax'), n)[1:-1]
            # colors
            if 'colors' not in kwargs.keys():
                kwargs['colors'] = 'k'
            if 'alpha' not in kwargs.keys():
                kwargs['alpha'] = 0.5
            # labels
            self._apply_labels(autolabel=kwargs.pop('autolabel', False),
                               xlabel=kwargs.pop('xlabel', None),
                               ylabel=kwargs.pop('ylabel', None),
                               data=data, channel_index=channel_index)
        else:
            data = None
            channel_index = 0
            signed = False
            kwargs = self._parse_limits(zi=args[2], dynamic_range=dynamic_range, **kwargs)
        # call parent
        return matplotlib.axes.Axes.contour(self, *args, **kwargs)  # why can't I use super?

    def contourf(self, *args, **kwargs):
        """Plot contours.

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
        args = list(args)  # offer pop, append etc
        channel = kwargs.pop('channel', 0)
        dynamic_range = kwargs.pop('dynamic_range', False)
        # unpack data object, if given
        if isinstance(args[0], Data):
            data = args.pop(0)
            if not data.dimensionality == 2:
                raise wt_exceptions.DimensionalityError(2, data.dimensionality)
            # arrays
            channel_index = wt_kit.get_index(data.channel_names, channel)
            xi = data.axes[0][:]
            yi = data.axes[1][:]
            zi = data.channels[channel_index][:].T
            args = [xi, yi, zi] + args
            # limits
            kwargs = self._parse_limits(data=data, channel_index=channel_index,
                                        dynamic_range=dynamic_range, **kwargs)
            # cmap
            kwargs = self._parse_cmap(data=data, channel_index=channel_index, **kwargs)
        else:
            data = None
            channel_index = 0
            kwargs = self._parse_limits(zi=args[2], dynamic_range=dynamic_range, **kwargs)
            kwargs = self._parse_cmap(**kwargs)
        # levels
        if 'levels' not in kwargs.keys():
            kwargs['levels'] = np.linspace(kwargs.pop('vmin'), kwargs.pop('vmax'), 256)
        # labels
        self._apply_labels(autolabel=kwargs.pop('autolabel', False),
                           xlabel=kwargs.pop('xlabel', None),
                           ylabel=kwargs.pop('ylabel', None),
                           data=data, channel_index=channel_index)
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
        kwargs['antialiased'] = False
        kwargs['extend'] = 'both'
        contours = matplotlib.axes.Axes.contourf(self, *args, **kwargs)  # why can't I use super?
        # fill lines
        zorder = contours.collections[0].zorder - 0.1
        levels = (contours.levels[1:] + contours.levels[:-1]) / 2
        matplotlib.axes.Axes.contour(self, *args[:3], levels=levels,
                                     cmap=contours.cmap,
                                     zorder=zorder)
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
        if 'fancybox' not in kwargs.keys():
            kwargs['fancybox'] = False
        if 'framealpha' not in kwargs.keys():
            kwargs['framealpha'] = 1.
        return super().legend(*args, **kwargs)

    def pcolor(self, *args, **kwargs):
        """Create a pseudocolor plot of a 2-D array.

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
        args = list(args)  # offer pop, append etc
        channel = kwargs.pop('channel', 0)
        dynamic_range = kwargs.pop('dynamic_range', False)
        # unpack data object, if given
        if isinstance(args[0], Data):
            data = args.pop(0)
            if not data.dimensionality == 2:
                raise wt_exceptions.DimensionalityError(2, data.dimensionality)
            # arrays
            channel_index = wt_kit.get_index(data.channel_names, channel)
            xi = data.axes[0][:]
            yi = data.axes[1][:]
            zi = data.channels[channel_index][:].T
            X, Y, Z = pcolor_helper(xi, yi, zi)
            args = [X, Y, Z] + args
            # limits
            kwargs = self._parse_limits(data=data, channel_index=channel_index,
                                        dynamic_range=dynamic_range, **kwargs)
            # cmap
            kwargs = self._parse_cmap(data=data, channel_index=channel_index, **kwargs)
        else:
            data = None
            channel_index = 0
            kwargs = self._parse_limits(zi=args[2], **kwargs)
            kwargs = self._parse_cmap(**kwargs)
        # labels
        self._apply_labels(autolabel=kwargs.pop('autolabel', False),
                           xlabel=kwargs.pop('xlabel', None),
                           ylabel=kwargs.pop('ylabel', None),
                           data=data, channel_index=channel_index)
        # call parent
        return matplotlib.axes.Axes.pcolor(self, *args, **kwargs)  # why can't I use super?

    def plot(self, *args, **kwargs):
        """Plot lines and/or markers.

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
            matplotlib.axes.Axes.pcolor__ optional keyword arguments.

        __ https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html

        Returns
        -------
        list
            list of matplotlib.lines.line2D objects
        """
        args = list(args)  # offer pop, append etc
        # unpack data object, if given
        if hasattr(args[0], 'id'):  # TODO: replace once class comparison works...
            data = args.pop(0)
            channel = kwargs.pop('channel', 0)
            if not data.dimensionality == 1:
                raise wt_exceptions.DimensionalityError(1, data.dimensionality)
            # arrays
            channel_index = wt_kit.get_index(data.channel_names, channel)
            xi = data.axes[0][:]
            zi = data.channels[channel_index][:].T
            args = [xi, zi] + args
        else:
            data = None
            channel_index = 0
        # labels
        self._apply_labels(autolabel=kwargs.pop('autolabel', False),
                           xlabel=kwargs.pop('xlabel', None),
                           ylabel=kwargs.pop('ylabel', None),
                           data=data, channel_index=channel_index)
        # call parent
        return matplotlib.axes.Axes.plot(self, *args, **kwargs)  # why can't I use super?

    def plot_data(self, data, channel=0, interpolate=False, coloring=None,
                  xlabel=True, ylabel=True, min=None, max=None):
        """DEPRECATED.

        Parameters
        ----------
        data : WrightTools.data.Data object
            The data object to plot.
        channel : int or str (optional)
            The channel to plot. Default is 0.
        interpolate : boolean (optional)
            Toggle interpolation. Default is False.
        coloring : str (optional)
            A key to the colormaps dictionary found in artists module, for
            two-dimensional data. A matplotlib color string for
            one-dimensional data. Default is None.
        xlabel : boolean (optional)
            Toggle xlabel. Default is True.
        ylabel : boolean (optional)
            Toggle ylabel. Default is True.
        min : number (optional)
            min. Default is None (inherited from channel).
        max : number (optional)
            max. Default is None (inherited from channel).

        .. plot::

           >>> import matplotlib
           >>> from matplotlib import pyplot as plt
           >>> plt.plot(range(10))

        """
        message = "plot_data is deprecated---use plot methods directly"
        warnings.warn(wt_exceptions.VisibleDeprecationWarning(message))
        # prepare ---------------------------------------------------------------------------------
        # get dimensionality
        # get channel
        channel_index = wt_kit.get_index(data.channel_names, channel)
        channel = data.channels[channel_index]
        # get axes
        xaxis = data.axes[0]
        # get min
        if min is None:
            min = channel.min()
        # get max
        if max is None:
            max = channel.max()
        # 1D --------------------------------------------------------------------------------------
        if data.dimensionality == 1:
            # get list of all datas
            # get color
            if coloring is None:
                c = self._get_lines.get_next_color()
            else:
                c = coloring
            # get arrays
            xi = xaxis[:]
            yi = data.channels[channel_index][:]
            # plot
            if interpolate:
                self.plot(xi, yi, c=c)
            else:
                self.scatter(xi, yi, c=c)
            # decoration
            if self.get_adjustable() == 'datalim':
                self.set_xlim(xi.min(), xi.max())
                self.set_ylim(min, max)
            # transposed catcher
            if self.transposed:
                for line in self.lines:
                    xdata, ydata = line.get_xdata(), line.get_ydata()
                    line.set_xdata(ydata)
                    line.set_ydata(xdata)
        # 2D --------------------------------------------------------------------------------------
        elif data.dimensionality == 2:
            yaxis = data.axes[1]
            # get colormap
            if coloring is None:
                if channel.signed:
                    cmap = colormaps['signed']
                else:
                    cmap = colormaps['default']
            else:
                cmap = colormaps[coloring]
            # get arrays
            xi = xaxis[:]
            yi = yaxis[:]
            zi = channel[:].T
            # plot
            if interpolate:
                # contourf
                levels = np.linspace(min, max, 256)
                self.contourf(xi, yi, zi, levels=levels, cmap=cmap)
            else:
                # pcolor
                X, Y, Z = pcolor_helper(xi, yi, zi)
                self.pcolor(X, Y, Z, vmin=min, vmax=max, cmap=cmap)
            # decoration
            self.set_xlim(xi.min(), xi.max())
            self.set_ylim(yi.min(), yi.max())
        # ND --------------------------------------------------------------------------------------
        else:
            pass
        # decoration ------------------------------------------------------------------------------
        if xlabel and not self.is_sideplot:
            self.set_xlabel(xaxis.label, fontsize=18)
        if ylabel and not self.is_sideplot:
            if data.dimensionality == 1:
                self.set_ylabel(channel.label, fontsize=18)
            if data.dimensionality == 2:
                self.set_ylabel(yaxis.label, fontsize=18)


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
        # projection
        if 'projection' not in kwargs.keys():
            projection = 'wright'
        else:
            projection = kwargs['projection']
        # must be arguments
        if not len(args):
            return
        # int args must be correct
        if len(args) == 1 and isinstance(args[0], int):
            args = tuple([int(c) for c in str(args[0])])
            if len(args) != 3:
                raise ValueError("Integer subplot specification must " +
                                 "be a three digit number.  " +
                                 "Not {n:d}".format(n=len(args)))
        # subplotbase args
        if isinstance(args[0], SubplotBase):
            a = args[0]
            if a.get_figure() is not self:
                msg = ("The Subplot must have been created in the present"
                       " figure")
                raise ValueError(msg)
            # make a key for the subplot (which includes the axes object id
            # in the hash)
            key = self._make_key(*args, **kwargs)
        else:
            projection_class, kwargs, key = matplotlib.figure.process_projection_requirements(
                self, *args, **kwargs)
            # try to find the axes with this key in the stack
            ax = self._axstack.get(key)
            if ax is not None:
                if isinstance(ax, projection_class):
                    # the axes already existed, so set it as active & return
                    self.sca(ax)
                    return ax
                else:
                    # Undocumented convenience behavior:
                    # subplot(111); subplot(111, projection='polar')
                    # will replace the first with the second.
                    # Without this, add_subplot would be simpler and
                    # more similar to add_axes.
                    self._axstack.remove(ax)
            if projection == 'wright':
                a = subplot_class_factory(Axes)(self, *args, **kwargs)
            else:
                a = subplot_class_factory(projection_class)(self, *args, **kwargs)
        self._axstack.add(key, a)
        self.sca(a)
        if int(matplotlib.__version__.split('.')[0]) > 1:
            a._remove_method = self.__remove_ax
            self.stale = True
            a.stale_callback = matplotlib.figure._stale_figure_callback
        # finish
        return a


class GridSpec(matplotlib.gridspec.GridSpec):
    """GridSpec."""

    pass


# --- artist helpers ------------------------------------------------------------------------------


def _title(fig, title, subtitle='', margin=1, fontsize=20, subfontsize=18):
    fig.suptitle(title, fontsize=fontsize)
    height = fig.get_figheight()  # inches
    distance = margin / 2.  # distance from top of plot, in inches
    ratio = 1 - distance / height
    fig.text(0.5, ratio, subtitle, fontsize=subfontsize, ha='center', va='top')


def add_sideplot(ax, along, pad=0., grid=True, zero_line=True,
                 arrs_to_bin=None, normalize_bin=True, ymin=0, ymax=1.1,
                 height=0.75, c='C0'):
    """Add a sideplot to an axis. Sideplots share their corresponding axis.

    Parameters
    ----------
    ax : matplotlib AxesSubplot object
        The axis to add a sideplot along.
    along : {'x', 'y'}
        The dimension to add a sideplot along.
    pad : number (optional)
        Distance between axis and sideplot. Default is 0.
    grid : bool (optional)
        Toggle for plotting grid on sideplot. Default is True.
    zero_line : bool (optional)
        Toggle for plotting black line at zero signal. Default is True.
    arrs_to_bin : list [xi, yi, zi] (optional)
        Bins are plotted if arrays are supplied. Default is None.
    normalize_bin : bool (optional)
        Normalize bin by max value. Default is True.
    ymin : number (optional)
        Bin minimum extent. Default is 0.
    ymax : number (optional)
        Bin maximum extent. Default is 1.1
    c : string (optional)
        Line color. Default is C0.

    Returns
    -------
    axCorr
        AxesSubplot object
    """
    # divider should only be created once
    if hasattr(ax, 'WrightTools_sideplot_divider'):
        divider = ax.WrightTools_sideplot_divider
    else:
        divider = make_axes_locatable(ax)
        setattr(ax, 'WrightTools_sideplot_divider', divider)
    # create sideplot axis
    if along == 'x':
        axCorr = divider.append_axes('top', height, pad=pad, sharex=ax)
    elif along == 'y':
        axCorr = divider.append_axes('right', height, pad=pad, sharey=ax)
    axCorr.autoscale(False)
    axCorr.set_adjustable('box-forced')
    # bin
    if arrs_to_bin is not None:
        xi, yi, zi = arrs_to_bin
        if along == 'x':
            b = np.nansum(zi, axis=0) * len(yi)
            if normalize_bin:
                b /= np.nanmax(b)
            axCorr.plot(xi, b, c=c, lw=2)
        elif along == 'y':
            b = np.nansum(zi, axis=1) * len(xi)
            if normalize_bin:
                b /= np.nanmax(b)
            axCorr.plot(b, yi, c=c, lw=2)
    # beautify
    if along == 'x':
        axCorr.set_ylim(ymin, ymax)
    elif along == 'y':
        axCorr.set_xlim(ymin, ymax)
    plt.grid(grid)
    if zero_line:
        if along == 'x':
            plt.axhline(0, c='k', lw=1)
        elif along == 'y':
            plt.axvline(0, c='k', lw=1)
    plt.setp(axCorr.get_xticklabels(), visible=False)
    plt.setp(axCorr.get_yticklabels(), visible=False)
    return axCorr


def apply_rcparams(kind='fast'):
    """Quickly apply rcparams for given purposes.

    Parameters
    ----------
    kind: {'default', 'fast', 'publication'} (optional)
        Settings to use. Default is 'fast'.
    """
    if kind == 'default':
        matplotlib.rcdefaults()
    elif kind == 'fast':
        matplotlib.rcParams['text.usetex'] = False
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['legend.edgecolor'] = 'grey'
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    elif kind == 'publication':
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['text.latex.unicode'] = True
        matplotlib.rcParams['text.latex.preamble'] = '\\usepackage[cm]{sfmath}'
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.serif'] = 'cm'
        matplotlib.rcParams['font.sans-serif'] = 'cm'
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['legend.edgecolor'] = 'grey'
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'


def corner_text(text, distance=0.075, ax=None, corner='UL', factor=200, bbox=True,
                fontsize=18, background_alpha=1, edgecolor=None):
    """Place some text in the corner of the figure.

    Parameters
    ----------
    text : str
        The text to use.
    distance : number (optional)
        Distance from the corner. Default is 0.05.
    ax : axis (optional)
        The axis object to label. If None, uses current axis. Default is None.
    corner : {'UL', 'LL', 'UR', 'LR'} (optional)
        The corner to label. Upper left, Lower left etc. Default is UL.
    factor : number (optional)
        Scaling factor. Default is 200.
    bbox : boolean (optional)
        Toggle bounding box. Default is True.
    fontsize : number (optional)
        Text fontsize. If None, uses the matplotlib default. Default is 18.
    background_alpha : number (optional)
        Opacity of background bounding box. Default is 1.
    edgecolor : string (optional)
        Frame edgecolor. Default is None (inherits from legend.edgecolor
        rcparam).

    Returns
    -------
    text
        The matplotlib text object.
    """
    # get axis
    if ax is None:
        ax = plt.gca()
    [h_scaled, v_scaled], [va, ha] = get_scaled_bounds(ax, corner, distance, factor)
    # get edgecolor
    if edgecolor is None:
        edgecolor = matplotlib.rcParams['legend.edgecolor']
    # apply text
    props = dict(boxstyle='square', facecolor='white', alpha=background_alpha,
                 edgecolor=edgecolor)
    args = [v_scaled, h_scaled, text]
    kwargs = {}
    kwargs['fontsize'] = fontsize
    kwargs['verticalalignment'] = va
    kwargs['horizontalalignment'] = ha
    if bbox:
        kwargs['bbox'] = props
    else:
        kwargs['path_effects'] = [PathEffects.withStroke(linewidth=3, foreground="w")]
    kwargs['transform'] = ax.transAxes
    if 'zlabel' in ax.properties().keys():  # axis is 3D projection
        out = ax.text2D(*args, **kwargs)
    else:
        out = ax.text(*args, **kwargs)
    return out


def create_figure(width='single', nrows=1, cols=[1], margin=1.,
                  hspace=0.25, wspace=0.25, cbar_width=0.25, aspects=[],
                  default_aspect=1):
    """Re-parameterization of matplotlib figure creation tools, exposing convenient variables.

    Figures are defined primarily by their width. Height is defined by the
    aspect ratios of the subplots contained within. hspace, wspace, and
    cbar_width are defined in inches, making it easier to make consistent
    plots. Margins are enforced to be equal around the entire plot, starting
    from the edges of the subplots.

    Parameters
    ----------
    width : {'single', 'double', 'dissertation'} or float (optional)
        The total width of the generated figure. Can be given in inches
        directly, or can be specified using keys. Default is 'single' (6.5
        inches).
    nrows : int (optional)
        The number of subplot rows in the figure. Default is 1.
    cols : list (optional)
        A list of numbers, defining the number and width-ratios of the
        figure columns. May also contain the special string 'cbar', defining
        a column as a colorbar-containing column. Default is [1].
    margin : float (optional)
        Margin in inches. Margin is applied evenly around the figure, starting
        from the subplot boundaries (so that ticks and labels appear in the
        margin). Default is 1.
    hspace : float (optional)
        The 'height space' (space seperating two subplots vertically), in
        inches. Default is 0.25.
    wspace : float (optional)
        The 'width space' (space seperating two subplots horizontally), in
        inches. Default is 0.25.
    cbar_width : float (optional)
        The width of the colorbar in inches. Default is 0.25.
    aspects : list of lists (optional)
        Define the aspect ratio of individual subplots. List of lists, each
        sub-ist having the format [[row, col], aspect]. The figure will expand
        vertically to acomidate the defined aspect ratio. Aspects are V/H so
        aspects larger than 1 will be taller than wide and vice-versa for
        aspects smaller than 1. You may only define the aspect for one subplot
        in each row. If no aspect is defined for a particular row, the leftmost
        subplot will have an aspect of ``default_aspect``. Default is [].

    Returns
    -------
    tuple
        (matplotlib.figure.Figure, matplotlib.gridspec.GridSpec). GridSpec
        contains SubplotSpec objects that can have axes placed into them.
        The SubplotSpec objects can be accessed through indexing: [row, col].
        Slicing works, for example ``cax = plt.subplot(gs[:, -1])``. See
        matplotlib gridspec_ documentation for more information.


    Notes
    -----
    To ensure the margins work as expected, save the fig with
    the same margins (``pad_inches``) as specified in this function. Common
    savefig call:
    ``plt.savefig(plt.savefig(output_path, dpi=300, transparent=True,
    pad_inches=1))``

    See Also
    --------
    wt.artists.plot_margins
        Plot lines to visualize the figure edges, margins, and centers. For
        debug and design purposes.
    wt.artsits.subplots_adjust
        Enforce margins for figure generated elsewhere.

    """
    # get width
    if width == 'double':
        figure_width = 14.
    elif width == 'single':
        figure_width = 6.5
    elif width == 'dissertation':
        figure_width = 13.
    else:
        figure_width = float(width)
    # check if aspect constraints are valid
    rows_in_aspects = [item[0][0] for item in aspects]
    if not len(rows_in_aspects) == len(set(rows_in_aspects)):
        raise Exception('can only specify aspect for one plot in each row')
    # get width avalible to subplots (not including colorbars)
    total_subplot_width = figure_width - 2 * margin
    total_subplot_width -= (len(cols) - 1) * wspace  # whitespace in width
    total_subplot_width -= cols.count('cbar') * cbar_width  # colorbar width
    # converters

    def in_to_mpl(inches, total, n):
        return (inches * n) / (total - inches * n + inches)

    def mpl_to_in(mpl, total, n):
        return (total / (n + mpl * (n - 1))) * mpl
    # calculate column widths, width_ratio
    subplot_ratios = np.array([i for i in cols if not i == 'cbar'], dtype=np.float)
    subplot_ratios /= sum(subplot_ratios)
    subplot_widths = total_subplot_width * subplot_ratios
    width_ratios = []
    cols_idxs = []
    i = 0
    for col in cols:
        if not col == 'cbar':
            width_ratios.append(subplot_widths[i])
            cols_idxs.append(i)
            i += 1
        else:
            width_ratios.append(cbar_width)
            cols_idxs.append(np.nan)
    # calculate figure height, height_ratios, figure height
    subplot_heights = []
    for row_index in range(nrows):
        if row_index in rows_in_aspects:
            aspect = aspects[rows_in_aspects.index(row_index)][1]
            col_index = aspects[rows_in_aspects.index(row_index)][0][1]
            height = subplot_widths[col_index] * aspect
            subplot_heights.append(height)
        else:
            # make the leftmost (zero indexed) plot square as default
            subplot_heights.append(subplot_widths[0] * default_aspect)
    height_ratios = subplot_heights
    figure_height = sum(subplot_heights)
    figure_height += (nrows - 1) * hspace
    figure_height += 2 * margin
    # make figure
    fig = plt.figure(figsize=[figure_width, figure_height], FigureClass=Figure)
    # get hspace, wspace in relative units
    hspace = in_to_mpl(hspace, figure_height - 2 * margin, nrows)
    wspace = in_to_mpl(wspace, figure_width - 2 * margin, len(cols))
    # make gridpsec
    gs = GridSpec(nrows, len(cols), hspace=hspace, wspace=wspace,
                  width_ratios=width_ratios, height_ratios=height_ratios)
    # finish
    subplots_adjust(fig, inches=margin)
    return fig, gs


def diagonal_line(xi, yi, ax=None, c='k', ls=':', lw=1, zorder=3):
    """Plot a diagonal line.

    Parameters
    ----------
    xi : 1D array-like
        The x axis points.
    yi : 1D array-like
        The y axis points.
    ax : axis (optional)
        Axis to plot on. If none is supplied, the current axis is used.
    c : string (optional)
        Line color. Default is k.
    ls : string (optional)
        Line style. Default is : (dotted).
    lw : float (optional)
        Line width. Default is 1.
    zorder : number (optional)
        Matplotlib zorder. Default is 3.

    Returns
    -------
    matplotlib.lines.Line2D object
        The plotted line.
    """
    # get axis
    if ax is None:
        ax = plt.gca()
    # make plot
    diag_min = max(min(xi), min(yi))
    diag_max = min(max(xi), max(yi))
    line = ax.plot([diag_min, diag_max], [diag_min, diag_max], c=c, ls=ls, lw=lw, zorder=zorder)
    return line


def get_color_cycle(n, cmap='rainbow', rotations=3):
    """Get a list of RGBA colors following a colormap.

    Useful for plotting lots of elements, keeping the color of each unique.

    Parameters
    ----------
    n : integer
        The number of colors to return.
    cmap : string (optional)
        The colormap to use in the cycle. Default is rainbow.
    rotations : integer (optional)
        The number of times to repeat the colormap over the cycle. Default is 3.

    Returns
    -------
    list
        List of RGBA lists.
    """
    cmap = colormaps[cmap]
    if np.mod(n, rotations) == 0:
        per = np.floor_divide(n, rotations)
    else:
        per = np.floor_divide(n, rotations) + 1
    vals = list(np.linspace(0, 1, per))
    vals = vals * rotations
    vals = vals[:n]
    out = cmap(vals)
    return out


def get_constant_text(constants):
    """Get a nicely formatted string representing all constants.

    Parameters
    ----------
    constants : list of WrightTools.data.Axis objects
        The constants to be formatted.

    Returns
    -------
    string
        The constant text.
    """
    string_list = [constant.get_label(show_units=True, points=True) for constant in constants]
    text = '    '.join(string_list)
    return text


def get_scaled_bounds(ax, position, distance=0.1, factor=200):
    """Get scaled bounds.

    Parameters
    ----------
    ax : Axes object
        Axes object.
    position : {'UL', 'LL', 'UR', 'LR'}
        Position.
    distance : number (optional)
        Distance. Default is 0.1.
    factor : number (optional)
        Factor. Default is 200.

    Returns
    -------
    ([h_scaled, v_scaled], [va, ha])
    """
    # get bounds
    x0, y0, width, height = ax.bbox.bounds
    width_scaled = width / factor
    height_scaled = height / factor
    # get scaled postions
    if position == 'UL':
        v_scaled = distance / width_scaled
        h_scaled = 1 - (distance / height_scaled)
        va = 'top'
        ha = 'left'
    elif position == 'LL':
        v_scaled = distance / width_scaled
        h_scaled = distance / height_scaled
        va = 'bottom'
        ha = 'left'
    elif position == 'UR':
        v_scaled = 1 - (distance / width_scaled)
        h_scaled = 1 - (distance / height_scaled)
        va = 'top'
        ha = 'right'
    elif position == 'LR':
        v_scaled = 1 - (distance / width_scaled)
        h_scaled = distance / height_scaled
        va = 'bottom'
        ha = 'right'
    else:
        print('corner not recognized')
        v_scaled = h_scaled = 1.
        va = 'center'
        ha = 'center'
    return [h_scaled, v_scaled], [va, ha]


def grayify_cmap(cmap):
    """Return a grayscale version of the colormap.

    `Source`__

     __ grayify_
    """
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)


def pcolor_helper(xi, yi, zi, transform=None):
    """Prepare a set of arrays for plotting using `pcolor`.

    The return values are suitable for feeding directly into ``matplotlib.pcolor``
    such that the pixels are properly centered.

    Parameters
    ----------
    xi : 1D array-like
        1D array of X-coordinates.
    yi : 1D array-like
        1D array of Y-coordinates.
    zi : 2D array-like
        Rectangular array of Z-coordinates.
    transform : function
        Transform function.

    Returns
    -------
    X : 2D ndarray
        X dimension for pcolor
    Y : 2D ndarray
        Y dimension for pcolor
    Z : 2D ndarray
        Z dimension for pcolor
    """
    x_points = np.zeros(len(xi) + 1)
    y_points = np.zeros(len(yi) + 1)
    for points, axis in [[x_points, xi], [y_points, yi]]:
        for j in range(len(points)):
            if j == 0:  # first point
                points[j] = axis[0] - (axis[1] - axis[0])
            elif j == len(points) - 1:  # last point
                points[j] = axis[-1] + (axis[-1] - axis[-2])
            else:
                points[j] = np.average([axis[j], axis[j - 1]])
    X, Y = np.meshgrid(x_points, y_points)
    if isinstance(transform, type(None)):
        return X, Y, zi
    else:
        for (x, y), value in np.ndenumerate(X):
            X[x][y], Y[x][y] = transform((X[x][y], Y[x][y]))
        return X, Y, zi


def plot_colorbar(cax=None, cmap='default', ticks=None, clim=None, vlim=None,
                  label=None, tick_fontsize=14, label_fontsize=18, decimals=None,
                  orientation='vertical', ticklocation='auto'):
    """Easily add a colormap to an axis.

    Parameters
    ----------
    cax : matplotlib axis (optional)
        The axis to plot the colorbar on. Finds the current axis if none is
        given.
    cmap : string or LinearSegmentedColormap (optional)
        The colormap to fill the colorbar with. Strings map as keys to the
        WrightTools colormaps dictionary. Default is `default`.
    ticks : 1D array-like (optional)
        Ticks. Default is None.
    clim : two element list (optional)
        The true limits of the colorbar, in the same units as ticks. If None,
        streaches the colorbar over the limits of ticks. Default is None.
    vlim : two element list-like (optional)
        The limits of the displayed colorbar, in the same units as ticks. If
        None, displays over clim. Default is None.
    label : str (optional)
        Label. Default is None.
    tick_fontsize : number (optional)
        Fontsize. Default is 14.
    label_fontsize : number (optional)
        Label fontsize. Default is 18.
    decimals : integer (optional)
        Number of decimals to appear in tick labels. Default is None (best guess).
    orientation : {'vertical', 'horizontal'} (optional)
        Colorbar orientation. Default is vertical.
    ticklocation : {'auto', 'left', 'right', 'top', 'bottom'} (optional)
        Tick location. Default is auto.

    Returns
    -------
    matplotlib.colorbar.ColorbarBase object
        The created colorbar.
    """
    # parse cax
    if cax is None:
        cax = plt.gca()
    # parse cmap
    if isinstance(cmap, string_type):
        cmap = colormaps[cmap]
    # parse ticks
    if ticks is None:
        ticks = np.linspace(0, 1, 11)
    # parse clim
    if clim is None:
        clim = [min(ticks), max(ticks)]
    # parse clim
    if vlim is None:
        vlim = clim
    # parse format
    if isinstance(decimals, int):
        format = '%.{0}f'.format(decimals)
    else:
        magnitude = int(np.log10(max(vlim) - min(vlim)) - 0.99)
        if 1 > magnitude > -3:
            format = '%.{0}f'.format(-magnitude + 1)
        elif magnitude in (1, 2, 3):
            format = '%i'
        else:
            # scientific notation
            def fmt(x, _):
                return '%.1f' % (x / float(10 ** magnitude))
            format = matplotlib.ticker.FuncFormatter(fmt)
            magnitude_label = r'  $\mathsf{\times 10^{%d}}$' % magnitude
            if label is None:
                label = magnitude_label
            else:
                label = ' '.join([label, magnitude_label])
    # make cbar
    norm = matplotlib.colors.Normalize(vmin=vlim[0], vmax=vlim[1])
    cbar = matplotlib.colorbar.ColorbarBase(ax=cax, cmap=cmap,
                                            norm=norm, ticks=ticks,
                                            orientation=orientation,
                                            ticklocation=ticklocation,
                                            format=format)
    # coerce properties
    cbar.set_clim(clim[0], clim[1])
    cbar.ax.tick_params(labelsize=tick_fontsize)
    if label:
        cbar.set_label(label, fontsize=label_fontsize)
    # finish
    return cbar


def savefig(path, fig=None, close=True, dpi=300):
    """Save a figure.

    Parameters
    ----------
    path : str
        Path to save figure at.
    fig : matplotlib.figure.Figure object (optional)
        The figure to plot onto. If None, gets current figure. Default is None.
    close : bool (optional)
        Toggle closing of figure after saving. Default is True.

    Returns
    -------
    str
        The full path where the figure was saved.
    """
    # get fig
    if fig is None:
        fig = plt.gcf()
    # get full path
    path = os.path.abspath(path)
    # save
    plt.savefig(path, dpi=dpi, transparent=False, pad_inches=1,
                facecolor='none')
    # close
    if close:
        plt.close(fig)
    # finish
    return path


def set_ax_labels(ax=None, xlabel=None, ylabel=None, xticks=None, yticks=None,
                  label_fontsize=18):
    """Set all axis labels properties easily.

    Parameters
    ----------
    ax : matplotlib AxesSubplot object (optional)
        Axis to set. If None, uses current axis. Default is None.
    xlabel : None or string (optional)
        x axis label. Default is None.
    ylabel : None or string (optional)
        y axis label. Default is None.
    xticks : None or False or list of numbers
        xticks. If False, ticks are hidden. Default is None.
    yticks : None or False or list of numbers
        yticks. If False, ticks are hidden. Default is None.
    label_fontsize : number
        Fontsize of label. Default is 18.

    See Also
    --------
    set_fig_labels
    """
    # get ax
    if ax is None:
        ax = plt.gca()
    # x
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if xticks is not None:
        if isinstance(xticks, bool):
            plt.setp(ax.get_xticklabels(), visible=xticks)
            if not xticks:
                ax.tick_params(axis='x', which='both', length=0)
        else:
            ax.set_xticks(xticks)
    # y
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if yticks is not None:
        if isinstance(yticks, bool):
            plt.setp(ax.get_yticklabels(), visible=yticks)
            if not yticks:
                ax.tick_params(axis='y', which='both', length=0)
        else:
            ax.set_yticks(yticks)


def set_ax_spines(ax=None, c='k', lw=3, zorder=10):
    """Easily the properties of all four axis spines.

    Parameters
    ----------
    ax : matplotlib AxesSubplot object (optional)
        Axis to set. If None, uses current axis. Default is None.
    c : any matplotlib color argument (optional)
        Spine color. Default is k.
    lw : number (optional)
        Spine linewidth. Default is 3.
    zorder : number (optional)
        Spine zorder. Default is 10.
    """
    # get ax
    if ax is None:
        ax = plt.gca()
    # apply
    for key in ['bottom', 'top', 'right', 'left']:
        ax.spines[key].set_color(c)
        ax.spines[key].set_linewidth(lw)
        ax.spines[key].zorder = zorder


def set_fig_labels(fig=None, xlabel=None, ylabel=None, xticks=None, yticks=None,
                   title=None, label_fontsize=18, title_fontsize=20):
    """Set all axis labels of a figure simultaniously.

    Only plots ticks and labels for edge axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure object (optional)
        Figure to set labels of. If None, uses current figure. Default is None.
    xlabel : None or string (optional)
        x axis label. Default is None.
    ylabel : None or string (optional)
        y axis label. Default is None.
    xticks : None or False or list of numbers (optional)
        xticks. If False, ticks are hidden. Default is None.
    yticks : None or False or list of numbers (optional)
        yticks. If False, ticks are hidden. Default is None.
    title : None or string (optional)
        Title of figure. Default is None.
    label_fontsize : number
        Fontsize of label. Default is 18.
    title_fontsize : number
        Fontsize of title. Default is 20.

    See Also
    --------
    set_ax_labels
    """
    # get fig
    if fig is None:
        fig = plt.gcf()
    # axes
    for ax in fig.axes:
        if ax.is_sideplot:
            continue
        elif ax.is_first_col() and ax.is_last_row():
            # lower left corner
            set_ax_labels(
                ax=ax,
                xlabel=xlabel,
                ylabel=ylabel,
                xticks=xticks,
                yticks=yticks,
                label_fontsize=label_fontsize)
        elif ax.is_first_col():
            # lefthand column
            set_ax_labels(
                ax=ax,
                ylabel=ylabel,
                xticks=False,
                yticks=yticks,
                label_fontsize=label_fontsize)
        elif ax.is_last_row():
            # bottom row
            set_ax_labels(
                ax=ax,
                xlabel=xlabel,
                xticks=xticks,
                yticks=False,
                label_fontsize=label_fontsize)
        else:
            set_ax_labels(ax=ax, xticks=False, yticks=False)
    # title
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)


def plot_gridlines(ax=None, c='grey', lw=1, diagonal=False, zorder=2,
                   makegrid=True):
    """Plot dotted gridlines onto an axis.

    Parameters
    ----------
    ax : matplotlib AxesSubplot object (optional)
        Axis to add gridlines to. If None, uses current axis. Default is None.
    c : matplotlib color argument (optional)
        Gridline color. Default is grey.
    lw : number (optional)
        Gridline linewidth. Default is 1.
    diagonal : boolean (optional)
        Toggle inclusion of diagonal gridline. Default is False.
    zorder : number (optional)
        zorder of plotted grid. Default is 2.
    """
    # get ax
    if ax is None:
        ax = plt.gca()
    ax.grid()
    # get dashes
    ls = ':'
    dashes = (lw / 2, lw)
    # grid
    # ax.grid(True)
    lines = ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines()
    for line in lines.copy():
        line.set_linestyle(':')
        line.set_color(c)
        line.set_linewidth(lw)
        line.set_zorder(zorder)
        line.set_dashes(dashes)
        ax.add_line(line)
    # diagonal
    if diagonal:
        min_xi, max_xi = ax.get_xlim()
        min_yi, max_yi = ax.get_ylim()
        diag_min = max(min_xi, min_yi)
        diag_max = min(max_xi, max_yi)
        ax.plot([diag_min, diag_max], [diag_min, diag_max], c=c,
                ls=ls, lw=lw, zorder=zorder, dashes=dashes)

        # Plot resets xlim and ylim sometimes for unknown reasons.
        # This is here to ensure that the xlim and ylim are unchanged
        # after adding a diagonal, whose limits are calculated so
        # as to not change the xlim and ylim.
        #           -- KFS 2017-09-26
        ax.set_ylim(min_yi, max_yi)
        ax.set_xlim(min_xi, max_xi)


def plot_margins(fig=None, inches=1., centers=True, edges=True):
    """Add lines onto a figure indicating the margins, centers, and edges.

    Useful for ensuring your figure design scripts work as intended, and for laying
    out figures.

    Parameters
    ----------
    fig : matplotlib.figure.Figure object (optional)
        The figure to plot onto. If None, gets current figure. Default is None.
    inches : float (optional)
        The size of the figure margin, in inches. Default is 1.
    centers : bool (optional)
        Toggle for plotting lines indicating the figure center. Default is
        True.
    edges : bool (optional)
        Toggle for plotting lines indicating the figure edges. Default is True.
    """
    if fig is None:
        fig = plt.gcf()
    size = fig.get_size_inches()  # [H, V]
    trans_vert = inches / size[0]
    left = matplotlib.lines.Line2D([trans_vert, trans_vert], [0, 1],
                                   transform=fig.transFigure, figure=fig)
    right = matplotlib.lines.Line2D([1 - trans_vert, 1 - trans_vert],
                                    [0, 1], transform=fig.transFigure, figure=fig)
    trans_horz = inches / size[1]
    bottom = matplotlib.lines.Line2D(
        [0, 1], [trans_horz, trans_horz], transform=fig.transFigure, figure=fig)
    top = matplotlib.lines.Line2D([0, 1], [1 - trans_horz, 1 - trans_horz],
                                  transform=fig.transFigure, figure=fig)
    fig.lines.extend([left, right, bottom, top])
    if centers:
        vert = matplotlib.lines.Line2D(
            [0.5, 0.5], [0, 1], transform=fig.transFigure, figure=fig, c='r')
        horiz = matplotlib.lines.Line2D(
            [0, 1], [0.5, 0.5], transform=fig.transFigure, figure=fig, c='r')
        fig.lines.extend([vert, horiz])
    if edges:
        left = matplotlib.lines.Line2D(
            [0, 0], [0, 1], transform=fig.transFigure, figure=fig, c='k')
        right = matplotlib.lines.Line2D(
            [1, 1], [0, 1], transform=fig.transFigure, figure=fig, c='k')
        bottom = matplotlib.lines.Line2D(
            [0, 1], [0, 0], transform=fig.transFigure, figure=fig, c='k')
        top = matplotlib.lines.Line2D([0, 1], [1, 1], transform=fig.transFigure, figure=fig, c='k')
        fig.lines.extend([left, right, bottom, top])


def subplots_adjust(fig=None, inches=1):
    """Enforce margin to be equal around figure, starting at subplots.

    .. note::

        You probably should be using wt.artists.create_figure instead.

    See Also
    --------
    wt.artists.plot_margins
        Visualize margins, for debugging / layout.
    wt.artists.create_figure
        Convinience method for creating well-behaved figures.
    """
    if fig is None:
        fig = plt.gcf()
    size = fig.get_size_inches()
    vert = inches / size[0]
    horz = inches / size[1]
    fig.subplots_adjust(vert, horz, 1 - vert, 1 - horz)


def stitch_to_animation(images, outpath=None, duration=0.5, palettesize=256,
                        verbose=True):
    """Stitch a series of images into an animation.

    Currently supports animated gifs, other formats coming as needed.

    Parameters
    ----------
    images : list of strings
        Filepaths to the images to stitch together, in order of apperence.
    outpath : string (optional)
        Path of output, including extension. If None, bases output path on path
        of first path in `images`. Default is None.
    duration : number or list of numbers (optional)
        Duration of (each) frame in seconds. Default is 0.5.
    palettesize : int (optional)
        The number of colors in the resulting animation. Input is rounded to
        the nearest power of 2. Default is 1024.
    verbose : bool (optional)
        Toggle talkback. Default is True.
    """
    # parse filename
    if outpath is None:
        outpath = os.path.splitext(images[0])[0] + '.gif'
    # write
    try:
        t = wt_kit.Timer(verbose=False)
        with t, imageio.get_writer(outpath, mode='I', duration=duration,
                                   palettesize=palettesize) as writer:
            for p in images:
                image = imageio.imread(p)
                writer.append_data(image)
    except BaseException:
        print('Error: {0}'.format(sys.exc_info()[0]))
        return None
    # finish
    if verbose:
        interval = np.round(t.interval, 2)
        print('gif generated in {0} seconds - saved at {1}'.format(interval, outpath))
    return outpath
