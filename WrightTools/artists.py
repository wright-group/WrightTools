""" Tools for visualizing data.
"""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import datetime
import collections

import numpy as np
from numpy import r_

import matplotlib
from matplotlib.axes import Axes, SubplotBase, subplot_class_factory
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as grd
import matplotlib.colors as mplcolors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as PathEffects
from matplotlib.ticker import FormatStrFormatter

from . import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


# string types
if sys.version[0] == '2':
    string_type = basestring  # recognize unicode and string types
else:
    string_type = str  # newer versions of python don't have unicode type


# --- classes -------------------------------------------------------------------------------------


class Axes(matplotlib.axes.Axes):
    transposed = False
    is_sideplot = False

    def add_sideplot(self, along, pad=0, height=0.75, ymin=0, ymax=1.1):
        """ Add a side axis.

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

    def contourf(self, *args, **kwargs):
        # I'm overloading contourf in an attempt to fix aliasing problems when saving vector graphics
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
        matplotlib.axes.Axes.contour(self, *args[:3], len(contours.levels), cmap=contours.cmap,
                                     zorder=zorder)
        # PathCollection modifications
        for c in contours.collections:
            pass
            # c.set_rasterized(True)
            # c.set_edgecolor('face')
        return contours

    def legend(self, *args, **kwargs):
        if 'fancybox' not in kwargs.keys():
            kwargs['fancybox'] = False
        if 'framealpha' not in kwargs.keys():
            kwargs['framealpha'] = 1.
        return super().legend(*args, **kwargs)

    def plot_data(self, data, channel=0, interpolate=False, coloring=None,
                  xlabel=True, ylabel=True, zmin=None, zmax=None):
        """ Plot directly from a data object.

        Parameters
        ----------
        data : WrightTools.data.Data object
            The data object to plot.
        channel : int or str (optional)
            The channel to plot. Default is 0.
        interpolate : boolean (optional)
            Toggle interpolation. Default is False.
        cmap : str (optional)
            A key to the colormaps dictionary found in artists module. Default
            is None (inherits from channel).
        xlabel : boolean (optional)
            Toggle xlabel. Default is True.
        ylabel : boolean (optional)
            Toggle ylabel. Default is True.
        zmin : number (optional)
            Zmin. Default is None (inherited from channel).
        zmax : number (optional)
            Zmax. Default is None (inherited from channel).


        .. plot::

           >>> import matplotlib
           >>> from matplotlib import pyplot as plt
           >>> plt.plot(range(10))

        """
        # TODO: should I store a reference to data (or list of refs?)
        # prepare ---------------------------------------------------------------------------------
        # get dimensionality
        # get channel
        if isinstance(channel, int):
            channel_index = channel
        elif isinstance(channel, string_type):
            channel_index = data.channel_names.index(channel)
        else:
            print('channel type', type(channel), 'not valid')
        channel = data.channels[channel_index]
        # get axes
        xaxis = data.axes[0]
        # get zmin
        if zmin is None:
            zmin = channel.zmin
        # get zmax
        if zmax is None:
            zmax = channel.zmax
        # 1D --------------------------------------------------------------------------------------
        if data.dimensionality == 1:
            # get list of all datas
            # get color
            if coloring is None:
                c = self._get_lines.get_next_color()
            else:
                c = coloring
            # get arrays
            xi = xaxis.points
            yi = data.channels[channel_index].values
            # plot
            if interpolate:
                self.plot(xi, yi, c=c)
            else:
                self.scatter(xi, yi, c=c)
            # decoration
            if self.get_adjustable() == 'datalim':
                self.set_xlim(xi.min(), xi.max())
                self.set_ylim(zmin, zmax)
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
            xi = xaxis.points
            yi = yaxis.points
            zi = channel.values.T
            # plot
            if interpolate:
                # contourf
                levels = np.linspace(zmin, zmax, 256)
                self.contourf(xi, yi, zi, levels=levels, cmap=cmap)
            else:
                # pcolor
                X, Y, Z = pcolor_helper(xi, yi, zi)
                self.pcolor(X, Y, Z, vmin=zmin, vmax=zmax, cmap=cmap)
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

    def add_subplot(self, *args, **kwargs):
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

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)


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
    """ Add a sideplot to an axis. Sideplots share their corresponding axis.

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
    """ Quickly apply rcparams.

    Parameters
    ----------

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
    """ Place some text in the corner of the figure.

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
    """ Re-parameterization of matplotlib figure creation tools, exposing variables
    convinient for the Wright Group.

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
        `matplotlib documentation <http://matplotlib.org/1.4.0/users/gridspec.html#gridspec-and-subplotspec>`_
        for more information.

    Notes
    -----
    To ensure the margins work as expected, save the fig with
    the same margins (``pad_inches``) as specified in this function. Common
    savefig call:
    ``plt.savefig(plt.savefig(output_path, dpi=300, transparent=True,
    pad_inches=1))``

    See also
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
    """ Plot a diagonal line.

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
    """ Get a list of RGBA colors. Useful for plotting lots of elements, keeping
    the color of each unique.

    Parameters
    ----------
    n : integer
        The number of colors to return.
    cmap : string (optional)
        The colormap to use in the cycle. Default is rainbow.
    rotations : integer (optional)
        The number of times to repeat the colormap over the cycle. Default is
        3.

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
    string_list = [constant.get_label(show_units=True, points=True) for constant in constants]
    text = '    '.join(string_list)
    return text


def get_scaled_bounds(ax, position, distance=0.1, factor=200):
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
    """Return a grayscale version of the colormap
    Source: https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/
    """
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)


def make_cubehelix(gamma=0.5, s=0.25, r=-1, h=1.3, reverse=False, darkest=0.7):
    """ Define cubehelix type colorbars. For more information see http://arxiv.org/abs/1108.5083.

    Parameters
    ----------
    gamma : number (optional)
        Intensity factor. Default is 0.5
    s : number (optional)
        Start color factor. Default is 0.25
    r : number (optional)
        Number and direction of rotations. Default is -1
    h : number (option)
        Hue factor. Default is 1.3
    reverse : boolean (optional)
        Toggle reversal of output colormap. By default (Reverse = False),
        colormap goes from light to dark.
    darkest : number (optional)
        Default is 0.7

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap

    See Also
    --------
    plot_colormap_components
        Displays RGB components of colormaps.
    """
    rr = (.213 / .30)
    rg = (.715 / .99)
    rb = (.072 / .11)

    def get_color_function(p0, p1):
        def color(x):
            # Calculate amplitude and angle of deviation from the black to
            # white diagonal in the plane of constant perceived intensity.
            xg = darkest * x**gamma
            lum = 1 - xg  # starts at 1
            if reverse:
                lum = lum[::-1]
            a = lum.copy()
            a[lum < 0.5] = h * lum[lum < 0.5] / 2.
            a[lum >= 0.5] = h * (1 - lum[lum >= 0.5]) / 2.
            phi = 2 * np.pi * (s / 3 + r * x)
            out = lum + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
            return out
        return color
    rgb_dict = {'red': get_color_function(-0.14861 * rr, 1.78277 * rr),
                'green': get_color_function(-0.29227 * rg, -0.90649 * rg),
                'blue': get_color_function(1.97294 * rb, 0.0)}
    cmap = matplotlib.colors.LinearSegmentedColormap('cubehelix', rgb_dict)
    return cmap


def make_colormap(seq, name='CustomMap', plot=False):
    """ Return a LinearSegmentedColormap

    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    from http://nbviewer.ipython.org/gist/anonymous/a4fa0adb08f9e9ea4f94
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    cmap = mplcolors.LinearSegmentedColormap(name, cdict)
    if plot:
        plot_colormap(cmap)
    return cmap


def nm_to_rgb(nm):
    """ returns list [r, g, b] (zero to one scale) for given input in nm

    original code - http://www.physics.sfasu.edu/astro/color/spectra.html
    """
    w = int(nm)
    # color ---------------------------------------------------------------------------------------
    if w >= 380 and w < 440:
        R = -(w - 440.) / (440. - 350.)
        G = 0.0
        B = 1.0
    elif w >= 440 and w < 490:
        R = 0.0
        G = (w - 440.) / (490. - 440.)
        B = 1.0
    elif w >= 490 and w < 510:
        R = 0.0
        G = 1.0
        B = -(w - 510.) / (510. - 490.)
    elif w >= 510 and w < 580:
        R = (w - 510.) / (580. - 510.)
        G = 1.0
        B = 0.0
    elif w >= 580 and w < 645:
        R = 1.0
        G = -(w - 645.) / (645. - 580.)
        B = 0.0
    elif w >= 645 and w <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    # intensity correction ------------------------------------------------------------------------
    if w >= 380 and w < 420:
        SSS = 0.3 + 0.7 * (w - 350) / (420 - 350)
    elif w >= 420 and w <= 700:
        SSS = 1.0
    elif w > 700 and w <= 780:
        SSS = 0.3 + 0.7 * (780 - w) / (780 - 700)
    else:
        SSS = 0.0
    SSS *= 255
    return [float(int(SSS * R) / 256.),
            float(int(SSS * G) / 256.),
            float(int(SSS * B) / 256.)]


def pcolor_helper(xi, yi, zi, transform=None):
    """

    accepts xi, yi, zi as the normal rectangular arrays
    that would be given to contorf etc

    returns list [X, Y, Z] appropriate for feeding directly
    into matplotlib.pyplot.pcolor so that the pixels are centered correctly.

    transform takes a function that accepts a
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
                  label=None, tick_fontsize=14, label_fontsize=18, decimals=3,
                  orientation='vertical', ticklocation='auto'):
    """ Easily add a colormap to an axis.

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
        Number of decimals to appear in tick labels. Default is 3.
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
    # make cbar
    format = '%.{0}f'.format(decimals)
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


def plot_colormap_components(cmap):
    """ Plot the components of a given colormap.  """

    plt.figure(figsize=[8, 4])
    gs = grd.GridSpec(2, 1, height_ratios=[1, 10], hspace=0.05)
    # colorbar
    ax = plt.subplot(gs[0])
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=cmap, vmin=0., vmax=1.)
    ax.set_axis_off()
    # components
    ax = plt.subplot(gs[1])
    x = gradient[0]
    r = cmap._segmentdata['red'](x)
    g = cmap._segmentdata['green'](x)
    b = cmap._segmentdata['blue'](x)
    k = .3 * r + .59 * g + .11 * b
    # truncate
    r.clip(0, 1, out=r)
    g.clip(0, 1, out=g)
    b.clip(0, 1, out=b)
    # plot
    plt.plot(x, r, 'r', linewidth=5, alpha=0.6)
    plt.plot(x, g, 'g', linewidth=5, alpha=0.6)
    plt.plot(x, b, 'b', linewidth=5, alpha=0.6)
    plt.plot(x, k, 'k:', linewidth=5, alpha=0.6)
    ax.set_ylim(-.1, 1.1)
    # finish
    plt.grid()
    plt.xlabel('value', fontsize=17)
    plt.ylabel('intensity', fontsize=17)


def savefig(path, fig=None, close=True, dpi=300):
    """ Save a figure.

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
    """ Set all axis labels properties easily.

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

    See also
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
        if xticks is not False:
            ax.set_xticks(xticks)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(axis='x', which='both', length=0)
    # y
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if yticks is not None:
        if yticks is not False:
            ax.set_yticks(yticks)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='y', which='both', length=0)


def set_ax_spines(ax=None, c='k', lw=3, zorder=10):
    """ Easily the properties of all four axis spines.

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
    """ Set all axis labels of a figure simultaniously.

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

    See also
    --------
    set_ax_labels
    """
    # get fig
    if fig is None:
        fig = plt.gcf()
    # axes
    for ax in fig.axes:
        if ax.is_first_col() and ax.is_last_row():
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
    """ Plot dotted gridlines onto an axis.

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
    for l in lines.copy():
        l = l
        l.set_linestyle(':')
        l.set_color(c)
        l.set_linewidth(lw)
        l.set_zorder(zorder)
        l.set_dashes(dashes)
        ax.add_line(l)
    # diagonal
    if diagonal:
        min_xi, max_xi = ax.get_xlim()
        min_yi, max_yi = ax.get_ylim()
        diag_min = max(min_xi, min_yi)
        diag_max = min(max_xi, max_yi)
        ax.plot([diag_min, diag_max], [diag_min, diag_max], c=c,
                ls=':', lw=lw, zorder=zorder, dashes=dashes)


def plot_margins(fig=None, inches=1., centers=True, edges=True):
    """ Add lines onto a figure indicating the margins, centers, and edges.

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
    """ Enforce margin to be equal around figure, starting at subplots.

    .. note::

        You probably should be using wt.artists.create_figure instead.

    See also
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
    """ Stitch a series of images into an animation.

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
    # import imageio
    try:
        import imageio
    except ImportError:
        raise ImportError(
            'WrightTools.artists.stitch_to_animation requires imageio - https://imageio.github.io/')
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


# --- color maps ----------------------------------------------------------------------------------


cubehelix = make_cubehelix()

experimental = ['#FFFFFF',
                '#0000FF',
                '#0080FF',
                '#00FFFF',
                '#00FF00',
                '#FFFF00',
                '#FF8000',
                '#FF0000',
                '#881111']

greenscale = ['#000000',  # black
              '#00FF00']  # green

greyscale = ['#FFFFFF',  # white
             '#000000']  # black

invisible = ['#FFFFFF',  # white
             '#FFFFFF']  # white

# isoluminant colorbar based on the research of Kindlmann et al.
# http://dx.doi.org/10.1109/VISUAL.2002.1183788
c = mplcolors.ColorConverter().to_rgb
isoluminant1 = make_colormap([
    c(r_[1.000, 1.000, 1.000]), c(r_[0.847, 0.057, 0.057]), 1 / 6.,
    c(r_[0.847, 0.057, 0.057]), c(r_[0.527, 0.527, 0.000]), 2 / 6.,
    c(r_[0.527, 0.527, 0.000]), c(r_[0.000, 0.592, 0.000]), 3 / 6.,
    c(r_[0.000, 0.592, 0.000]), c(r_[0.000, 0.559, 0.559]), 4 / 6.,
    c(r_[0.000, 0.559, 0.559]), c(r_[0.316, 0.316, 0.991]), 5 / 6.,
    c(r_[0.316, 0.316, 0.991]), c(r_[0.718, 0.000, 0.718])],
    name='isoluminant`')

isoluminant2 = make_colormap([
    c(r_[1.000, 1.000, 1.000]), c(r_[0.718, 0.000, 0.718]), 1 / 6.,
    c(r_[0.718, 0.000, 0.718]), c(r_[0.316, 0.316, 0.991]), 2 / 6.,
    c(r_[0.316, 0.316, 0.991]), c(r_[0.000, 0.559, 0.559]), 3 / 6.,
    c(r_[0.000, 0.559, 0.559]), c(r_[0.000, 0.592, 0.000]), 4 / 6.,
    c(r_[0.000, 0.592, 0.000]), c(r_[0.527, 0.527, 0.000]), 5 / 6.,
    c(r_[0.527, 0.527, 0.000]), c(r_[0.847, 0.057, 0.057])],
    name='isoluminant2')

isoluminant3 = make_colormap([
    c(r_[1.000, 1.000, 1.000]), c(r_[0.316, 0.316, 0.991]), 1 / 5.,
    c(r_[0.316, 0.316, 0.991]), c(r_[0.000, 0.559, 0.559]), 2 / 5.,
    c(r_[0.000, 0.559, 0.559]), c(r_[0.000, 0.592, 0.000]), 3 / 5.,
    c(r_[0.000, 0.592, 0.000]), c(r_[0.527, 0.527, 0.000]), 4 / 5.,
    c(r_[0.527, 0.527, 0.000]), c(r_[0.847, 0.057, 0.057])],
    name='isoluminant3')

signed = ['#0000FF',  # blue
          '#002AFF',
          '#0055FF',
          '#007FFF',
          '#00AAFF',
          '#00D4FF',
          '#00FFFF',
          '#FFFFFF',  # white
          '#FFFF00',
          '#FFD400',
          '#FFAA00',
          '#FF7F00',
          '#FF5500',
          '#FF2A00',
          '#FF0000']  # red

signed_old = ['#0000FF',  # blue
              '#00BBFF',  # blue-aqua
              '#00FFFF',  # aqua
              '#FFFFFF',  # white
              '#FFFF00',  # yellow
              '#FFBB00',  # orange
              '#FF0000']  # red

skyebar = ['#FFFFFF',  # white
           '#000000',  # black
           '#0000FF',  # blue
           '#00FFFF',  # cyan
           '#64FF00',  # light green
           '#FFFF00',  # yellow
           '#FF8000',  # orange
           '#FF0000',  # red
           '#800000']  # dark red

skyebar_d = ['#000000',  # black
             '#0000FF',  # blue
             '#00FFFF',  # cyan
             '#64FF00',  # light green
             '#FFFF00',  # yellow
             '#FF8000',  # orange
             '#FF0000',  # red
             '#800000']  # dark red

skyebar_i = ['#000000',  # black
             '#FFFFFF',  # white
             '#0000FF',  # blue
             '#00FFFF',  # cyan
             '#64FF00',  # light green
             '#FFFF00',  # yellow
             '#FF8000',  # orange
             '#FF0000',  # red
             '#800000']  # dark red

wright = ['#FFFFFF',
          '#0000FF',
          '#00FFFF',
          '#00FF00',
          '#FFFF00',
          '#FF0000',
          '#881111']

colormaps = collections.OrderedDict()
colormaps['coolwarm'] = plt.get_cmap('coolwarm')
colormaps['cubehelix'] = plt.get_cmap('cubehelix_r')
colormaps['default'] = cubehelix
colormaps['flag'] = plt.get_cmap('flag')
colormaps['greenscale'] = mplcolors.LinearSegmentedColormap.from_list('greenscale', greenscale)
colormaps['greyscale'] = mplcolors.LinearSegmentedColormap.from_list('greyscale', greyscale)
colormaps['invisible'] = mplcolors.LinearSegmentedColormap.from_list('invisible', invisible)
colormaps['isoluminant1'] = isoluminant1
colormaps['isoluminant2'] = isoluminant2
colormaps['isoluminant3'] = isoluminant3
colormaps['prism'] = plt.get_cmap('prism')
colormaps['rainbow'] = plt.get_cmap('rainbow')
colormaps['seismic'] = plt.get_cmap('seismic')
colormaps['signed'] = plt.get_cmap('bwr')
colormaps['signed_old'] = mplcolors.LinearSegmentedColormap.from_list('signed', signed_old)
colormaps['skyebar1'] = mplcolors.LinearSegmentedColormap.from_list('skyebar', skyebar)
colormaps['skyebar2'] = mplcolors.LinearSegmentedColormap.from_list('skyebar dark', skyebar_d)
colormaps['skyebar3'] = mplcolors.LinearSegmentedColormap.from_list('skyebar inverted', skyebar_i)
colormaps['wright'] = mplcolors.LinearSegmentedColormap.from_list('wright', wright)


# enforce grey as 'bad' value for colormaps
for cmap in colormaps.values():
    cmap.set_bad([0.75] * 3, 1)


# a nice set of line colors
overline_colors = ['#CCFF00', '#FE4EDA', '#FF6600', '#00FFBF', '#00B7EB']


# --- general purpose artists ---------------------------------------------------------------------


class mpl_1D:

    def __init__(self, data, xaxis=0, at={}, verbose=True):
        # import data
        self.data = data
        self.chopped = self.data.chop(xaxis, at=at, verbose=False)
        if verbose:
            print('mpl_1D recieved data to make %d plots' % len(self.chopped))
        # defaults
        self.font_size = 15

    def plot(self, channel=0, local=False, autosave=False, output_folder=None,
             fname=None, lines=True, verbose=True):
        # get channel index
        if type(channel) in [int, float]:
            channel_index = int(channel)
        elif isinstance(channel, string_type):
            channel_index = self.chopped[0].channel_names.index(channel)
        else:
            print('channel type not recognized in mpl_1D!')
        # prepare figure
        fig = None
        if len(self.chopped) > 10:
            if not autosave:
                print('too many images will be generated ({}): forcing autosave'.format(len(self.chopped)))
                autosave = True
        # prepare output folders
        if autosave:
            if output_folder:
                pass
            else:
                if len(self.chopped) == 1:
                    output_folder = os.getcwd()
                    if fname:
                        pass
                    else:
                        fname = self.data.name
                else:
                    folder_name = 'mpl_1D ' + wt_kit.TimeStamp().path
                    os.mkdir(folder_name)
                    output_folder = folder_name
        # chew through image generation
        outfiles = [''] * len(self.chopped)
        for i in range(len(self.chopped)):
            if fig and autosave:
                plt.close(fig)
            aspects = [[[0, 0], 0.5]]
            fig, gs = create_figure(width='single', nrows=1, cols=[1], aspects=aspects)
            current_chop = self.chopped[i]
            axes = current_chop.axes
            channels = current_chop.channels
            constants = current_chop.constants
            axis = axes[0]
            xi = axes[0].points
            zi = channels[channel_index].values
            plt.plot(xi, zi, lw=2)
            plt.scatter(xi, zi, color='grey', alpha=0.5, edgecolor='none')
            plt.grid()
            # variable marker lines
            if lines:
                for constant in constants:
                    if constant.units_kind == 'energy':
                        if axis.units == constant.units:
                            plt.axvline(constant.points, color='k', linewidth=4, alpha=0.25)
            # limits
            if local:
                pass
            else:
                plt.ylim(channels[channel_index].zmin, channels[channel_index].zmax)
            # label axes
            plt.xlabel(axes[0].get_label(), fontsize=18)
            plt.ylabel(channels[channel_index].name, fontsize=18)
            plt.xticks(rotation=45)
            plt.xlim(xi.min(), xi.max())
            # title
            title = self.data.name
            constant_text = get_constant_text(constants)
            if not constant_text == '':
                title += '\n' + constant_text
            plt.suptitle(title, fontsize=20)
            # save
            if autosave:
                if fname:
                    file_name = fname + ' ' + str(i).zfill(3)
                else:
                    file_name = str(i).zfill(3)
                fpath = os.path.join(output_folder, file_name + '.png')
                plt.savefig(fpath, transparent=True, dpi=300, pad_inches=1.)
                plt.close()
                if verbose:
                    print('image saved at', fpath)
                outfiles[i] = fpath
        return outfiles


class mpl_2D:

    def __init__(self, data, xaxis=1, yaxis=0, at={}, verbose=True):
        # import data
        self.data = data
        self.chopped = self.data.chop(yaxis, xaxis, at=at, verbose=False)
        if verbose:
            print('mpl_2D recieved data to make %d plots' % len(self.chopped))
        # defaults
        self._xsideplot = False
        self._ysideplot = False
        self._xsideplotdata = []
        self._ysideplotdata = []
        self._onplotdata = []

    def get_lims(self, transform=None):
        """ Find plot limits using transform.

        Assumes that the corners of the axes are also the most extreme points
        of the transformed axes.

        Parameters
        ----------
        axis1 : axis
            The x[0] axis
        axis2 : axis
            The x[1] axis
        transform: callable (optoinal)
            The transform function, accepts a tuple, ouptus transformed tuple

        Returns
        ----------
        xlim : tuple of floats
            (min_x, max_x)
        ylim : tuple of floats
            (min_y, max_y)
        """
        if not isinstance(transform, type(None)):
            x_corners = []
            y_corners = []
            for idx1 in [-1, 1]:
                for idx2 in [-1, 1]:
                    x, y = transform((self.xaxis.points[idx1], self.yaxis.points[idx2]))
                    x_corners.append(x)
                    y_corners.append(y)
            xlim = (min(x_corners), max(x_corners))
            ylim = (min(y_corners), max(y_corners))
        else:
            xlim = (self.xaxis.points.min(), self.xaxis.points.max())
            ylim = (self.yaxis.points.min(), self.yaxis.points.max())
        return xlim, ylim

    def sideplot(self, data, x=True, y=True):
        data = data.copy()
        if x:
            if self.chopped[0].axes[1].units_kind == data.axes[0].units_kind:
                data.convert(self.chopped[0].axes[1].units)
                self._xsideplot = True
                self._xsideplotdata.append([data.axes[0].points, data.channels[0].values])
            else:
                print('given data ({0}), does not aggree with x ({1})'.format(
                    data.axes[0].units_kind, self.chopped[0].axes[1].units_kind))
        if y:
            if self.chopped[0].axes[0].units_kind == data.axes[0].units_kind:
                data.convert(self.chopped[0].axes[0].units)
                self._ysideplot = True
                self._ysideplotdata.append([data.axes[0].points, data.channels[0].values])
            else:
                print('given data ({0}), does not aggree with y ({1})'.format(
                    data.axes[0].units_kind, self.chopped[0].axes[0].units_kind))

    def onplot(self, xi, yi, c='k', lw=5, alpha=0.3, **kwargs):
        kwargs['c'] = c
        kwargs['lw'] = lw
        kwargs['alpha'] = alpha
        self._onplotdata.append((xi, yi, kwargs))

    def plot(self, channel=0,
             contours=0, pixelated=True, lines=True, cmap='automatic',
             facecolor='w', dynamic_range=False, local=False,
             contours_local=True, normalize_slices='both', xbin=False,
             ybin=False, xlim=None, ylim=None, autosave=False,
             output_folder=None, fname=None, verbose=True,
             transform=None, contour_thickness=None):
        """ Draw the plot(s).

        Parameters
        ----------
        channel : int or string (optional)
            The index or name of the channel to plot. Default is 0.
        contours : int (optional)
            The number of black contour lines to add to the plot. You may set
            contours to 0 to use no contours in your plot. Default is 9.
        pixelated : bool (optional)
            Toggle between pclolormesh and contourf (deulaney) as plotting
            method. Default is True.
        lines : bool (optional)
            Toggle attempt to plot lines showing value of 'corresponding'
            color dimensions. Default is True.
        cmap : str (optional)
            A key to the colormaps dictionary found in artists module. Default
            is 'default'.
        facecolor : str (optional)
            Facecolor. Default is 'w'.
        dyanmic_range : bool (optional)
            Force the colorbar to use all of its colors. Only changes behavior
            for signed data. Default is False.
        local : bool (optional)
            Toggle plotting locally. Default is False.
        contours_local : bool (optional)
            Toggle plotting contours locally. Default is True.
        normalize_slices : {'both', 'horizontal', 'vertical'} (optional)
            Normalization strategy. Default is both.
        xbin : bool (optional)
            Plot xbin. Default is False.
        ybin : bool (optional)
            Plot ybin. Default is False.
        xlim : float (optional)
            Control limit of plot in x. Default is None (data limit).
        ylim : float (optional)
            Control limit of plot in y. Default is None (data limit).
        autosave : bool (optional)
            Autosave.
        output_folder : str (optional)
            Output folder.
        fname : str (optional)
            File name.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        """
        # get channel index
        if type(channel) in [int, float]:
            channel_index = int(channel)
        elif isinstance(channel, string_type):
            channel_index = self.chopped[0].channel_names.index(channel)
        else:
            print('channel type not recognized in mpl_2D!')
        # prepare figure
        fig = None
        if len(self.chopped) > 10:
            if not autosave:
                print('too many images will be generated: forcing autosave')
                autosave = True
        # prepare output folder
        if autosave:
            if output_folder:
                pass
            else:
                if len(self.chopped) == 1:
                    output_folder = os.getcwd()
                    if fname:
                        pass
                    else:
                        fname = self.data.name
                else:
                    folder_name = 'mpl_2D ' + wt_kit.get_timestamp(style='short')
                    os.mkdir(folder_name)
                    output_folder = folder_name
        # chew through image generation
        outfiles = [''] * len(self.chopped)
        for i in range(len(self.chopped)):
            # get data to plot --------------------------------------------------------------------
            current_chop = self.chopped[i]
            axes = current_chop.axes
            channels = current_chop.channels
            constants = current_chop.constants
            self.xaxis = axes[1]
            self.yaxis = axes[0]
            channel = channels[channel_index]
            zi = channel.values
            zi = np.ma.masked_invalid(zi)
            # normalize slices --------------------------------------------------------------------
            if normalize_slices == 'both':
                pass
            elif normalize_slices == 'horizontal':
                nmin = channel.znull
                # normalize all x traces to a common value
                maxes = zi.max(axis=1)
                numerator = (zi - nmin)
                denominator = (maxes - nmin)
                for j in range(zi.shape[0]):
                    zi[j] = numerator[j] / denominator[j]
                channel.zmax = zi.max()
                channel.zmin = zi.min()
                channel.znull = 0
            elif normalize_slices == 'vertical':
                nmin = channel.znull
                maxes = zi.max(axis=0)
                numerator = (zi - nmin)
                denominator = (maxes - nmin)
                for j in range(zi.shape[1]):
                    zi[:, j] = numerator[:, j] / denominator[j]
                channel.zmax = zi.max()
                channel.zmin = zi.min()
                channel.znull = 0
            # create figure -----------------------------------------------------------------------
            if fig and autosave:
                plt.close(fig)

            find_xlim = isinstance(xlim, type(None))
            find_ylim = isinstance(ylim, type(None))
            if find_ylim or find_xlim:
                if find_ylim and find_xlim:
                    xlim, ylim = self.get_lims(transform)
                elif find_ylim:
                    toss, ylim = self.get_lims(transform)
                else:
                    xlim, toss = self.get_lims(transform)

            if self.xaxis.units == self.yaxis.units:
                xr = xlim[1] - xlim[0]
                yr = ylim[1] - ylim[0]
                aspect = np.abs(yr / xr)
                if 3 < aspect or aspect < 1 / 3.:
                    # TODO: raise warning here
                    aspect = np.clip(aspect, 1 / 3., 3.)
            else:
                aspect = 1
            fig, gs = create_figure(width='single', nrows=1, cols=[
                                    1, 'cbar'], aspects=[[[0, 0], aspect]])
            subplot_main = plt.subplot(gs[0])
            subplot_main.patch.set_facecolor(facecolor)
            # levels ------------------------------------------------------------------------------
            if channel.signed:
                if local:
                    print('signed local')
                    limit = max(abs(channel.znull - np.nanmin(zi)),
                                abs(channel.znull - np.nanmax(zi)))
                else:
                    if dynamic_range:
                        limit = min(abs(channel.znull - channel.zmin),
                                    abs(channel.znull - channel.zmax))
                    else:
                        limit = channel.zmag
                if np.isnan(limit):
                    limit = 1.
                if limit is np.ma.masked:
                    limit = 1.
                levels = np.linspace(-limit + channel.znull, limit + channel.znull, 200)
            else:
                if local:
                    levels = np.linspace(channel.znull, np.nanmax(zi), 200)
                else:
                    if channel.zmax < channel.znull:
                        levels = np.linspace(channel.zmin, channel.znull, 200)
                    else:
                        levels = np.linspace(channel.znull, channel.zmax, 200)
            # main plot ---------------------------------------------------------------------------
            # get colormap
            if cmap == 'automatic':
                if channel.signed:
                    cmap = 'signed'
                else:
                    cmap = 'default'
            mycm = colormaps[cmap]
            mycm.set_bad([0.75, 0.75, 0.75], 1.)
            mycm.set_under(facecolor)
            # fill in main data environment
            # always plot pcolormesh
            if pixelated:
                X, Y, Z = pcolor_helper(self.xaxis.points, self.yaxis.points,
                                        zi, transform=transform)
                cax = plt.pcolormesh(X, Y, Z, cmap=mycm,
                                     vmin=levels.min(), vmax=levels.max())
            plt.xlim(self.xaxis.points.min(), self.xaxis.points.max())
            plt.ylim(self.yaxis.points.min(), self.yaxis.points.max())
            # overlap with contourf if not pixelated
            if not pixelated:
                X, Y = np.meshgrid(self.xaxis.points, self.yaxis.points)
                if not isinstance(transform, type(None)):
                    for (x, y), value in np.ndenumerate(X):
                        X[x][y], Y[x][y] = transform((X[x][y], Y[x][y]))
                # if not type(transform)==type(None):
                #    X,Y,Z = self.sort_points(X,Y,zi)
                cax = subplot_main.contourf(X, Y, zi,
                                            levels, cmap=mycm)

            plt.xticks(rotation=45, fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel(self.xaxis.get_label(), fontsize=18)
            plt.ylabel(self.yaxis.get_label(), fontsize=17)
            # delay space deliniation lines -------------------------------------------------------
            if lines:
                if self.xaxis.units_kind == 'delay':
                    plt.axvline(0, lw=2, c='k')
                if self.yaxis.units_kind == 'delay':
                    plt.axhline(0, lw=2, c='k')
                if self.xaxis.units_kind == 'delay' and self.xaxis.units == self.yaxis.units:
                    diagonal_line(self.xaxis.points, self.yaxis.points, c='k', lw=2, ls='-')
            # variable marker lines ---------------------------------------------------------------
            if lines:
                for constant in constants:
                    if constant.units_kind == 'energy':
                        # x axis
                        if self.xaxis.units == constant.units:
                            plt.axvline(constant.points, color='k', linewidth=4, alpha=0.25)
                        # y axis
                        if self.yaxis.units == constant.units:
                            plt.axhline(constant.points, color='k', linewidth=4, alpha=0.25)
            # grid --------------------------------------------------------------------------------
            plt.grid(b=True)
            if self.xaxis.units == self.yaxis.units:
                # add diagonal line
                if xlim:
                    x = xlim
                else:
                    x = self.xaxis.points
                if ylim:
                    y = ylim
                else:
                    y = self.yaxis.points
                diag_min = max(min(x), min(y))
                diag_max = min(max(x), max(y))
                plt.plot([diag_min, diag_max], [diag_min, diag_max], 'k:')
            # contour lines -----------------------------------------------------------------------
            if contours:
                if contours_local:
                    # force top and bottom contour to be just outside of data range
                    # add two contours
                    contours_levels = np.linspace(
                        channel.znull - 1e-10, np.nanmax(zi) + 1e-10, contours + 2)
                else:
                    contours_levels = contours
                if contour_thickness is None:
                    subplot_main.contour(X, Y, zi, contours_levels, colors='k')
                else:
                    subplot_main.contour(X, Y, zi, contours_levels, colors='k',
                                         linewidths=contour_thickness)
            # finish main subplot -----------------------------------------------------------------
            if xlim:
                subplot_main.set_xlim(xlim[0], xlim[1])
            else:
                subplot_main.set_xlim(self.xaxis.points[0], self.xaxis.points[-1])
            if ylim:
                subplot_main.set_ylim(ylim[0], ylim[1])
            else:
                subplot_main.set_ylim(self.yaxis.points[0], self.yaxis.points[-1])
            # sideplots ---------------------------------------------------------------------------
            divider = make_axes_locatable(subplot_main)
            if xbin or self._xsideplot:
                axCorrx = divider.append_axes('top', 0.75, pad=0.0, sharex=subplot_main)
                axCorrx.autoscale(False)
                axCorrx.set_adjustable('box-forced')
                plt.setp(axCorrx.get_xticklabels(), visible=False)
                plt.setp(axCorrx.get_yticklabels(), visible=False)
                plt.grid(b=True)
                if channel.signed:
                    axCorrx.set_ylim([-1.1, 1.1])
                else:
                    axCorrx.set_ylim([0, 1.1])
                # bin
                if xbin:
                    x_ax_int = np.nansum(zi, axis=0) - channel.znull * len(self.yaxis.points)
                    x_ax_int[x_ax_int == 0] = np.nan
                    # normalize (min is a pixel)
                    xmax = max(np.abs(x_ax_int))
                    x_ax_int = x_ax_int / xmax
                    axCorrx.plot(self.xaxis.points, x_ax_int, lw=2)
                    axCorrx.set_xlim([self.xaxis.points.min(), self.xaxis.points.max()])
                # data
                if self._xsideplot:
                    for s_xi, s_zi in self._xsideplotdata:
                        xlim = axCorrx.get_xlim()
                        min_index = np.argmin(abs(s_xi - min(xlim)))
                        max_index = np.argmin(abs(s_xi - max(xlim)))
                        s_zi_in_range = s_zi[min(min_index, max_index):max(min_index, max_index)]
                        if len(s_zi_in_range) == 0:
                            continue
                        #s_zi = s_zi - min(s_zi_in_range)
                        s_zi_in_range = s_zi[min(min_index, max_index):max(min_index, max_index)]
                        s_zi = s_zi / max(s_zi_in_range)
                        axCorrx.plot(s_xi, s_zi, lw=2)
                # line
                if lines:
                    for constant in constants:
                        if constant.units_kind == 'energy':
                            if self.xaxis.units == constant.units:
                                axCorrx.axvline(constant.points, color='k',
                                                linewidth=4, alpha=0.25)
            if ybin or self._ysideplot:
                axCorry = divider.append_axes('right', 0.75, pad=0.0, sharey=subplot_main)
                axCorry.autoscale(False)
                axCorry.set_adjustable('box-forced')
                plt.setp(axCorry.get_xticklabels(), visible=False)
                plt.setp(axCorry.get_yticklabels(), visible=False)
                plt.grid(b=True)
                if channel.signed:
                    axCorry.set_xlim([-1.1, 1.1])
                else:
                    axCorry.set_xlim([0, 1.1])
                # bin
                if ybin:
                    y_ax_int = np.nansum(zi, axis=1) - channel.znull * len(self.xaxis.points)
                    y_ax_int[y_ax_int == 0] = np.nan
                    # normalize (min is a pixel)
                    ymax = max(np.abs(y_ax_int))
                    y_ax_int = y_ax_int / ymax
                    axCorry.plot(y_ax_int, self.yaxis.points, lw=2)
                    axCorry.set_ylim([self.yaxis.points.min(), self.yaxis.points.max()])
                # data
                if self._ysideplot:
                    for s_xi, s_zi in self._ysideplotdata:
                        xlim = axCorry.get_ylim()
                        min_index = np.argmin(abs(s_xi - min(xlim)))
                        max_index = np.argmin(abs(s_xi - max(xlim)))
                        s_zi_in_range = s_zi[min(min_index, max_index):max(min_index, max_index)]
                        if len(s_zi_in_range) == 0:
                            continue
                        #s_zi = s_zi - min(s_zi_in_range)
                        s_zi_in_range = s_zi[min(min_index, max_index):max(min_index, max_index)]
                        s_zi = s_zi / max(s_zi_in_range)
                        axCorry.plot(s_zi, s_xi, lw=2)
                # line
                if lines:
                    for constant in constants:
                        if constant.units_kind == 'energy':
                            if self.yaxis.units == constant.units:
                                axCorry.axvline(constant.points, color='k',
                                                linewidth=4, alpha=0.25)
            # onplot ------------------------------------------------------------------------------
            for xi, yi, kwargs in self._onplotdata:
                subplot_main.plot(xi, yi, **kwargs)
            # colorbar ----------------------------------------------------------------------------
            subplot_cb = plt.subplot(gs[1])
            cbar_ticks = np.linspace(levels.min(), levels.max(), 11)
            if cbar_ticks.max() == 1.0:
                cbar = plt.colorbar(cax, cax=subplot_cb, cmap=mycm,
                                    ticks=cbar_ticks, format='%.1f')
            else:
                cbar = plt.colorbar(cax, cax=subplot_cb, cmap=mycm,
                                    ticks=cbar_ticks, format='%.3f')
            cbar.set_label(channel.name, fontsize=18)
            cbar.ax.tick_params(labelsize=14)
            # title -------------------------------------------------------------------------------
            title_text = self.data.name
            constants_text = get_constant_text(constants)
            _title(fig, title_text, constants_text)
            # save figure -------------------------------------------------------------------------
            if autosave:
                if fname.endswith('.pdf'):
                    file_name = fname.split('.')[0] + ' ' + str(i).zfill(3) + '.pdf'
                    fpath = os.path.join(output_folder, file_name)
                    plt.savefig(fpath)
                else:
                    if fname:
                        file_name = fname + ' ' + str(i).zfill(3)
                    else:
                        file_name = str(i).zfill(3)
                    fpath = os.path.join(output_folder, file_name + '.png')
                    plt.savefig(fpath, facecolor='none', transparent=True, dpi=300, pad_inches=1.)
                plt.close()
                if verbose:
                    print('image saved at', fpath)
                outfiles[i] = fpath
        return outfiles


# --- specific artists ----------------------------------------------------------------------------


class Absorbance:

    def __init__(self, data):

        if not isinstance(data, list):
            data = [data]

        self.data = data

    def plot(self, channel_index=0, xlim=None, ylim=None,
             yticks=True, derivative=True, n_smooth=10,):

        # prepare plot environment ----------------------------------------------------------------

        self.font_size = 14

        if derivative:
            aspects = [[[0, 0], 0.35], [[1, 0], 0.35]]
            hspace = 0.1
            fig, gs = create_figure(width='single', cols=[
                                    1], hspace=hspace, nrows=2, aspects=aspects)
            self.ax1 = plt.subplot(gs[0])
            plt.ylabel('OD', fontsize=18)
            plt.grid()
            plt.setp(self.ax1.get_xticklabels(), visible=False)
            self.ax2 = plt.subplot(gs[1], sharex=self.ax1)
            plt.grid()
            plt.ylabel('2nd der.', fontsize=18)
        else:
            aspects = [[[0, 0], 0.35]]
            fig, gs = create_figure(width='single', cols=[1], aspects=aspects)
            self.ax1 = plt.subplot(111)
            plt.ylabel('OD', fontsize=18)
            plt.grid()

        plt.xticks(rotation=45)

        for data in self.data:

            # import data -------------------------------------------------------------------------

            xi = data.axes[0].points
            zi = data.channels[channel_index].values

            # scale -------------------------------------------------------------------------------

            if xlim:
                plt.xlim(xlim[0], xlim[1])
                min_index = np.argmin(abs(xi - min(xlim)))
                max_index = np.argmin(abs(xi - max(xlim)))
                zi_truncated = zi[min(min_index, max_index):max(min_index, max_index)]
                zi -= zi_truncated.min()
                zi_truncated = zi[min(min_index, max_index):max(min_index, max_index)]
                zi /= zi_truncated.max()
            else:
                xlim = xi.min(), xi.max()

            # plot absorbance ---------------------------------------------------------------------

            self.ax1.plot(xi, zi, lw=2)
            self.ax1.set_xlim(*xlim)

            # now plot 2nd derivative -------------------------------------------------------------

            if derivative:
                print('hello')
                # compute second derivative
                xi2, zi2 = self._smooth(np.array([xi, zi]), n_smooth)
                diff = wt_kit.diff(xi2, zi2, order=2)
                # plot the data!
                self.ax2.plot(xi2, diff, lw=2)
                self.ax2.grid(b=True)
                plt.xlabel(data.axes[0].get_label(), fontsize=18)

        # legend ----------------------------------------------------------------------------------

        #self.ax1.legend([data.name for data in self.data])

        # ticks -----------------------------------------------------------------------------------

        if not yticks:
            self.ax1.get_yaxis().set_ticks([])
        if derivative:
            self.ax2.get_yaxis().set_ticks([])
            self.ax2.axhline(0, color='k', ls=':')

        # title -----------------------------------------------------------------------------------

        if len(self.data) == 1:  # only attempt this if we are plotting one data object
            title_text = self.data[0].name
            plt.suptitle(title_text, fontsize=self.font_size)

        # finish ----------------------------------------------------------------------------------

        if xlim:
            plt.xlim(xlim[0], xlim[1])
            for axis, xi, zi in [[self.ax1, xi, zi], [self.ax2, xi2, diff]]:
                min_index = np.argmin(abs(xi - min(xlim)))
                max_index = np.argmin(abs(xi - max(xlim)))
                zi_truncated = zi[min_index:max_index]
                extra = (zi_truncated.max() - zi_truncated.min()) * 0.1
                axis.set_ylim(zi_truncated.min() - extra, zi_truncated.max() + extra)

        if ylim:
            self.ax1.set_ylim(ylim)

    def _smooth(self, dat1, n=20, window_type='default'):
        """
        data is an array of type [xlis,ylis]
        smooth to prevent 2nd derivative from being noisy
        """
        for i in range(n, len(dat1[1]) - n):
            # change the x value to the average
            window = dat1[1][i - n:i + n].copy()
            dat1[1][i] = window.mean()
        return dat1[:][:, n:-n]


class Diff2D():

    def __init__(self, minuend, subtrahend, xaxis=1, yaxis=0, at={},
                 verbose=True):
        """ plot the difference between exactly two datasets in 2D

        both data objects must have the same axes with the same name
        axes do not need to be in the same order or have the same points
        """
        self.minuend = minuend.copy()
        self.subtrahend = subtrahend.copy()
        # check if axes are valid - same axis names in both data objects
        minuend_counter = collections.Counter(self.minuend.axis_names)
        subrahend_counter = collections.Counter(self.subtrahend.axis_names)
        if minuend_counter == subrahend_counter:
            pass
        else:
            print('axes are not equivalent - difference_2D cannot initialize')
            print('  minuhend axes -', self.minuend.axis_names)
            print('  subtrahend axes -', self.subtrahend.axis_names)
            raise RuntimeError('axes incompataible')
        # transpose subrahend to agree with minuend
        transpose_order = [self.minuend.axis_names.index(
            name) for name in self.subtrahend.axis_names]
        self.subtrahend.transpose(transpose_order, verbose=False)
        # map subtrahend axes onto minuend axes
        for i in range(len(self.minuend.axes)):
            self.subtrahend.axes[i].convert(self.minuend.axes[i].units)
            self.subtrahend.map_axis(i, self.minuend.axes[i].points)
        # chop
        self.minuend_chopped = self.minuend.chop(yaxis, xaxis, at=at, verbose=False)
        self.subtrahend_chopped = self.subtrahend.chop(yaxis, xaxis, at=at, verbose=False)
        if verbose:
            print('difference_2D recieved data to make %d plots' % len(self.minuend_chopped))
        # defaults
        self.font_size = 18

    def plot(self, channel_index=0,
             contours=9, pixelated=True, cmap='default', facecolor='grey',
             dynamic_range=False, local=False, contours_local=True,
             xlim=None, ylim=None,
             autosave=False, output_folder=None, fname=None,
             verbose=True):
        """ set contours to zero to turn off

        dynamic_range forces the colorbar to use all of its colors (only matters
        for signed data)
        """
        fig = None
        if len(self.minuend_chopped) > 10:
            if not autosave:
                print('too many images will be generated: forcing autosave')
                autosave = True

        # prepare output folder
        if autosave:
            plt.ioff()
            if output_folder:
                pass
            else:
                if len(self.minuend_chopped) == 1:
                    output_folder = os.getcwd()
                    if fname:
                        pass
                    else:
                        fname = self.minuend.name
                else:
                    folder_name = 'difference_2D ' + wt_kit.get_timestamp()
                    os.mkdir(folder_name)
                    output_folder = folder_name

        # chew through image generation
        for i in range(len(self.minuend_chopped)):

            # create figure -----------------------------------------------------------------------

            if fig:
                plt.close(fig)

            fig = plt.figure(figsize=(22, 7))

            gs = grd.GridSpec(1, 6, width_ratios=[20, 20, 1, 1, 20, 1], wspace=0.1)

            subplot_main = plt.subplot(gs[0])
            subplot_main.patch.set_facecolor(facecolor)

            # levels ------------------------------------------------------------------------------

            """
            if channel.signed:

                if dynamic_range:
                    limit = min(abs(channel.znull - channel.zmin), abs(channel.znull - channel.zmax))
                else:
                    limit = max(abs(channel.znull - channel.zmin), abs(channel.znull - channel.zmax))
                levels = np.linspace(-limit + channel.znull, limit + channel.znull, 200)

            else:

                if local:
                    levels = np.linspace(channel.znull, zi.max(), 200)
                else:
                    levels = np.linspace(channel.znull, channel.zmax, 200)
            """
            levels = np.linspace(0, 1, 200)

            # main plot ---------------------------------------------------------------------------

            # get colormap
            mycm = colormaps[cmap]
            mycm.set_bad(facecolor)
            mycm.set_under(facecolor)

            for j in range(2):

                if j == 0:
                    current_chop = chopped = self.minuend_chopped[i]
                elif j == 1:
                    current_chop = self.subtrahend_chopped[i]

                axes = current_chop.axes
                channels = current_chop.channels
                constants = current_chop.constants

                xaxis = axes[1]
                yaxis = axes[0]
                channel = channels[channel_index]
                zi = channel.values

                plt.subplot(gs[j])

                # fill in main data environment
                if pixelated:
                    xi, yi, zi = pcolor_helper(xaxis.points, yaxis.points, zi)
                    cax = plt.pcolormesh(xi, yi, zi, cmap=mycm,
                                         vmin=levels.min(), vmax=levels.max())
                    plt.xlim(xaxis.points.min(), xaxis.points.max())
                    plt.ylim(yaxis.points.min(), yaxis.points.max())
                else:
                    cax = subplot_main.contourf(xaxis.points, yaxis.points, zi,
                                                levels, cmap=mycm)

                plt.xticks(rotation=45)
                #plt.xlabel(xaxis.get_label(), fontsize = self.font_size)
                #plt.ylabel(yaxis.get_label(), fontsize = self.font_size)

                # grid ----------------------------------------------------------------------------

                plt.grid(b=True)

                if xaxis.units == yaxis.units:
                    # add diagonal line
                    if xlim:
                        x = xlim
                    else:
                        x = xaxis.points
                    if ylim:
                        y = ylim
                    else:
                        y = yaxis.points

                    diag_min = max(min(x), min(y))
                    diag_max = min(max(x), max(y))
                    plt.plot([diag_min, diag_max], [diag_min, diag_max], 'k:')

                # contour lines -------------------------------------------------------------------

                if contours:
                    if contours_local:
                        # force top and bottom contour to be just outside of data range
                        # add two contours
                        contours_levels = np.linspace(
                            channel.znull - 1e-10, np.nanmax(zi) + 1e-10, contours + 2)
                    else:
                        contours_levels = contours
                    plt.contour(xaxis.points, yaxis.points, zi,
                                contours_levels, colors='k')

                # finish main subplot -------------------------------------------------------------

                if xlim:
                    subplot_main.set_xlim(xlim[0], xlim[1])
                else:
                    subplot_main.set_xlim(xaxis.points[0], xaxis.points[-1])
                if ylim:
                    subplot_main.set_ylim(ylim[0], ylim[1])
                else:
                    subplot_main.set_ylim(yaxis.points[0], yaxis.points[-1])

            # colorbar ----------------------------------------------------------------------------

            subplot_cb = plt.subplot(gs[2])
            cbar_ticks = np.linspace(levels.min(), levels.max(), 11)
            cbar = plt.colorbar(cax, cax=subplot_cb, ticks=cbar_ticks)

            # difference --------------------------------------------------------------------------

            # get colormap
            mycm = colormaps['seismic']
            mycm.set_bad(facecolor)
            mycm.set_under(facecolor)

            dzi = self.minuend_chopped[i].channels[0].values - \
                self.subtrahend_chopped[i].channels[0].values

            dax = plt.subplot(gs[4])
            plt.subplot(dax)

            X, Y, Z = pcolor_helper(xaxis.points, yaxis.points, dzi)

            largest = np.nanmax(np.abs(dzi))

            dcax = dax.pcolor(X, Y, Z, vmin=-largest, vmax=largest, cmap=mycm)

            dax.set_xlim(xaxis.points.min(), xaxis.points.max())
            dax.set_ylim(yaxis.points.min(), yaxis.points.max())

            differenc_cb = plt.subplot(gs[5])
            dcbar = plt.colorbar(dcax, cax=differenc_cb)
            dcbar.set_label(self.minuend.channels[channel_index].name +
                            ' - ' + self.subtrahend.channels[channel_index].name)

            # title -------------------------------------------------------------------------------

            title_text = self.minuend.name + ' - ' + self.subtrahend.name

            constants_text = '\n' + get_constant_text(constants)

            plt.suptitle(title_text + constants_text, fontsize=self.font_size)

            plt.figtext(0.03, 0.5, yaxis.get_label(), fontsize=self.font_size, rotation=90)
            plt.figtext(0.5, 0.01, xaxis.get_label(),
                        fontsize=self.font_size, horizontalalignment='center')

            # cleanup -----------------------------------------------------------------------------

            fig.subplots_adjust(left=0.075, right=1 - 0.075, top=0.90, bottom=0.15)

            plt.setp(plt.subplot(gs[1]).get_yticklabels(), visible=False)
            plt.setp(plt.subplot(gs[4]).get_yticklabels(), visible=False)

            # save figure -------------------------------------------------------------------------

            if autosave:
                if fname:
                    file_name = fname + ' ' + str(i).zfill(3)
                else:
                    file_name = str(i).zfill(3)
                fpath = os.path.join(output_folder, file_name + '.png')
                plt.savefig(fpath, facecolor='none')
                plt.close()

                if verbose:
                    print('image saved at', fpath)

        plt.ion()


# --- artists in progress -------------------------------------------------------------------------


class PDFAll2DSlices:

    def __init__(self, datas, name='', data_signed=False):
        """
        I'm working on this. Expect nothing.
        - Blaise 2016.03.28
        """
        self.datas = datas
        self.name = name

        self.sideplot_dictionary = {n: [] for n in self.datas[0].axis_names}
        self.data_signed = data_signed
        if self.data_signed:
            self.sideplot_limits = [-1.1, 1.1]
            self.cmap = colormaps['seismic']
        else:
            self.sideplot_limits = [0, 1.1]
            self.cmap = colormaps['default']

    def _fill_plot(self, xaxis, yaxis, zi, ax, cax, title, yticks, vmin=None,
                   vmax=None):
        xi = xaxis.points
        yi = yaxis.points
        X, Y, Z = pcolor_helper(xi, yi, zi)
        if vmax is None:
            vmax = np.nanmax(Z)
        if vmin is None:
            vmin = np.nanmin(Z)
        if self.data_signed:
            extent = max(vmax, -vmin)
            vmin = -extent
            vmax = extent
        # pcolor
        mappable = ax.pcolor(X, Y, Z, cmap=self.cmap, vmin=vmin, vmax=vmax)
        ax.set_xlim(xi.min(), xi.max())
        ax.set_ylim(yi.min(), yi.max())
        ax.grid()
        if xaxis.units_kind == yaxis.units_kind:
            diagonal_line(xi, yi, ax=ax)
        plt.setp(ax.get_yticklabels(), visible=yticks)
        # x sideplot
        sp = add_sideplot(ax, 'x')
        b = np.nansum(zi, axis=0) * len(yaxis.points)
        b[b == 0] = np.nan
        b /= np.nanmax(b)
        sp.plot(xi, b, lw=2, c='b')
        sp.set_xlim([xi.min(), xi.max()])
        sp.set_ylim(self.sideplot_limits)
        for data, channel_index, c in self.sideplot_dictionary[xaxis.name]:
            data.convert(xaxis.units, verbose=False)
            sp_xi = data.axes[0].points
            sp_zi = data.channels[channel_index].values
            sp_zi[sp_xi < xi.min()] = 0
            sp_zi[sp_xi > xi.max()] = 0
            sp_zi /= np.nanmax(sp_zi)
            sp.plot(sp_xi, sp_zi, lw=2, c=c)
        sp.grid()
        if self.data_signed:
            sp.axhline(0, c='k', lw=1)
        sp.set_title(title, fontsize=18)
        sp0 = sp
        # y sideplot
        sp = add_sideplot(ax, 'y')
        b = np.nansum(zi, axis=1) * len(xaxis.points)
        b[b == 0] = np.nan
        b /= np.nanmax(b)
        sp.plot(b, yi, lw=2, c='b')
        sp.set_xlim(self.sideplot_limits)
        sp.set_ylim([yi.min(), yi.max()])
        for data, channel_index, c in self.sideplot_dictionary[yaxis.name]:
            data.convert(xaxis.units, verbose=False)
            sp_xi = data.axes[0].points
            sp_zi = data.channels[channel_index].values
            sp_zi[sp_xi < xi.min()] = 0
            sp_zi[sp_xi > xi.max()] = 0
            sp_zi /= np.nanmax(sp_zi)
            sp.plot(sp_zi, sp_xi, lw=2, c=c)
        sp.grid()
        if self.data_signed:
            sp.axvline(0, c='k', lw=1)
        sp1 = sp
        # colorbar
        plt.colorbar(mappable=mappable, cax=cax)
        return [sp0, sp1]

    def _fill_row(self, data, channel_index, gs, row_index, global_limits):
        xaxis = data.axes[1]
        yaxis = data.axes[0]
        vmin, vmax = global_limits
        # local
        ax0 = plt.subplot(gs[row_index, 0])
        cax = plt.subplot(gs[row_index, 1])
        zi = data.channels[channel_index].values
        kwargs = {}
        if not self.data_signed:
            kwargs['vmin'] = vmin
        sps0 = self._fill_plot(xaxis, yaxis, zi, ax0, cax,
                               title=data.name + ' local', yticks=True, **kwargs)
        # global
        ax1 = plt.subplot(gs[row_index, 3])
        cax = plt.subplot(gs[row_index, 4])
        zi = data.channels[channel_index].values
        sps1 = self._fill_plot(xaxis, yaxis, zi, ax1, cax, title=data.name +
                               ' global', vmin=vmin, vmax=vmax, yticks=False)
        return [[ax0, ax1], [sps0, sps1]]

    def _label_slide(self, pdf, text):
        cols = [1, 'cbar', 0.25, 1, 'cbar']
        fig, gs = create_figure(width='double', nrows=len(self.datas), cols=cols, hspace=0.5)
        fig.text(0.5, 0.5, text, fontsize=40, ha='center', va='center')
        pdf.savefig()
        plt.close(fig)

    def sideplot(self, data, c, axes, channel=0):
        # get channel index
        if type(channel) in [int, float]:
            channel_index = int(channel)
        elif isinstance(channel, str):
            channel_index = self.datas[0].channel_names.index(channel)
        else:
            print('channel type not recognized in mpl_2D!')
        # add to sideplot_dictionary
        for axis_name in axes:
            self.sideplot_dictionary[axis_name].append([data, channel_index, c])

    def plot(self, channel=0, output_path=None, w1w2=True, w1_wigner=True,
             w2_wigner=True):
        # get channel index
        if type(channel) in [int, float]:
            channel_index = int(channel)
        elif isinstance(channel, str):
            channel_index = self.datas[0].channel_names.index(channel)
        else:
            print('channel type not recognized in mpl_2D!')
        # create pdf
        with PdfPages(output_path) as pdf:
            if w1w2:
                # 2D Frequencies
                self.chopped_datas = [d.chop('w2', 'wmw1') for d in self.datas]  # y, x
                self._label_slide(pdf, '2D frequencies')
                for slice_index in range(len(self.chopped_datas[0])):  # for each chop...
                    print('2D frequency', slice_index)
                    cols = [1, 'cbar', 0.25, 1, 'cbar']
                    fig, gs = create_figure(width='double', nrows=len(
                        self.datas), cols=cols, hspace=0.5)
                    for data_index in range(len(self.datas)):
                        data = self.chopped_datas[data_index][slice_index]
                        if self.data_signed:
                            global_limits = [self.datas[data_index].channels[channel_index].zmin,
                                             self.datas[data_index].channels[channel_index].zmax]
                        else:
                            global_limits = [self.datas[data_index].channels[channel_index].znull,
                                             self.datas[data_index].channels[channel_index].zmax]
                        axs, spss = self._fill_row(
                            data, channel_index, gs, data_index, global_limits)
                        if not data_index == len(self.datas) - 1:
                            for ax in axs:
                                plt.setp(ax.get_xticklabels(), visible=False)
                    constant_text = get_constant_text(data.constants)
                    _title(fig, self.name, constant_text)
                    pdf.savefig()
                    plt.close(fig)
            if w1_wigner:
                # w1 Wigners
                self.chopped_datas = [d.chop('d2', 'wmw1') for d in self.datas]  # y, x
                self._label_slide(pdf, 'w1 wigners')
                for slice_index in range(len(self.chopped_datas[0])):  # for each chop...
                    print('w1 wigner', slice_index)
                    cols = [1, 'cbar', 0.25, 1, 'cbar']
                    fig, gs = create_figure(width='double', nrows=len(
                        self.datas), cols=cols, hspace=0.5)
                    for data_index in range(len(self.datas)):
                        data = self.chopped_datas[data_index][slice_index]
                        if self.data_signed:
                            global_limits = [self.datas[data_index].channels[channel_index].zmin,
                                             self.datas[data_index].channels[channel_index].zmax]
                        else:
                            global_limits = [self.datas[data_index].channels[channel_index].znull,
                                             self.datas[data_index].channels[channel_index].zmax]
                        axs, spss = self._fill_row(
                            data, channel_index, gs, data_index, global_limits)
                        if not data_index == len(self.datas) - 1:
                            for ax in axs:
                                plt.setp(ax.get_xticklabels(), visible=False)
                        for ax, sps in zip(axs, spss):
                            ax.axhline(0, c='k', lw=4)
                            sps[1].axhline(0, c='k', lw=4)
                            ax.axvline(data.constants[0].points, c='k', alpha=0.5, lw=4)
                            sps[0].axvline(data.constants[0].points, c='k', alpha=0.5, lw=4)
                    constant_text = get_constant_text(data.constants)
                    _title(fig, self.name, constant_text)
                    pdf.savefig()
                    plt.close(fig)
            if w2_wigner:
                # w2 Wigners
                self.chopped_datas = [d.chop('d2', 'w2') for d in self.datas]  # y, x
                self._label_slide(pdf, 'w2 wigners')
                for slice_index in range(len(self.chopped_datas[0])):  # for each chop...
                    print('w2 wigner', slice_index)
                    cols = [1, 'cbar', 0.25, 1, 'cbar']
                    fig, gs = create_figure(width='double', nrows=len(
                        self.datas), cols=cols, hspace=0.5)
                    for data_index in range(len(self.datas)):
                        data = self.chopped_datas[data_index][slice_index]
                        if self.data_signed:
                            global_limits = [self.datas[data_index].channels[channel_index].zmin,
                                             self.datas[data_index].channels[channel_index].zmax]
                        else:
                            global_limits = [self.datas[data_index].channels[channel_index].znull,
                                             self.datas[data_index].channels[channel_index].zmax]
                        axs, spss = self._fill_row(
                            data, channel_index, gs, data_index, global_limits)
                        if not data_index == len(self.datas) - 1:
                            for ax in axs:
                                plt.setp(ax.get_xticklabels(), visible=False)
                        for ax, sps in zip(axs, spss):
                            ax.axhline(0, c='k', lw=4)
                            sps[1].axhline(0, c='k', lw=4)
                            ax.axvline(data.constants[0].points, c='k', alpha=0.5, lw=4)
                            sps[0].axvline(data.constants[0].points, c='k', alpha=0.5, lw=4)
                    constant_text = get_constant_text(data.constants)
                    _title(fig, self.name, constant_text)
                    pdf.savefig()
                    plt.close(fig)
            # pdf metadata
            d = pdf.infodict()
            d['Title'] = os.path.basename(output_path)
            d['Author'] = u'WrightTools\xe4nen'
            d['Subject'] = 'CMDS data'
            d['Keywords'] = ''
            d['CreationDate'] = datetime.datetime.today()
            d['ModDate'] = datetime.datetime.today()
