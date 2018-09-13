"""Functions to help with plotting."""


# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

from scipy.interpolate import interp2d

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable

import imageio
import warnings

from .. import exceptions as wt_exceptions
from .. import kit as wt_kit
from ._base import Figure, GridSpec
from ._colors import colormaps


# --- define --------------------------------------------------------------------------------------


__all__ = [
    "_title",
    "add_sideplot",
    "corner_text",
    "create_figure",
    "diagonal_line",
    "get_scaled_bounds",
    "pcolor_helper",
    "plot_colorbar",
    "plot_margins",
    "plot_gridlines",
    "savefig",
    "set_ax_labels",
    "set_ax_spines",
    "set_fig_labels",
    "subplots_adjust",
    "stitch_to_animation",
]


# --- functions -----------------------------------------------------------------------------------


def _title(fig, title, subtitle="", *, margin=1, fontsize=20, subfontsize=18):
    """Add a title to a figure.

    Parameters
    ----------
    fig : matplotlib Figure
        Figure.
    title : string
        Title.
    subtitle : string
        Subtitle.
    margin : number (optional)
        Distance from top of plot, in inches. Default is 1.
    fontsize : number (optional)
        Title fontsize. Default is 20.
    subfontsize : number (optional)
        Subtitle fontsize. Default is 18.
    """
    fig.suptitle(title, fontsize=fontsize)
    height = fig.get_figheight()  # inches
    distance = margin / 2.  # distance from top of plot, in inches
    ratio = 1 - distance / height
    fig.text(0.5, ratio, subtitle, fontsize=subfontsize, ha="center", va="top")


def add_sideplot(
    ax,
    along,
    pad=0.,
    *,
    grid=True,
    zero_line=True,
    arrs_to_bin=None,
    normalize_bin=True,
    ymin=0,
    ymax=1.1,
    height=0.75,
    c="C0"
):
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
    if hasattr(ax, "WrightTools_sideplot_divider"):
        divider = ax.WrightTools_sideplot_divider
    else:
        divider = make_axes_locatable(ax)
        setattr(ax, "WrightTools_sideplot_divider", divider)
    # create sideplot axis
    if along == "x":
        axCorr = divider.append_axes("top", height, pad=pad, sharex=ax)
    elif along == "y":
        axCorr = divider.append_axes("right", height, pad=pad, sharey=ax)
    axCorr.autoscale(False)
    axCorr.set_adjustable("box")
    # bin
    if arrs_to_bin is not None:
        xi, yi, zi = arrs_to_bin
        if along == "x":
            b = np.nansum(zi, axis=0) * len(yi)
            if normalize_bin:
                b /= np.nanmax(b)
            axCorr.plot(xi, b, c=c, lw=2)
        elif along == "y":
            b = np.nansum(zi, axis=1) * len(xi)
            if normalize_bin:
                b /= np.nanmax(b)
            axCorr.plot(b, yi, c=c, lw=2)
    # beautify
    if along == "x":
        axCorr.set_ylim(ymin, ymax)
        axCorr.tick_params(axis="x", which="both", length=0)
    elif along == "y":
        axCorr.set_xlim(ymin, ymax)
        axCorr.tick_params(axis="y", which="both", length=0)
    plt.grid(grid)
    if zero_line:
        if along == "x":
            plt.axhline(0, c="k", lw=1)
        elif along == "y":
            plt.axvline(0, c="k", lw=1)
    plt.setp(axCorr.get_xticklabels(), visible=False)
    plt.setp(axCorr.get_yticklabels(), visible=False)
    return axCorr


def corner_text(
    text,
    distance=0.075,
    *,
    ax=None,
    corner="UL",
    factor=200,
    bbox=True,
    fontsize=18,
    background_alpha=1,
    edgecolor=None
):
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
    [h_scaled, v_scaled], [va, ha] = get_scaled_bounds(
        ax, corner, distance=distance, factor=factor
    )
    # get edgecolor
    if edgecolor is None:
        edgecolor = matplotlib.rcParams["legend.edgecolor"]
    # apply text
    props = dict(boxstyle="square", facecolor="white", alpha=background_alpha, edgecolor=edgecolor)
    args = [v_scaled, h_scaled, text]
    kwargs = {}
    kwargs["fontsize"] = fontsize
    kwargs["verticalalignment"] = va
    kwargs["horizontalalignment"] = ha
    if bbox:
        kwargs["bbox"] = props
    else:
        kwargs["path_effects"] = [PathEffects.withStroke(linewidth=3, foreground="w")]
    kwargs["transform"] = ax.transAxes
    if "zlabel" in ax.properties().keys():  # axis is 3D projection
        out = ax.text2D(*args, **kwargs)
    else:
        out = ax.text(*args, **kwargs)
    return out


def create_figure(
    *,
    width="single",
    nrows=1,
    cols=[1],
    margin=1.,
    hspace=0.25,
    wspace=0.25,
    cbar_width=0.25,
    aspects=[],
    default_aspect=1
):
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
        matplotlib gridspec__ documentation for more information.

        __ http://matplotlib.org/users/gridspec.html#gridspec-and-subplotspec


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
    wt.artists.subplots_adjust
        Enforce margins for figure generated elsewhere.

    """
    # get width
    if width == "double":
        figure_width = 14.
    elif width == "single":
        figure_width = 6.5
    elif width == "dissertation":
        figure_width = 13.
    else:
        figure_width = float(width)
    # check if aspect constraints are valid
    rows_in_aspects = [item[0][0] for item in aspects]
    if not len(rows_in_aspects) == len(set(rows_in_aspects)):
        raise Exception("can only specify aspect for one plot in each row")
    # get width avalible to subplots (not including colorbars)
    total_subplot_width = figure_width - 2 * margin
    total_subplot_width -= (len(cols) - 1) * wspace  # whitespace in width
    total_subplot_width -= cols.count("cbar") * cbar_width  # colorbar width
    # converters

    def in_to_mpl(inches, total, n):
        return (inches * n) / (total - inches * n + inches)

    def mpl_to_in(mpl, total, n):
        return (total / (n + mpl * (n - 1))) * mpl

    # calculate column widths, width_ratio
    subplot_ratios = np.array([i for i in cols if not i == "cbar"], dtype=np.float)
    subplot_ratios /= sum(subplot_ratios)
    subplot_widths = total_subplot_width * subplot_ratios
    width_ratios = []
    cols_idxs = []
    i = 0
    for col in cols:
        if not col == "cbar":
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
    gs = GridSpec(
        nrows,
        len(cols),
        hspace=hspace,
        wspace=wspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    # finish
    subplots_adjust(fig, inches=margin)
    return fig, gs


def diagonal_line(xi=None, yi=None, *, ax=None, c=None, ls=None, lw=None, zorder=3):
    """Plot a diagonal line.

    Parameters
    ----------
    xi : 1D array-like (optional)
        The x axis points. If None, taken from axis limits. Default is None.
    yi : 1D array-like
        The y axis points. If None, taken from axis limits. Default is None.
    ax : axis (optional)
        Axis to plot on. If none is supplied, the current axis is used.
    c : string (optional)
        Line color. Default derives from rcParams grid color.
    ls : string (optional)
        Line style. Default derives from rcParams linestyle.
    lw : float (optional)
        Line width. Default derives from rcParams linewidth.
    zorder : number (optional)
        Matplotlib zorder. Default is 3.

    Returns
    -------
    matplotlib.lines.Line2D object
        The plotted line.
    """
    if ax is None:
        ax = plt.gca()
    # parse xi, yi
    if xi is None:
        xi = ax.get_xlim()
    if yi is None:
        yi = ax.get_ylim()
    # parse style
    if c is None:
        c = matplotlib.rcParams["grid.color"]
    if ls is None:
        ls = matplotlib.rcParams["grid.linestyle"]
    if lw is None:
        lw = matplotlib.rcParams["grid.linewidth"]
    # get axis
    if ax is None:
        ax = plt.gca()
    # make plot
    diag_min = max(min(xi), min(yi))
    diag_max = min(max(xi), max(yi))
    line = ax.plot([diag_min, diag_max], [diag_min, diag_max], c=c, ls=ls, lw=lw, zorder=zorder)
    return line


def get_scaled_bounds(ax, position, *, distance=0.1, factor=200):
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
    if position == "UL":
        v_scaled = distance / width_scaled
        h_scaled = 1 - (distance / height_scaled)
        va = "top"
        ha = "left"
    elif position == "LL":
        v_scaled = distance / width_scaled
        h_scaled = distance / height_scaled
        va = "bottom"
        ha = "left"
    elif position == "UR":
        v_scaled = 1 - (distance / width_scaled)
        h_scaled = 1 - (distance / height_scaled)
        va = "top"
        ha = "right"
    elif position == "LR":
        v_scaled = 1 - (distance / width_scaled)
        h_scaled = distance / height_scaled
        va = "bottom"
        ha = "right"
    else:
        print("corner not recognized")
        v_scaled = h_scaled = 1.
        va = "center"
        ha = "center"
    return [h_scaled, v_scaled], [va, ha]


def pcolor_helper(xi, yi, zi=None):
    """Prepare a set of arrays for plotting using `pcolor`.

    The return values are suitable for feeding directly into ``matplotlib.pcolor``
    such that the pixels are properly centered.

    Parameters
    ----------
    xi : 1D or 2D array-like
        Array of X-coordinates.
    yi : 1D or 2D array-like
        Array of Y-coordinates.
    zi : 2D array (optional, deprecated)
        If zi is not None, it is returned unchanged in the output.

    Returns
    -------
    X : 2D ndarray
        X dimension for pcolor
    Y : 2D ndarray
        Y dimension for pcolor
    zi : 2D ndarray
        if zi parameter is not None, returns zi parameter unchanged
    """
    xi = xi.copy()
    yi = yi.copy()
    if xi.ndim == 1:
        xi.shape = (xi.size, 1)
    if yi.ndim == 1:
        yi.shape = (1, yi.size)
    shape = wt_kit.joint_shape(xi, yi)

    # full
    def full(arr):
        for i in range(arr.ndim):
            if arr.shape[i] == 1:
                arr = np.repeat(arr, shape[i], axis=i)
        return arr

    xi = full(xi)
    yi = full(yi)
    # pad
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    f_xi = interp2d(x, y, xi)
    f_yi = interp2d(x, y, yi)
    x_new = np.arange(-1, shape[1] + 1)
    y_new = np.arange(-1, shape[0] + 1)
    xi = f_xi(x_new, y_new)
    yi = f_yi(x_new, y_new)
    # fill
    X = np.empty([s - 1 for s in xi.shape])
    Y = np.empty([s - 1 for s in yi.shape])
    for orig, out in [[xi, X], [yi, Y]]:
        for idx in np.ndindex(out.shape):
            ul = orig[idx[0] + 1, idx[1] + 0]
            ur = orig[idx[0] + 1, idx[1] + 1]
            ll = orig[idx[0] + 0, idx[1] + 0]
            lr = orig[idx[0] + 0, idx[1] + 1]
            out[idx] = np.mean([ul, ur, ll, lr])
    if zi is not None:
        warnings.warn(
            "zi argument is not used in pcolor_helper and is not required",
            wt_exceptions.VisibleDeprecationWarning,
        )
        return X, Y, zi.copy()
    else:
        return X, Y


def plot_colorbar(
    cax=None,
    cmap="default",
    ticks=None,
    clim=None,
    vlim=None,
    label=None,
    tick_fontsize=14,
    label_fontsize=18,
    decimals=None,
    orientation="vertical",
    ticklocation="auto",
    extend="neither",
    extendfrac=None,
    extendrect=False,
):
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
    extend : {‘neither’, ‘both’, ‘min’, ‘max’} (optional)
        If not ‘neither’, make pointed end(s) for out-of- range values.
        These are set for a given colormap using the colormap set_under and set_over methods.
    extendfrac :	{None, ‘auto’, length, lengths} (optional)
        If set to None, both the minimum and maximum triangular colorbar extensions
        have a length of 5% of the interior colorbar length (this is the default setting).
        If set to ‘auto’, makes the triangular colorbar extensions the same lengths
        as the interior boxes
        (when spacing is set to ‘uniform’) or the same lengths as the respective adjacent
        interior boxes (when spacing is set to ‘proportional’).
        If a scalar, indicates the length of both the minimum and maximum triangular
        colorbar extensions as a fraction of the interior colorbar length.
        A two-element sequence of fractions may also be given, indicating the lengths
        of the minimum and maximum colorbar extensions respectively as a fraction
        of the interior colorbar length.
    extendrect :	bool (optional)
        If False the minimum and maximum colorbar extensions will be triangular (the default).
        If True the extensions will be rectangular.

    Returns
    -------
    matplotlib.colorbar.ColorbarBase object
        The created colorbar.
    """
    # parse cax
    if cax is None:
        cax = plt.gca()
    # parse cmap
    if isinstance(cmap, str):
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
    if max(vlim) == min(vlim):
        vlim[-1] += 1e-1
    # parse format
    if isinstance(decimals, int):
        format = "%.{0}f".format(decimals)
    else:
        magnitude = int(np.log10(max(vlim) - min(vlim)) - 0.99)
        if 1 > magnitude > -3:
            format = "%.{0}f".format(-magnitude + 1)
        elif magnitude in (1, 2, 3):
            format = "%i"
        else:
            # scientific notation
            def fmt(x, _):
                return "%.1f" % (x / float(10 ** magnitude))

            format = matplotlib.ticker.FuncFormatter(fmt)
            magnitude_label = r"  $\mathsf{\times 10^{%d}}$" % magnitude
            if label is None:
                label = magnitude_label
            else:
                label = " ".join([label, magnitude_label])
    # make cbar
    norm = matplotlib.colors.Normalize(vmin=vlim[0], vmax=vlim[1])
    cbar = matplotlib.colorbar.ColorbarBase(
        ax=cax,
        cmap=cmap,
        norm=norm,
        ticks=ticks,
        orientation=orientation,
        ticklocation=ticklocation,
        format=format,
        extend=extend,
        extendfrac=extendfrac,
        extendrect=extendrect,
    )
    # coerce properties
    cbar.set_clim(clim[0], clim[1])
    cbar.ax.tick_params(labelsize=tick_fontsize)
    if label:
        cbar.set_label(label, fontsize=label_fontsize)
    # finish
    return cbar


def plot_margins(*, fig=None, inches=1., centers=True, edges=True):
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
    left = matplotlib.lines.Line2D(
        [trans_vert, trans_vert], [0, 1], transform=fig.transFigure, figure=fig
    )
    right = matplotlib.lines.Line2D(
        [1 - trans_vert, 1 - trans_vert], [0, 1], transform=fig.transFigure, figure=fig
    )
    trans_horz = inches / size[1]
    bottom = matplotlib.lines.Line2D(
        [0, 1], [trans_horz, trans_horz], transform=fig.transFigure, figure=fig
    )
    top = matplotlib.lines.Line2D(
        [0, 1], [1 - trans_horz, 1 - trans_horz], transform=fig.transFigure, figure=fig
    )
    fig.lines.extend([left, right, bottom, top])
    if centers:
        vert = matplotlib.lines.Line2D(
            [0.5, 0.5], [0, 1], transform=fig.transFigure, figure=fig, c="r"
        )
        horiz = matplotlib.lines.Line2D(
            [0, 1], [0.5, 0.5], transform=fig.transFigure, figure=fig, c="r"
        )
        fig.lines.extend([vert, horiz])
    if edges:
        left = matplotlib.lines.Line2D(
            [0, 0], [0, 1], transform=fig.transFigure, figure=fig, c="k"
        )
        right = matplotlib.lines.Line2D(
            [1, 1], [0, 1], transform=fig.transFigure, figure=fig, c="k"
        )
        bottom = matplotlib.lines.Line2D(
            [0, 1], [0, 0], transform=fig.transFigure, figure=fig, c="k"
        )
        top = matplotlib.lines.Line2D([0, 1], [1, 1], transform=fig.transFigure, figure=fig, c="k")
        fig.lines.extend([left, right, bottom, top])


def plot_gridlines(ax=None, c="grey", lw=1, diagonal=False, zorder=2, makegrid=True):
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
    ls = ":"
    dashes = (lw / 2, lw)
    # grid
    # ax.grid(True)
    lines = ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines()
    for line in lines.copy():
        line.set_linestyle(":")
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
        ax.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            c=c,
            ls=ls,
            lw=lw,
            zorder=zorder,
            dashes=dashes,
        )

        # Plot resets xlim and ylim sometimes for unknown reasons.
        # This is here to ensure that the xlim and ylim are unchanged
        # after adding a diagonal, whose limits are calculated so
        # as to not change the xlim and ylim.
        #           -- KFS 2017-09-26
        ax.set_ylim(min_yi, max_yi)
        ax.set_xlim(min_xi, max_xi)


def savefig(path, fig=None, close=True, *, dpi=300):
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
    plt.savefig(path, dpi=dpi, transparent=False, pad_inches=1, facecolor="none")
    # close
    if close:
        plt.close(fig)
    # finish
    return path


def set_ax_labels(ax=None, xlabel=None, ylabel=None, xticks=None, yticks=None, label_fontsize=18):
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
                ax.tick_params(axis="x", which="both", length=0)
        else:
            ax.set_xticks(xticks)
    # y
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if yticks is not None:
        if isinstance(yticks, bool):
            plt.setp(ax.get_yticklabels(), visible=yticks)
            if not yticks:
                ax.tick_params(axis="y", which="both", length=0)
        else:
            ax.set_yticks(yticks)


def set_ax_spines(ax=None, *, c="k", lw=3, zorder=10):
    """Easily set the properties of all four axis spines.

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
    for key in ["bottom", "top", "right", "left"]:
        ax.spines[key].set_color(c)
        ax.spines[key].set_linewidth(lw)
        ax.spines[key].zorder = zorder


def set_fig_labels(
    fig=None,
    xlabel=None,
    ylabel=None,
    xticks=None,
    yticks=None,
    title=None,
    label_fontsize=18,
    title_fontsize=20,
):
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
                label_fontsize=label_fontsize,
            )
        elif ax.is_first_col():
            # lefthand column
            set_ax_labels(
                ax=ax, ylabel=ylabel, xticks=False, yticks=yticks, label_fontsize=label_fontsize
            )
        elif ax.is_last_row():
            # bottom row
            set_ax_labels(
                ax=ax, xlabel=xlabel, xticks=xticks, yticks=False, label_fontsize=label_fontsize
            )
        else:
            set_ax_labels(ax=ax, xticks=False, yticks=False)
    # title
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)


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


def stitch_to_animation(images, outpath=None, *, duration=0.5, palettesize=256, verbose=True):
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
        outpath = os.path.splitext(images[0])[0] + ".gif"
    # write
    t = wt_kit.Timer(verbose=False)
    with t, imageio.get_writer(
        outpath, mode="I", duration=duration, palettesize=palettesize
    ) as writer:
        for p in images:
            image = imageio.imread(p)
            writer.append_data(image)
    # finish
    if verbose:
        interval = np.round(t.interval, 2)
        print("gif generated in {0} seconds - saved at {1}".format(interval, outpath))
    return outpath
