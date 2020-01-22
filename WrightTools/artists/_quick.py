"""Quick plotting."""


# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

import matplotlib.pyplot as plt

from ._helpers import _title, create_figure, plot_colorbar, savefig
from ._colors import colormaps
from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


__all__ = ["quick1D", "quick2D"]


# --- general purpose plotting functions ----------------------------------------------------------


def quick1D(
    data,
    axis=0,
    at={},
    channel=0,
    *,
    local=False,
    autosave=False,
    save_directory=None,
    fname=None,
    verbose=True
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
         Toggle autosave. Default is False.
    save_directory : string (optional)
         Location to save image(s). Default is None (auto-generated).
    fname : string (optional)
         File name. If None, data name is used. Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    list of strings
        List of saved image files (if any).
    """
    # channel index
    channel_index = wt_kit.get_index(data.channel_names, channel)
    shape = data.channels[channel_index].shape
    collapse = [i for i in range(len(shape)) if shape[i] == 1]
    at = at.copy()
    at.update({c: 0 for c in collapse})
    # prepare data
    chopped = data.chop(axis, at=at, verbose=False)
    # prepare figure
    fig = None
    if len(chopped) > 10:
        if not autosave:
            print("more than 10 images will be generated: forcing autosave")
            autosave = True
    # prepare output folders
    if autosave:
        if save_directory:
            pass
        else:
            if len(chopped) == 1:
                save_directory = os.getcwd()
                if fname:
                    pass
                else:
                    fname = data.natural_name
            else:
                folder_name = "quick1D " + wt_kit.TimeStamp().path
                os.mkdir(folder_name)
                save_directory = folder_name
    # determine ymin and ymax for global axis scale
    data_channel = data.channels[channel_index]
    ymin, ymax = data_channel.min(), data_channel.max()
    dynamic_range = ymax - ymin
    ymin -= dynamic_range * 0.05
    ymax += dynamic_range * 0.05
    if np.sign(ymin) != np.sign(data_channel.min()):
        ymin = 0
    if np.sign(ymax) != np.sign(data_channel.max()):
        ymax = 0
    # chew through image generation
    out = []
    for i, d in enumerate(chopped.values()):
        # unpack data -----------------------------------------------------------------------------
        axis = d.axes[0]
        xi = axis.full
        channel = d.channels[channel_index]
        zi = channel[:]
        # create figure ---------------------------------------------------------------------------
        aspects = [[[0, 0], 0.5]]
        fig, gs = create_figure(width="single", nrows=1, cols=[1], aspects=aspects)
        ax = plt.subplot(gs[0, 0])
        # plot ------------------------------------------------------------------------------------
        plt.plot(xi, zi, lw=2)
        plt.scatter(xi, zi, color="grey", alpha=0.5, edgecolor="none")
        # decoration ------------------------------------------------------------------------------
        plt.grid()
        # limits
        if local:
            pass
        else:
            plt.ylim(ymin, ymax)
        # label axes
        ax.set_xlabel(axis.label, fontsize=18)
        ax.set_ylabel(channel.natural_name, fontsize=18)
        plt.xticks(rotation=45)
        plt.axvline(0, lw=2, c="k")
        plt.xlim(xi.min(), xi.max())
        # add constants to title
        ls = []
        for constant in d.constants:
            ls.append(constant.label)
        title = ", ".join(ls)
        _title(fig, data.natural_name, subtitle=title)
        # variable marker lines
        for constant in d.constants:
            if constant.units is not None:
                if axis.units_kind == constant.units_kind:
                    constant.convert(axis.units)
                    plt.axvline(constant.value, color="k", linewidth=4, alpha=0.25)
        # save ------------------------------------------------------------------------------------
        if autosave:
            if fname:
                file_name = fname + " " + str(i).zfill(3)
            else:
                file_name = str(i).zfill(3)
            fpath = os.path.join(save_directory, file_name + ".png")
            savefig(fpath, fig=fig, facecolor="white")
            plt.close()
            if verbose:
                print("image saved at", fpath)
            out.append(fpath)
    chopped.close()
    return out


def quick2D(
    data,
    xaxis=0,
    yaxis=1,
    at={},
    channel=0,
    *,
    contours=0,
    pixelated=True,
    dynamic_range=False,
    local=False,
    contours_local=True,
    autosave=False,
    save_directory=None,
    fname=None,
    verbose=True
):
    """Quickly plot 2D slice(s) of data.

    Parameters
    ----------
    data : WrightTools.Data object.
        Data to plot.
    xaxis : string or integer (optional)
        Expression or index of horizontal axis. Default is 0.
    yaxis : string or integer (optional)
        Expression or index of vertical axis. Default is 1.
    at : dictionary (optional)
        Dictionary of parameters in non-plotted dimension(s). If not
        provided, plots will be made at each coordinate.
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
         Toggle autosave. Default is False.
    save_directory : string (optional)
         Location to save image(s). Default is None (auto-generated).
    fname : string (optional)
         File name. If None, data name is used. Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    list of strings
        List of saved image files (if any).
    """
    # channel index
    channel_index = wt_kit.get_index(data.channel_names, channel)
    shape = data.channels[channel_index].shape
    collapse = [i for i in range(len(shape)) if shape[i] == 1]
    at = at.copy()
    at.update({c: 0 for c in collapse})
    # prepare data
    chopped = data.chop(xaxis, yaxis, at=at, verbose=False)
    # colormap
    # get colormap
    if data.channels[channel_index].signed:
        cmap = "signed"
    else:
        cmap = "default"
    cmap = colormaps[cmap]
    cmap.set_bad([0.75] * 3, 1.0)
    cmap.set_under([0.75] * 3, 1.0)
    # fname
    if fname is None:
        fname = data.natural_name
    # autosave
    if len(chopped) > 10:
        if not autosave:
            print("more than 10 images will be generated: forcing autosave")
            autosave = True
    # output folder
    if autosave:
        if save_directory:
            pass
        else:
            if len(chopped) == 1:
                save_directory = os.getcwd()
            else:
                folder_name = "quick2D " + wt_kit.TimeStamp().path
                os.mkdir(folder_name)
                save_directory = folder_name
    # loop through image generation
    out = []
    for i, d in enumerate(chopped.values()):
        # unpack data -----------------------------------------------------------------------------
        xaxis = d.axes[0]
        xlim = xaxis.min(), xaxis.max()
        yaxis = d.axes[1]
        ylim = yaxis.min(), yaxis.max()
        channel = d.channels[channel_index]
        zi = channel[:]
        zi = np.ma.masked_invalid(zi)
        # create figure ---------------------------------------------------------------------------
        if xaxis.units == yaxis.units:
            xr = xlim[1] - xlim[0]
            yr = ylim[1] - ylim[0]
            aspect = np.abs(yr / xr)
            if 3 < aspect or aspect < 1 / 3.0:
                # TODO: raise warning here
                aspect = np.clip(aspect, 1 / 3.0, 3.0)
        else:
            aspect = 1
        fig, gs = create_figure(
            width="single", nrows=1, cols=[1, "cbar"], aspects=[[[0, 0], aspect]]
        )
        ax = plt.subplot(gs[0])
        ax.patch.set_facecolor("w")
        # levels ----------------------------------------------------------------------------------
        if channel.signed:
            if local:
                limit = channel.mag()
            else:
                data_channel = data.channels[channel_index]
                if dynamic_range:
                    limit = min(
                        abs(data_channel.null - data_channel.min()),
                        abs(data_channel.null - data_channel.max()),
                    )
                else:
                    limit = data_channel.mag()
            levels = np.linspace(-limit + channel.null, limit + channel.null, 200)
        else:
            if local:
                levels = np.linspace(channel.null, np.nanmax(zi), 200)
            else:
                data_channel = data.channels[channel_index]
                if data_channel.max() < data_channel.null:
                    levels = np.linspace(data_channel.min(), data_channel.null, 200)
                else:
                    levels = np.linspace(data_channel.null, data_channel.max(), 200)
        # colors ----------------------------------------------------------------------------------
        if pixelated:
            ax.pcolor(d, channel=channel_index, cmap=cmap, vmin=levels.min(), vmax=levels.max())
        else:
            ax.contourf(d, channel=channel_index, cmap=cmap, levels=levels)
        # contour lines ---------------------------------------------------------------------------
        if contours:
            # get contour levels
            # force top and bottom contour to be data range then clip them out
            if channel.signed:
                if contours_local:
                    limit = channel.mag()
                else:
                    limit = data_channel.mag()
                contour_levels = np.linspace(
                    -limit + channel.null, limit + channel.null, contours + 2
                )[1:-1]
            else:
                if contours_local:
                    limit = channel.max()
                else:
                    limit = data_channel.max()
                contour_levels = np.linspace(channel.null, limit, contours + 2)[1:-1]
            ax.contour(d, channel=channel_index, levels=contour_levels)
        # decoration ------------------------------------------------------------------------------
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_xlabel(xaxis.label, fontsize=18)
        ax.set_ylabel(yaxis.label, fontsize=18)
        ax.grid()
        # lims
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # add zero lines
        plt.axvline(0, lw=2, c="k")
        plt.axhline(0, lw=2, c="k")
        # add constants to title
        ls = []
        for constant in d.constants:
            ls.append(constant.label)
        title = ", ".join(ls)
        _title(fig, data.natural_name, subtitle=title)
        # variable marker lines
        for constant in d.constants:
            if constant.units is not None:
                # x axis
                if xaxis.units_kind == constant.units_kind:
                    constant.convert(xaxis.units)
                    plt.axvline(constant.value, color="k", linewidth=4, alpha=0.25)
                # y axis
                if yaxis.units_kind == constant.units_kind:
                    constant.convert(yaxis.units)
                    plt.axhline(constant.value, color="k", linewidth=4, alpha=0.25)
        # colorbar
        cax = plt.subplot(gs[1])
        cbar_ticks = np.linspace(levels.min(), levels.max(), 11)
        plot_colorbar(cax=cax, ticks=cbar_ticks, label=channel.natural_name, cmap=cmap)
        plt.sca(ax)
        # save figure -----------------------------------------------------------------------------
        if autosave:
            if fname:
                file_name = fname + " " + str(i).zfill(3)
            else:
                file_name = str(i).zfill(3)
            fpath = os.path.join(save_directory, file_name + ".png")
            savefig(fpath, fig=fig, facecolor="white")
            plt.close()
            if verbose:
                print("image saved at", fpath)
            out.append(fpath)
    chopped.close()
    return out
