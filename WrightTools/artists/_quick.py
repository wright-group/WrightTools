"""Quick plotting."""


# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

import matplotlib.pyplot as plt

from ._base import create_figure, plot_colorbar
from ._colors import colormaps
from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


__all__ = ['quick1D', 'quick2D']


# --- general purpose plotting functions ----------------------------------------------------------


def quick1D(data, axis, at={}, channel=0, autosave=False, save_directory=None, fname=None,
            verbose=True):
    """Quickly plot 1D slice(s) of data.

    Parameters
    ----------
    data : WrightTools.Data object
        Data to plot.
    axis : string or integer (optional)
        Expression or index of axis. Default is 0.
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
    raise NotImplementedError


def quick2D(data, xaxis=1, yaxis=0, at={}, channel=0, contours=0, pixelated=True,
            dynamic_range=False, local=False, contours_local=True, autosave=False,
            save_directory=None, fname=None, verbose=True):
    """Quickly plot 2D slice(s) of data.

    Parameters
    ----------
    data : WrightTools.Data object.
        Data to plot.
    xaxis : string or integer (optional)
        Expression or index of horizontal axis. Default is 1.
    yaxis : string or integer (optional)
        Expression or index of vertical axis. Default is 0.
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
    # prepare data
    chopped = data.chop(xaxis, yaxis, at=at, verbose=False)
    # channel index
    channel_index = wt_kit.get_index(data.channel_names, channel)
    # fname
    if fname is None:
        fname = data.natural_name
    # autosave
    if len(chopped) > 10:
        if not autosave:
            print('more than 10 images will be generated: forcing autosave')
            autosave = True
    # output folder
    if autosave:
        if save_directory:
            pass
        else:
            if len(chopped) == 1:
                save_directory = os.getcwd()
            else:
                folder_name = 'quick2D ' + wt_kit.get_timestamp(style='short')
                os.mkdir(folder_name)
                save_directory = folder_name
    # loop through image generation
    fig = None
    outfiles = [''] * len(chopped)
    for i, d in enumerate(chopped.values()):
        xaxis = d.axes[1]
        xlim = xaxis.min(), xaxis.max()
        yaxis = d.axes[0]
        ylim = xaxis.min(), yaxis.max()
        channel = d.channels[channel_index]
        zi = channel[:]
        zi = np.ma.masked_invalid(zi)
        # create figure ---------------------------------------------------------------------------
        if fig and autosave:
            plt.close(fig)
        if xaxis.units == yaxis.units:
            xr = xlim[1] - xlim[0]
            yr = ylim[1] - ylim[0]
            aspect = np.abs(yr / xr)
            if 3 < aspect or aspect < 1 / 3.:
                # TODO: raise warning here
                aspect = np.clip(aspect, 1 / 3., 3.)
        else:
            aspect = 1
        fig, gs = create_figure(width='single', nrows=1, cols=[1, 'cbar'],
                                aspects=[[[0, 0], aspect]])
        ax = plt.subplot(gs[0])
        ax.patch.set_facecolor('w')
        # levels ----------------------------------------------------------------------------------
        if channel.signed:
            if local:
                print('signed local')
                limit = max(abs(channel.null - np.nanmin(zi)),
                            abs(channel.null - np.nanmax(zi)))
            else:
                if dynamic_range:
                    limit = min(abs(channel.null - channel.min()),
                                abs(channel.null - channel.max()))
                else:
                    limit = channel.mag()
            if np.isnan(limit):
                limit = 1.
            if limit is np.ma.masked:
                limit = 1.
            levels = np.linspace(-limit + channel.null, limit + channel.null, 200)
        else:
            if local:
                levels = np.linspace(channel.null, np.nanmax(zi), 200)
            else:
                if channel.max() < channel.null:
                    levels = np.linspace(channel.min(), channel.null, 200)
                else:
                    levels = np.linspace(channel.null, channel.max(), 200)
        # main plot -------------------------------------------------------------------------------
        # get colormap
        if channel.signed:
            cmap = 'signed'
        else:
            cmap = 'default'
        cmap = colormaps[cmap]
        cmap.set_bad([0.75] * 3, 1.)
        cmap.set_under([0.75] * 3, 1.)
        # fill in main data environment
        # always plot pcolormesh
        plt.pcolor(d, cmap=cmap, vmin=levels.min(), vmax=levels.max())
        # overlap with contourf if not pixelated
        if not pixelated:
            ax.contourf(d, cmap=cmap)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_xlabel(xaxis.label, fontsize=18)
        ax.set_ylabel(yaxis.label, fontsize=18)
        # contour lines ---------------------------------------------------------------------------
        if contours:
            raise NotImplementedError
        # colorbar --------------------------------------------------------------------------------
        cax = plt.subplot(gs[1])
        cbar_ticks = np.linspace(levels.min(), levels.max(), 11)
        plot_colorbar(cax=cax, ticks=cbar_ticks, label=channel.name, cmap=cmap)
        # save figure -----------------------------------------------------------------------------
        if autosave:
            if fname:
                file_name = fname + ' ' + str(i).zfill(3)
            else:
                file_name = str(i).zfill(3)
            fpath = os.path.join(save_directory, file_name + '.png')
            plt.savefig(fpath, facecolor='none', transparent=True, dpi=300, pad_inches=1.)
            plt.close()
            if verbose:
                print('image saved at', fpath)
            outfiles[i] = fpath
    return outfiles
