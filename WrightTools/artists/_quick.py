"""Quick plotting."""


# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

import matplotlib.pyplot as plt

from ._helpers import create_figure, plot_colorbar, savefig, add_sideplot
from ._colors import colormaps
from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


__all__ = ["quick1D", "quick2D", "quick2D_interactive"]


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
    # prepare data
    chopped = data.chop(axis, at=at, verbose=False)
    # channel index
    channel_index = wt_kit.get_index(data.channel_names, channel)
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
                folder_name = "mpl_1D " + wt_kit.TimeStamp().path
                os.mkdir(folder_name)
                save_directory = folder_name
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
            data_channel = data.channels[channel_index]
            plt.ylim(data_channel.min(), data_channel.max())
        # label axes
        ax.set_xlabel(axis.label, fontsize=18)
        ax.set_ylabel(channel.name, fontsize=18)
        plt.xticks(rotation=45)
        plt.xlim(xi.min(), xi.max())
        # save ------------------------------------------------------------------------------------
        if autosave:
            if fname:
                file_name = fname + " " + str(i).zfill(3)
            else:
                file_name = str(i).zfill(3)
            fpath = os.path.join(save_directory, file_name + ".png")
            savefig(fpath, fig=fig)
            plt.close()
            if verbose:
                print("image saved at", fpath)
            out.append(fpath)
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
    verbose=True,
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
    # prepare data
    chopped = data.chop(xaxis, yaxis, at=at, verbose=False)
    # channel index
    channel_index = wt_kit.get_index(data.channel_names, channel)
    # colormap
    # get colormap
    if data.channels[channel_index].signed:
        cmap = "signed"
    else:
        cmap = "default"
    cmap = colormaps[cmap]
    cmap.set_bad([0.75] * 3, 1.)
    cmap.set_under([0.75] * 3, 1.)
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
        ylim = xaxis.min(), yaxis.max()
        channel = d.channels[channel_index]
        zi = channel[:]
        zi = np.ma.masked_invalid(zi)
        # create figure ---------------------------------------------------------------------------
        if xaxis.units == yaxis.units:
            xr = xlim[1] - xlim[0]
            yr = ylim[1] - ylim[0]
            aspect = np.abs(yr / xr)
            if 3 < aspect or aspect < 1 / 3.:
                # TODO: raise warning here
                aspect = np.clip(aspect, 1 / 3., 3.)
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
            # print(channel.name, d.channel_names, channel, channel_index)
            ax.pcolor(d, channel=channel_index, cmap=cmap, vmin=levels.min(), vmax=levels.max())
        else:
            ax.contourf(d, channel=channel_index, cmap=cmap, levels=levels)
        # contour lines ---------------------------------------------------------------------------
        if contours:
            raise NotImplementedError
        # decoration ------------------------------------------------------------------------------
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_xlabel(xaxis.label, fontsize=18)
        ax.set_ylabel(yaxis.label, fontsize=18)
        # colorbar
        cax = plt.subplot(gs[1])
        cbar_ticks = np.linspace(levels.min(), levels.max(), 11)
        plot_colorbar(cax=cax, ticks=cbar_ticks, label=channel.natural_name, cmap=cmap)
        # save figure -----------------------------------------------------------------------------
        if autosave:
            if fname:
                file_name = fname + " " + str(i).zfill(3)
            else:
                file_name = str(i).zfill(3)
            fpath = os.path.join(save_directory, file_name + ".png")
            savefig(fpath, fig=fig)
            plt.close()
            if verbose:
                print("image saved at", fpath)
            out.append(fpath)
    return out


def quick2D_interactive(
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
    verbose=True
):
    """
    Quickly plot 2D slice of data with interaction features projections
    (qtx). Cursor clicks on the 2D subplot will select which slices to
    show on the side plots for 3+ dimensional data. Interactive slider(s)
    are used to select which 2D slice of the data is shown.

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

    def draw_int_sideplots(ax, data, channel_index):
        sp_x = add_sideplot(ax, 'x', pad=0.3)
        sp_y = add_sideplot(ax, 'y', pad=0.3)

        xi = data.axes[0].points
        yi = data.axes[1].points
        zi = data.channels[channel_index][:]

        ylims = ax0.get_ylim()
        xlims = ax0.get_xlim()

        spx_indices = np.where(np.logical_and(xi > xlims[0],
                                              xi < xlims[1]))[0]
        spy_indices = np.where(np.logical_and(yi > ylims[0],
                                              yi < ylims[1]))[0]

        x_proj = zi[spx_indices[:, None], spy_indices[None, :]].mean(axis=0)
        x_proj /= x_proj.max()
        y_proj = zi[spx_indices[:, None], spy_indices[None, :]].mean(axis=1)
        y_proj /= y_proj.max()
        sp_x.fill_between(xi[spx_indices], y_proj, 0, color='k', alpha=0.3)
        sp_y.fill_betweenx(yi[spy_indices], x_proj, 0, color='k', alpha=0.3)
        return sp_x, sp_y

    def draw_side_plots(x0, y0):
        x_temp = np.abs(data.axes[0].points - x0)
        x_index = np.argmin(x_temp)
        side_plot = data.signal[x_index]
        line_sp_y.set_data(side_plot / side_plot.max(), data.axes[1][:])
        line_sp_y.set_visible(True)

        y_temp = np.abs(data.axes[1].points - y0)
        y_index = np.argmin(y_temp)
        side_plot = data.signal[:, y_index]
        line_sp_x.set_data(data.axes[0][:, 0], side_plot / side_plot.max())
        line_sp_x.set_visible(True)

    def draw_crosshairs(x0, y0, xlim, ylim):
        if verbose:
            print('drawing crosshairs at ({0}, {1})'.format(x0, y0))
        crosshair_along_x.set_data(np.array([xlim, [y0, y0]]))
        crosshair_along_x.set_visible(True)
        ax0.add_line(crosshair_along_x)
        crosshair_along_y.set_data(np.array([[x0, x0], ylim]))
        crosshair_along_y.set_visible(True)
        ax0.add_line(crosshair_along_y)
        plt.matplotlib.pyplot.draw()

    def handle_click_release(erelease):
        x0 = erelease.xdata
        y0 = erelease.ydata
        xlim = ax0.get_xlim()
        ylim = ax0.get_ylim()
        if x0 > xlim[0] and x0 < xlim[1] and y0 > ylim[0] and y0 < ylim[1]:
            if erelease.button == 1:  # left click
                pass
            elif erelease.button == 3:  # right click
                pass
            else:
                pass
        draw_side_plots(x0, y0)
        draw_crosshairs(x0, y0, xlim, ylim)

    # prepare data
    # TODO:  delay use of chop for interactive sliders
    # channel index
    channel_index = wt_kit.get_index(data.channel_names, channel)
    # colormap
    # get colormap
    if data.channels[channel_index].signed:
        cmap = "signed"
    else:
        cmap = "default"
    cmap = colormaps[cmap]
    cmap.set_bad([0.75] * 3, 1.)
    cmap.set_under([0.75] * 3, 1.)

    xaxis = data.axes[0]
    xlim = xaxis.min(), xaxis.max()

    yaxis = data.axes[1]
    ylim = xaxis.min(), yaxis.max()

    channel = data.channels[channel_index]
    zi = channel[:]
    zi = np.ma.masked_invalid(zi)
    # initialize figure objects ---------------------------------------------------------------
    if xaxis.units == yaxis.units:
        xr = xlim[1] - xlim[0]
        yr = ylim[1] - ylim[0]
        aspect = np.abs(yr / xr)
        if 3 < aspect or aspect < 1 / 3.:
            # TODO: raise warning here
            aspect = np.clip(aspect, 1 / 3., 3.)
    else:
        aspect = 1
    fig, gs = create_figure(
        width="single", nrows=1, cols=[1, "cbar"], aspects=[[[0, 0], aspect]]
    )
    ax0 = plt.subplot(gs[0])
    ax0.patch.set_facecolor("w")
    ax0.grid(b=True)

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
        ax0.pcolormesh(data, channel=channel_index,
                       cmap=cmap, vmin=levels.min(), vmax=levels.max())
        #ax0.pcolormesh(data.axes[0].points[::5], data.axes[1].points[::10], data.signal[::5, ::10].T,
        #               channel=channel_index, cmap=cmap, vmin=levels.min(), vmax=levels.max())
    else:
        ax0.contourf(data, channel=channel_index, cmap=cmap, levels=levels)
    # contour lines ---------------------------------------------------------------------------
    if contours:
        raise NotImplementedError
    # initialize sideplots, dynamic objects
    sp_x, sp_y = draw_int_sideplots(ax0, data, channel_index)

    xlims = ax0.get_xlim()
    ylims = ax0.get_ylim()

    crosshair_along_x = ax0.plot([xlims], [ylims], visible=False,
                                 color='b', alpha=0.3, linewidth=1)[0]
    crosshair_along_y = ax0.plot([xlims], [ylims], visible=False,
                                 color='b', alpha=0.3, linewidth=1)[0]

    line_sp_x = sp_x.plot([0], [0], visible=False, color='b')[0]
    line_sp_y = sp_y.plot([0], [0], visible=False, color='b')[0]

    side_plotter = plt.matplotlib.widgets.AxesWidget(ax0)
    side_plotter.connect_event('button_release_event', handle_click_release)

    # decoration ------------------------------------------------------------------------------
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    ax0.set_xlabel(xaxis.label, fontsize=18)
    ax0.set_ylabel(yaxis.label, fontsize=18)
    # colorbar
    cax = plt.subplot(gs[1])
    cbar_ticks = np.linspace(levels.min(), levels.max(), 11)
    plot_colorbar(cax=cax, ticks=cbar_ticks, label=channel.natural_name, cmap=cmap)
