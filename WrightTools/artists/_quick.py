"""Quick plotting."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._base import _title, create_figure, pcolor_helper, plot_colorbar, get_constant_text
from ._base import diagonal_line
from ._colors import colormaps
from .. import kit as wt_kit


# --- define --------------------------------------------------------------------------------------


__all__ = ['Quick2D', 'Quick1D']


# --- general purpose artists ---------------------------------------------------------------------


class Quick1D:
    """matplotlib 1D."""

    def __init__(self, data, xaxis=0, at={}, verbose=True):
        """Plot generic 1D slice(s) quickly and easily.

        Parameters
        ----------
        data : WrightTools.data.Data object.
            The data object to plot.
        xaxis : string or int (optional)
            xaxis name or index. Default is zero.
        at : dictionary (optional)
            Dictionary of parameters in non-plotted dimension(s). If not
            provided, plots will be made at each coordinate.
        verbose : boolean (optional)
            Toggle talkback. Default is True.
        """
        # import data
        self.data = data
        self.chopped = self.data.chop(xaxis, at=at, verbose=False)
        if verbose:
            print('mpl_1D recieved data to make %d plots' % len(self.chopped))
        # defaults
        self.font_size = 15

    def plot(self, channel=0, local=False, autosave=False, save_directory=None,
             fname=None, lines=True, verbose=True):
        """Plot.

        Parameters
        ----------
        channel : string or int (optional)
            Name or index of plotted channel. Default is zero.
        local : boolean (optional)
            Toggle rescaling for each generated plot. Default is False.
        autosave : boolean (optional)
            Toggle saving. Default is False.
        save_directory : path (optional)
            Location to place generated images. If None, a new timestamped
            folder will be created. Default is None.
        fname : string (optional)
            Name of generated image files. If None, a simple name is used.
            Default is None.
        lines : boolean (optional)
            Toggle plotting of lines to indicate value of constant axes in a
            visual way. Default is True.
        verbose : boolean (optional)
            Toggle talkback. Default is True.

        Returns
        -------
        list of filepaths
            List of generated filepaths, if files were saved.
        """
        # get channel index
        if type(channel) in [int, float]:
            channel_index = int(channel)
        elif isinstance(channel, str):
            channel_index = self.chopped[0].channel_names.index(channel)
        else:
            print('channel type not recognized in mpl_1D!')
        # prepare figure
        fig = None
        if len(self.chopped) > 10:
            if not autosave:
                print('too many images will be generated ({}): forcing autosave'.format(
                    len(self.chopped)))
                autosave = True
        # prepare output folders
        if autosave:
            if save_directory:
                pass
            else:
                if len(self.chopped) == 1:
                    save_directory = os.getcwd()
                    if fname:
                        pass
                    else:
                        fname = self.data.natural_name
                else:
                    folder_name = 'mpl_1D ' + wt_kit.TimeStamp().path
                    os.mkdir(folder_name)
                    save_directory = folder_name
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
                plt.ylim(channels[channel_index].min(), channels[channel_index].max())
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
                fpath = os.path.join(save_directory, file_name + '.png')
                plt.savefig(fpath, transparent=True, dpi=300, pad_inches=1.)
                plt.close()
                if verbose:
                    print('image saved at', fpath)
                outfiles[i] = fpath
        return outfiles


class Quick2D:
    """matplotlib 2D."""

    def __init__(self, data, xaxis=1, yaxis=0, at={}, verbose=True):
        """Plot generic 2D slice(s) quickly and easily.

        Parameters
        ----------
        data : WrightTools.data.Data object.
            The data object to plot.
        xaxis : string or int (optional)
            xaxis name or index. Default is 1.
        yaxis : string or int (optional)
            yaxis name or index. Default is 0.
        at : dictionary (optional)
            Dictionary of parameters in non-plotted dimension(s). If not
            provided, plots will be made at each coordinate.
        verbose : boolean (optional)
            Toggle talkback. Default is True.
        """
        # import data
        self.data = data
        #self.chopped = self.data.chop(yaxis, xaxis, at=at, verbose=False)
        self.chopped = [data]  # hack
        if verbose:
            print('mpl_2D recieved data to make %d plots' % len(self.chopped))
        # defaults
        self._xsideplot = False
        self._ysideplot = False
        self._xsideplotdata = []
        self._ysideplotdata = []
        self._onplotdata = []

    def get_lims(self, transform=None):
        """Find plot limits using transform.

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
        -------
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
            xlim = (self.xaxis.min(), self.xaxis.max())
            ylim = (self.yaxis.min(), self.yaxis.max())
        return xlim, ylim

    def sideplot(self, data, x=True, y=True):
        """Add data to sideplot(s).

        Parameters
        ----------
        data : 1D WrightTools.data.Data object
            Data to add to sideplot.
        x : boolean (optional)
            Toggle plotting along horizontal sideplot. Default is True.
        y : boolean (optional)
            Toggle plotting along vertical sideplot. Default is True.
        """
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
        """Plot a line directly onto the plot.

        Parameters
        ----------
        xi : 1D array-like
            X points.
        yi : 1D array-like
            Y points.
        c : matplotlib color (optional)
            Line color. Default is 'k'.
        lw : number (optional)
            Line width. Default is 5.
        alpha : number (optional)
            Line opacity. Default is 0.3
        **kwargs
            Additional matplotlib.pyplot.plot arguments.
        """
        kwargs['c'] = c
        kwargs['lw'] = lw
        kwargs['alpha'] = alpha
        self._onplotdata.append((xi, yi, kwargs))

    def plot(self, channel=0,
             contours=0, pixelated=True, lines=True, cmap='automatic', facecolor='w',
             dynamic_range=False, local=False, contours_local=True, xbin=False, ybin=False,
             xlim=None, ylim=None, autosave=False, save_directory=None, fname=None, verbose=True,
             transform=None, contour_thickness=None):
        """Draw the plot(s).

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
        save_directory : str (optional)
            Output folder.
        fname : str (optional)
            File name. If None, data name is used. Default is None.
        verbose : bool (optional)
            Toggle talkback. Default is True.
        """
        # get channel index
        if type(channel) in [int, float]:
            channel_index = int(channel)
        elif isinstance(channel, str):
            channel_index = self.chopped[0].channel_names.index(channel)
        else:
            print('channel type not recognized in mpl_2D!')
        # get fname
        if fname:
            pass
        else:
            fname = self.data.natural_name
        # prepare figure
        fig = None
        if len(self.chopped) > 10:
            if not autosave:
                print('too many images will be generated: forcing autosave')
                autosave = True
        # prepare output folder
        if autosave:
            if save_directory:
                pass
            else:
                if len(self.chopped) == 1:
                    save_directory = os.getcwd()
                else:
                    folder_name = 'mpl_2D ' + wt_kit.get_timestamp(style='short')
                    os.mkdir(folder_name)
                    save_directory = folder_name
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
            zi = channel[:]
            zi = np.ma.masked_invalid(zi)
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
                X, Y, Z = pcolor_helper(self.xaxis, self.yaxis,
                                        zi, transform=transform)
                cax = plt.pcolormesh(X, Y, Z, cmap=mycm,
                                     vmin=levels.min(), vmax=levels.max())
            plt.xlim(self.xaxis.min(), self.xaxis.max())
            plt.ylim(self.yaxis.min(), self.yaxis.max())
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
                # Ensure that X, Y are that expected by contour, not pcolor
                X, Y = np.meshgrid(self.xaxis.points, self.yaxis.points)
                if contours_local:
                    # force top and bottom contour to be just outside of data range
                    # add two contours
                    contours_levels = np.linspace(
                        channel.null() - 1e-10, np.nanmax(zi) + 1e-10, contours + 2)
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
                    x_ax_int = np.nansum(zi, axis=0) - channel.null() * len(self.yaxis.points)
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
                    y_ax_int = np.nansum(zi, axis=1) - channel.null() * len(self.xaxis.points)
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
            cax = plt.subplot(gs[1])
            cbar_ticks = np.linspace(levels.min(), levels.max(), 11)
            plot_colorbar(cax=cax, ticks=cbar_ticks, label=channel.name, cmap=mycm)
            # title -------------------------------------------------------------------------------
            title_text = self.data.natural_name
            constants_text = get_constant_text(constants)
            _title(fig, title_text, constants_text)
            # save figure -------------------------------------------------------------------------
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
