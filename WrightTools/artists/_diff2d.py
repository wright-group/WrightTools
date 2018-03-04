"""Diff2D."""
# --- import --------------------------------------------------------------------------------------

import collections
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

from ..WrightTools import kit as wt_kit
from ._base import pcolor_helper, get_constant_text
from ._colors import colormaps


class Diff2D():
    """Diff2D."""

    def __init__(self, minuend, subtrahend, xaxis=1, yaxis=0, at={},
                 verbose=True):
        """Plot the difference between exactly two datasets in 2D.

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
            self.subtrahend.map_axis(i, self.minuend.axes[i][:])
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
             autosave=False, save_directory=None, fname=None,
             verbose=True):
        """Set contours to zero to turn off.

        dynamic_range forces the colorbar to use all of its colors (only matters
        for signed data)
        """
        # TODO: add parameters to this plot function (KS)
        fig = None
        if len(self.minuend_chopped) > 10:
            if not autosave:
                print('too many images will be generated: forcing autosave')
                autosave = True
        # prepare output folder
        if autosave:
            plt.ioff()
            if save_directory:
                pass
            else:
                if len(self.minuend_chopped) == 1:
                    save_directory = os.getcwd()
                    if fname:
                        pass
                    else:
                        fname = self.minuend.name
                else:
                    folder_name = 'difference_2D ' + wt_kit.get_timestamp()
                    os.mkdir(folder_name)
                    save_directory = folder_name
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
            levels = np.linspace(0, 1, 200)
            # main plot ---------------------------------------------------------------------------
            # get colormap
            mycm = colormaps[cmap]
            mycm.set_bad(facecolor)
            mycm.set_under(facecolor)
            for j in range(2):
                if j == 0:
                    current_chop = self.minuend_chopped[i]
                elif j == 1:
                    current_chop = self.subtrahend_chopped[i]
                axes = current_chop.axes
                channels = current_chop.channels
                constants = current_chop.constants
                xaxis = axes[1]
                yaxis = axes[0]
                channel = channels[channel_index]
                zi = channel[:]
                plt.subplot(gs[j])
                # fill in main data environment
                if pixelated:
                    xi, yi, zi = pcolor_helper(xaxis[:], yaxis[:], zi)
                    cax = plt.pcolormesh(xi, yi, zi, cmap=mycm,
                                         vmin=levels.min(), vmax=levels.max())
                    plt.xlim(xaxis.min(), xaxis.max())
                    plt.ylim(yaxis.min(), yaxis.max())
                else:
                    cax = subplot_main.contourf(xaxis[:], yaxis[:], zi,
                                                levels, cmap=mycm)
                plt.xticks(rotation=45)
                # grid ----------------------------------------------------------------------------
                plt.grid(b=True)
                if xaxis.units == yaxis.units:
                    # add diagonal line
                    if xlim:
                        x = xlim
                    else:
                        x = xaxis[:]
                    if ylim:
                        y = ylim
                    else:
                        y = yaxis[:]

                    diag_min = max(min(x), min(y))
                    diag_max = min(max(x), max(y))
                    plt.plot([diag_min, diag_max], [diag_min, diag_max], 'k:')
                # contour lines -------------------------------------------------------------------
                if contours:
                    if contours_local:
                        # force top and bottom contour to be just outside of data range
                        # add two contours
                        contours_levels = np.linspace(
                            channel.null() - 1e-10, np.nanmax(zi) + 1e-10, contours + 2)
                    else:
                        contours_levels = contours
                    plt.contour(xaxis[:], yaxis[:], zi,
                                contours_levels, colors='k')
                # finish main subplot -------------------------------------------------------------
                if xlim:
                    subplot_main.set_xlim(xlim[0], xlim[1])
                else:
                    subplot_main.set_xlim(xaxis[:][0], xaxis[:][-1])
                if ylim:
                    subplot_main.set_ylim(ylim[0], ylim[1])
                else:
                    subplot_main.set_ylim(yaxis[:][0], yaxis[:][-1])
            # colorbar ----------------------------------------------------------------------------
            subplot_cb = plt.subplot(gs[2])
            cbar_ticks = np.linspace(levels.min(), levels.max(), 11)
            plt.colorbar(cax, cax=subplot_cb, ticks=cbar_ticks)
            # difference --------------------------------------------------------------------------
            # get colormap
            mycm = colormaps['seismic']
            mycm.set_bad(facecolor)
            mycm.set_under(facecolor)
            dzi = self.minuend_chopped[i].channels[0][:] - \
                self.subtrahend_chopped[i].channels[0][:]
            dax = plt.subplot(gs[4])
            plt.subplot(dax)
            X, Y, Z = pcolor_helper(xaxis[:], yaxis[:], dzi)
            largest = np.nanmax(np.abs(dzi))
            dcax = dax.pcolor(X, Y, Z, vmin=-largest, vmax=largest, cmap=mycm)
            dax.set_xlim(xaxis.min(), xaxis.max())
            dax.set_ylim(yaxis.min(), yaxis.max())
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
                fpath = os.path.join(save_directory, file_name + '.png')
                plt.savefig(fpath, facecolor='none')
                plt.close()
                if verbose:
                    print('image saved at', fpath)
        plt.ion()
