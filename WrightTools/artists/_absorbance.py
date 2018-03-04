"""Tools for visualizing data."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import matplotlib.pyplot as plt

from .. import kit as wt_kit
from ._base import create_figure

# --- classes -------------------------------------------------------------------------------------


class Absorbance:
    """Absorbance plot."""

    def __init__(self, data):
        """Absorbance plot.

        Parameters
        ----------
        data : WrightTools.data.Data object, or list thereof
           Absorbance data to plot.
        """
        raise NotImplementedError
        if not isinstance(data, list):
            data = [data]
        self.data = data

    def plot(self, channel_index=0, xlim=None, ylim=None,
             yticks=True, derivative=True, n_smooth=10,):
        """Plot.

        Parameters
        ----------
        channel_index : int (optional)
            Channel index. Default is 0.
        xlim : [xmin, xmax] (optional)
            Energy axis limits. Default is None (inherits from data).
        ylim : [ymin, ymax] (optional)
            Absorption axis limits. Default is None (inherits from data).
        yticks : boolean (optional)
            Toggle yticks. Default is True.
        derivative : boolean (optional)
            Toggle plotting derivative. Default is True.
        n_smooth : integer (optioanl)
            Smoothing factor. Default is 10.
        """
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
            xi = data.axes[0][:]
            zi = data.channels[channel_index][:]
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
                # compute second derivative
                xi2, zi2 = self._smooth(np.array([xi, zi]), n_smooth)
                diff = wt_kit.diff(xi2, zi2, order=2)
                # plot the data!
                self.ax2.plot(xi2, diff, lw=2)
                self.ax2.grid(b=True)
                plt.xlabel(data.axes[0].get_label(), fontsize=18)
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
        """Smooth to prevent 2nd derivative from being too noisy.

        Parameters
        ----------
        dat1 : 2D array
            [xlis,ylis]
        n : integer
            Smoothing amount.
        window_type : string
            Does literally nothing.
        """
        for i in range(n, len(dat1[1]) - n):
            # change the x value to the average
            window = dat1[1][i - n:i + n].copy()
            dat1[1][i] = window.mean()
        return dat1[:][:, n:-n]
