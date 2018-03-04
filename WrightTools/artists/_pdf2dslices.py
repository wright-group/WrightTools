"""PDF 2D slices."""
# --- import --------------------------------------------------------------------------------------

import datetime
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import _title, add_sideplot, create_figure, colormaps, pcolor_helper
from ._base import get_constant_text, diagonal_line


class PDF2DSlices:
    """PDF 2D slices."""

    def __init__(self, datas, name='', data_signed=False):
        """Initialize the 2D slice PDF generator.

        Parameters
        ----------
        datas : list of WrightTools.data.Data objects
            Datas to plot.
        name : string (optional)
            Name. Default is ''.
        data_signed : boolean (optional)
            Toggle data signed. Default is False.
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
        xi = xaxis[:]
        yi = yaxis[:]
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
        b = np.nansum(zi, axis=0) * len(yaxis[:])
        b[b == 0] = np.nan
        b /= np.nanmax(b)
        sp.plot(xi, b, lw=2, c='b')
        sp.set_xlim([xi.min(), xi.max()])
        sp.set_ylim(self.sideplot_limits)
        for data, channel_index, c in self.sideplot_dictionary[xaxis.name]:
            data.convert(xaxis.units, verbose=False)
            sp_xi = data.axes[0][:]
            sp_zi = data.channels[channel_index][:]
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
        b = np.nansum(zi, axis=1) * len(xaxis[:])
        b[b == 0] = np.nan
        b /= np.nanmax(b)
        sp.plot(b, yi, lw=2, c='b')
        sp.set_xlim(self.sideplot_limits)
        sp.set_ylim([yi.min(), yi.max()])
        for data, channel_index, c in self.sideplot_dictionary[yaxis.name]:
            data.convert(xaxis.units, verbose=False)
            sp_xi = data.axes[0][:]
            sp_zi = data.channels[channel_index][:]
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
        zi = data.channels[channel_index][:]
        kwargs = {}
        if not self.data_signed:
            kwargs['vmin'] = vmin
        sps0 = self._fill_plot(xaxis, yaxis, zi, ax0, cax,
                               title=data.name + ' local', yticks=True, **kwargs)
        # global
        ax1 = plt.subplot(gs[row_index, 3])
        cax = plt.subplot(gs[row_index, 4])
        zi = data.channels[channel_index][:]
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
        """Add sideplot."""
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
        """Plot."""
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
                            global_limits = [self.datas[data_index].channels[channel_index].min(),
                                             self.datas[data_index].channels[channel_index].max()]
                        else:
                            global_limits = [self.datas[data_index].channels[channel_index].null(),
                                             self.datas[data_index].channels[channel_index].max()]
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
                            global_limits = [self.datas[data_index].channels[channel_index].min(),
                                             self.datas[data_index].channels[channel_index].max()]
                        else:
                            global_limits = [self.datas[data_index].channels[channel_index].null(),
                                             self.datas[data_index].channels[channel_index].max()]
                        axs, spss = self._fill_row(
                            data, channel_index, gs, data_index, global_limits)
                        if not data_index == len(self.datas) - 1:
                            for ax in axs:
                                plt.setp(ax.get_xticklabels(), visible=False)
                        for ax, sps in zip(axs, spss):
                            ax.axhline(0, c='k', lw=4)
                            sps[1].axhline(0, c='k', lw=4)
                            ax.axvline(data.constants[0][:], c='k', alpha=0.5, lw=4)
                            sps[0].axvline(data.constants[0][:], c='k', alpha=0.5, lw=4)
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
                            global_limits = [self.datas[data_index].channels[channel_index].min(),
                                             self.datas[data_index].channels[channel_index].max()]
                        else:
                            global_limits = [self.datas[data_index].channels[channel_index].null(),
                                             self.datas[data_index].channels[channel_index].max()]
                        axs, spss = self._fill_row(
                            data, channel_index, gs, data_index, global_limits)
                        if not data_index == len(self.datas) - 1:
                            for ax in axs:
                                plt.setp(ax.get_xticklabels(), visible=False)
                        for ax, sps in zip(axs, spss):
                            ax.axhline(0, c='k', lw=4)
                            sps[1].axhline(0, c='k', lw=4)
                            ax.axvline(data.constants[0][:], c='k', alpha=0.5, lw=4)
                            sps[0].axvline(data.constants[0][:], c='k', alpha=0.5, lw=4)
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
