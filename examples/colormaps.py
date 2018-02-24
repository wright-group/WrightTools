# -*- coding: utf-8 -*-
"""
Colorbars
=========

Different colorbars.
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import WrightTools as wt
from WrightTools import datasets

fig, gs = wt.artists.create_figure(width='double', cols=[1, 1, 'cbar'], nrows=3)

p = datasets.COLORS.v2p1_MoS2_TrEE_movie
data = wt.data.from_COLORS(p, verbose=False)
data.level(0, 2, -3)
data.convert('eV')
data.ai0.symmetric_root(0.5)
data = data.chop('w1=wm', 'w2', at={'d2': [-600, 'fs']})[0]
data.ai0.normalize()

def fill_row(row, cmap):
    # greyscale
    ax = plt.subplot(gs[row, 0])
    ax.pcolor(data, cmap=wt.artists.grayify_cmap(cmap))
    # color
    ax = plt.subplot(gs[row, 1])
    ax.pcolor(data, cmap=cmap)
    # cbar
    cax = plt.subplot(gs[row, 2])
    wt.artists.plot_colorbar(cax=cax, label='amplitude', cmap=cmap)

cmap = wt.artists.colormaps['default']
fill_row(0, cmap)
cmap = wt.artists.colormaps['wright']
fill_row(1, cmap)
cmap = cm.viridis
fill_row(2, cmap)

# label
wt.artists.set_fig_labels(xlabel=data.w1__e__wm.label, ylabel=data.w2.label)
