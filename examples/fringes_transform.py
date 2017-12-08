# -*- coding: utf-8 -*-
"""
Fringes transform
=================

An example of transform on a dataset containing fringes.
"""

import matplotlib.pyplot as plt

import WrightTools as wt
from WrightTools import datasets

p = datasets.PyCMDS.w2_w1_000
data = wt.data.from_PyCMDS(p)

fig, gs = wt.artists.create_figure(width='double', cols=[1, 1, 'cbar'])

# as taken
ax = plt.subplot(gs[0, 0])
ax.pcolor(data.w2.full, data.w1.full, data.signal_mean)
wt.artists.set_ax_labels(xlabel=data.w2.label, ylabel=data.w1.label)
ax.set_title('as taken', fontsize=20)

# transformed
ax = plt.subplot(gs[0, 1])
data.transform(['wm', 'w1'])
data.convert('wn')
ax.pcolor(data.wm.full, data.w1.full, data.signal_mean)
wt.artists.set_ax_labels(xlabel=data.wm.label, yticks=False)
ax.set_title('transformed', fontsize=20)

# colorbar
cax = plt.subplot(gs[0, -1])
wt.artists.plot_colorbar(cax, label='amplitude')
