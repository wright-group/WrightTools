# -*- coding: utf-8 -*-
"""
DOVE transform
=================

An example of transform on a dataset from a DOVE experiment.
"""

import matplotlib.pyplot as plt

import WrightTools as wt
from WrightTools import datasets

p = datasets.KENT.LDS821_DOVE
data = wt.data.from_KENT(p, ignore=["d1", "d2", "wm"], verbose=False)
data.channels[0].normalize()

fig, gs = wt.artists.create_figure(width="double", cols=[1, 1, "cbar"], wspace=0.7)

# as taken
ax = plt.subplot(gs[0, 0])
data.transform("w2", "w1")
ax.pcolor(data)
wt.artists.set_ax_labels(xlabel=data.w2.label, ylabel=data.w1.label)
ax.grid()
ax.set_title("as taken", fontsize=20)

# transformed
ax = plt.subplot(gs[0, 1])
data.transform("w2", "w1-w2")
art = ax.pcolor(data)
wt.artists.set_ax_labels(xlabel=data.w2.label)
ax.grid()
ax.set_title("transformed", fontsize=20)

# colorbar
cax = plt.subplot(gs[0, -1])
fig.colorbar(art, cax=cax, label="Intensity")
