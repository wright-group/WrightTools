# -*- coding: utf-8 -*-
"""
Level
=====

Leveling a dataset.
"""

import matplotlib.pyplot as plt

import WrightTools as wt
from WrightTools import datasets

fig, gs = wt.artists.create_figure(width="double", cols=[1, 1, "cbar"])

p = datasets.COLORS.v2p1_MoS2_TrEE_movie
data = wt.data.from_COLORS(p, verbose=False)
data.convert("eV")
data.ai0.symmetric_root(0.5)

# as taken
ax = plt.subplot(gs[0, 0])
chop = data.chop("w1=wm", "d2", at={"w2": [1.7, "eV"]})[0]
ax.pcolor(chop)
ax.contour(chop)

# leveled
ax = plt.subplot(gs[0, 1])
data.level(0, 2, -3)
chop = data.chop("w1=wm", "d2", at={"w2": [1.7, "eV"]})[0]
ax.pcolor(chop)
ax.contour(chop)

# label
wt.artists.set_fig_labels(xlabel=data.w1__e__wm.label, ylabel=data.d2.label)

# colorbar
cax = plt.subplot(gs[0, -1])
wt.artists.plot_colorbar(cax=cax, label="amplitude")
