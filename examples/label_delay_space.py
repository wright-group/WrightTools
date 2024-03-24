# -*- coding: utf-8 -*-
"""
Label delay space
=================

Using WrightTools to label delay space.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import WrightTools as wt
from WrightTools import datasets

fig, gs = wt.artists.create_figure(width="double", cols=[1, 1, "cbar"])


def set_lim(ax):
    ax.set_xlim(-175, 175)
    ax.set_ylim(-175, 175)


norm = Normalize(vmin=0, vmax=1, clip=True)

# traditional delay space
ax = plt.subplot(gs[0, 0])
p = datasets.PyCMDS.d1_d2_000
data = wt.data.from_PyCMDS(p)
data.convert("fs")
data.channels[0].symmetric_root(2)
data.channels[0].normalize()
ax.pcolor(data, norm=norm)
wt.diagrams.delay.label_sectors(ax=ax)  # using default labels
set_lim(ax)
ax.set_title(r"$\mathsf{\vec{k}_1 - \vec{k}_2 + \vec{k}_{2^\prime}}$", fontsize=20)

# conjugate delay space
ax = plt.subplot(gs[0, 1])
p = datasets.PyCMDS.d1_d2_001
data = wt.data.from_PyCMDS(p)
data.convert("fs")
data.channels[0].symmetric_root(2)
data.channels[0].normalize()
art = ax.pcolor(data, norm=norm)
labels = ["II", "I", "III", "V", "VI", "IV"]
wt.diagrams.delay.label_sectors(ax=ax, labels=labels)
set_lim(ax)
ax.set_title(r"$\mathsf{\vec{k}_1 + \vec{k}_2 - \vec{k}_{2^\prime}}$", fontsize=20)

# label
wt.artists.set_fig_labels(xlabel=data.d1.label, ylabel=data.d2.label)

# colorbar
cax = plt.subplot(gs[:, -1])
fig.colorbar(art, cax=cax, label="amplitude")
