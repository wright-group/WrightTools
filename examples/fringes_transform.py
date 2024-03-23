# -*- coding: utf-8 -*-
"""
Fringes transform
=================

An example of transform on a dataset containing fringes.
"""

import matplotlib.pyplot as plt

import WrightTools as wt
from WrightTools import datasets

try:
    p = datasets.PyCMDS.w2_w1_000
except AttributeError as e:
    e.add_note(f"valid attrs are {datasets.PyCMDS.__dict__.items()}")
    e.add_note(f"KENT has other attrs: {datasets.KENT.__dict__.items()}")
    raise

p = datasets.here / "PyCMDS" / "w2 w1 000.data"
data = wt.data.from_PyCMDS(p)

data.signal_mean.symmetric_root(2)  # to amplitude level
data.convert("wn")

fig, gs = wt.artists.create_figure(width="double", cols=[1, 1, "cbar"])

# as taken
ax = plt.subplot(gs[0, 0])
ax.pcolor(data)
wt.artists.set_ax_labels(xlabel=data.w2.label, ylabel=data.w1.label)
ax.grid()
ax.set_title("as taken", fontsize=20)

# transformed
ax = plt.subplot(gs[0, 1])
data.transform("wm", "w1")
data.convert("wn")
art = ax.pcolor(data)
wt.artists.set_ax_labels(xlabel=data.wm.label, yticks=False)
ax.grid()
ax.set_title("transformed", fontsize=20)

# colorbar
cax = plt.subplot(gs[0, -1])
fig.colorbar(art, cax, label="amplitude")
