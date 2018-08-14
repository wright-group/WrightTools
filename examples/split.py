"""Split

Show some examples of how splitting works
"""
import numpy as np
from matplotlib import pyplot as plt
import WrightTools as wt
from WrightTools import datasets

d = wt.data.from_PyCMDS(datasets.PyCMDS.w2_w1_000)

d.convert("wn", convert_variables=True)

a = d.split("w2", [7000, 8000])
b = d.split("w1+w2+7000", [20400, 23000], units="wn")
d.create_variable("strange", values=d.channels[0].points / d.channels[0].max())
c = d.split("strange", [.2, .4])

fig, gs = wt.artists.create_figure(nrows=len(c), cols=[1, 1, 1])
for i, da in enumerate(a.values()):
    ax = plt.subplot(gs[i, 0])
    ax.pcolor(da)
    ax.set_xlim(d.axes[0].min(), d.axes[0].max())
    ax.set_ylim(d.axes[1].min(), d.axes[1].max())

for i, da in enumerate(b.values()):
    ax = plt.subplot(gs[i, 1])
    ax.pcolor(da)
    ax.set_xlim(d.axes[0].min(), d.axes[0].max())
    ax.set_ylim(d.axes[1].min(), d.axes[1].max())

for i, da in enumerate(c.values()):
    ax = plt.subplot(gs[i, 2])
    ax.pcolor(da)
    ax.set_xlim(d.axes[0].min(), d.axes[0].max())
    ax.set_ylim(d.axes[1].min(), d.axes[1].max())

wt.artists.set_fig_labels(xlabel=d.axes[0].label, ylabel=d.axes[1].label)
