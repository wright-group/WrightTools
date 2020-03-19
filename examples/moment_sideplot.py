# -*- coding: utf-8 -*-
"""
Sideplots and Moments
=====================
An example showing how to use moments and sideplots.
"""

import WrightTools as wt
from WrightTools import datasets
import matplotlib.pyplot as plt

# get dataset
p = datasets.PyCMDS.d1_d2_000
d = wt.data.from_PyCMDS(p)
d.convert("fs")
d.channels[0].symmetric_root(2)
d.channels[0].normalize()
d.channels[0].clip(min=0, replace="value")
# calculate moments
d.moment(axis="d1", channel=0, moment=0)  # integral
d["signal_diff_d1_moment_0"].normalize()
d.moment(axis="d1", channel=0, moment=1)  # center of mass
d.moment(axis="d2", channel=0, moment=0)  # integral
d["signal_diff_d2_moment_0"].normalize()
d.moment(axis="d2", channel=0, moment=1)  # center of mass

# create figure
fig, gs = wt.artists.create_figure(cols=[1, "cbar"])
ax = plt.subplot(gs[0])
# instantiate sideplots
axcorrx = ax.add_sideplot(along="x", pad=0.1, ymin=-0.1, ymax=1.1)
axcorry = ax.add_sideplot(along="y", pad=0.1, ymin=-0.1, ymax=1.1)
# plot data
ax.pcolor(d, autolabel="both")
# plot integral moments in sideplot
axcorrx.plot(d, channel="signal_diff_d2_moment_0", color="k", linewidth=3)
# this sideplot is uncouth.
# the independent axis is 'y' and the dependent is 'x'
y_ = d.d2.points
x_ = d["signal_diff_d1_moment_0"].points
axcorry.plot(x_, y_, color="k", linewidth=3)
# plot center of mass for each slice
ax.plot(d["signal_diff_d2_moment_1"].points, d.d2.points, label="COM along y")
ax.plot(d.d1.points, d["signal_diff_d1_moment_1"].points, label="COM along x")
ax.legend(fontsize=8)
# grids
for ax_ in [ax, axcorrx, axcorry]:
    ax_.grid()
axcorrx.set_title(r"$\mathsf{\vec{k}_1 - \vec{k}_2 + \vec{k}_{2^\prime}}$", fontsize=20)
# plot colorbar
cax = plt.subplot(gs[-1])
wt.artists.plot_colorbar(cax=cax, label="amplitude")
