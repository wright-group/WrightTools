# -*- coding: utf-8 -*-
"""
Sideplots and Moments
=====================
An example showing how to use moments and sideplots.
"""

import WrightTools as wt
import numpy as np
import matplotlib.pyplot as plt


def S(x):
    return (1 + np.exp(-1 * x)) ** -1


# instantiate data object
d1 = np.linspace(-1, 3, 101)[:, None]
d2 = np.linspace(-1, 3, 102)[None, :]
arr = S(10 * (d1 + 0.5)) * S(10 * d2) * S(-2 * d1) + 0.1 * S(5 * d1) * S(5 * (d2 - 1))
d = wt.Data(name="test")
d.create_variable("d1", values=d1, units="ps", label="1")
d.create_variable("d2", values=d2, units="ps", label="2")
d.create_channel("z", values=arr)
d.transform("d1", "d2")

# calculate moments
d.moment(axis="d1", channel=0, moment=0)  # integral
d["z_d1_moment_0"].normalize()
d.moment(axis="d1", channel=0, moment=1)  # center of mass
d.moment(axis="d2", channel=0, moment=0)  # integral
d["z_d2_moment_0"].normalize()
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
axcorrx.plot(d, channel="z_d2_moment_0", color="k", linewidth=3)
# this sideplot is uncouth.
# the independent axis is 'y' and the dependent is 'x'
y_ = d.d2.points
x_ = d["z_d1_moment_0"].points
axcorry.plot(x_, y_, color="k", linewidth=3)
# plot center of mass for each slice
ax.plot(d["z_d1_moment_1"].points, d.d2.points, label="COM along y")
ax.plot(d.d1.points, d["z_d2_moment_1"].points, label="COM along x")
ax.legend(fontsize=12, loc=4)
# grids
for ax_ in [ax, axcorrx, axcorry]:
    ax_.grid()
# plot colorbar
cax = plt.subplot(gs[-1])
wt.artists.plot_colorbar(cax=cax, label="amplitude")
