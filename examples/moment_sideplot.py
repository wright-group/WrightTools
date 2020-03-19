# -*- coding: utf-8 -*-
"""
Sideplots and Moments
=====================
An example showing how to use moments and sideplots.
"""

import WrightTools as wt
import matplotlib.pyplot as plt
import numpy as np

# create original arrays
x = np.linspace(-3, 3, 41)[:, None]
y = np.linspace(-3, 3, 40)[None, :]
z = np.exp(-1 * (x ** 2 + y ** 2))  # create data object
# create data object
d = wt.data.Data()
d.create_variable("x", values=x)
d.create_variable("y", values=y)
d.create_channel("z", z)
d.transform("x", "y")

# create figure
fig, gs = wt.artists.create_figure(cols=[1, "cbar"])
ax = plt.subplot(gs[0])
# instantiate sideplots
axcorrx = ax.add_sideplot(along="x", pad=0.1, ymin=-0.1, ymax=2)
axcorry = ax.add_sideplot(along="y", pad=0.1, ymin=-0.1, ymax=2)
# plot data
ax.pcolor(d, autolabel="both")
# calculate moments
d.moment(axis="x", channel=0, moment=0)  # integral
d.moment(axis="x", channel=0, moment=1)  # center of mass
d.moment(axis="y", channel=0, moment=0)  # integral
d.moment(axis="y", channel=0, moment=1)  # center of mass
# plot integral moments in sideplot
axcorrx.plot(d, channel="z_y_moment_0", color="k", linewidth=3, label="mean along x")
# this sideplot is uncouth.
# the independent axis is 'y' and the dependent is 'x'
y_ = d.y.points
x_ = d["z_x_moment_0"].points
axcorry.plot(x_, y_, color="k", linewidth=3, label="mean along y")
# plot center of mass for each slice
ax.plot(d["z_x_moment_1"].points, d.y.points, label="COM along y")
ax.plot(d.x.points, d["z_y_moment_1"].points, label="COM along x")
# legends
for ax_ in [ax, axcorrx]:
    ax_.legend(fontsize=8)
# plot colorbar
cax = plt.subplot(gs[-1])
wt.artists.plot_colorbar(cax=cax, label="z")
