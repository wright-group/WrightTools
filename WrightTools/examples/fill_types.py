# -*- coding: utf-8 -*-
"""
Fill types
==========

Different ways to plot 2D data.
"""

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import WrightTools as wt
from WrightTools import datasets

cmap = wt.artists.colormaps["default"]

fig, gs = wt.artists.create_figure(width="double", nrows=2, cols=[1, 1, 1, 1, "cbar"])

# get data
p = datasets.COLORS.v0p2_d1_d2_diagonal
data = wt.data.from_COLORS(p, invert_d1=False)
data.level(0, 0, 1)
data.ai0.symmetric_root(2)
data.ai0.normalize()
data.ai0.clip(min=0, replace="value")


def dot_pixel_centers(ax, xi, yi):
    for x in xi:
        ax.scatter([x] * len(xi), yi, edgecolor=None, s=5, color="k")


def decorate(ax):
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)


# pcolor
ax = plt.subplot(gs[0, 0])
ax.pcolor(data, cmap=cmap)
ax.set_title("pcolor", fontsize=20)
decorate(ax)
ax = plt.subplot(gs[1, 0])
ax.pcolor(data, cmap=cmap, edgecolors="k")
dot_pixel_centers(ax, data.d1.points, data.d2.points)
decorate(ax)

# tripcolor
xi = data.d1.points
yi = data.d2.points
zi = data.channels[0][:].T
ax = plt.subplot(gs[0, 1])
points = [xi, yi]
x, y = tuple(np.meshgrid(*points, indexing="ij"))
ax.tripcolor(x.flatten(), y.flatten(), zi.T.flatten(), cmap=cmap, vmin=0, vmax=1)
decorate(ax)
ax.set_title("tripcolor", fontsize=20)
ax = plt.subplot(gs[1, 1])
ax.tripcolor(x.flatten(), y.flatten(), zi.T.flatten(), edgecolor="k", cmap=cmap, vmin=0, vmax=1)
decorate(ax)
dot_pixel_centers(ax, xi, yi)


def plot_delaunay_edges(ax, xi, yi, zi):
    x, y = tuple(np.meshgrid(xi, yi, indexing="ij"))
    tri = matplotlib.tri.Triangulation(x.flatten(), y.flatten())
    for i, j in tri.edges:
        plt.plot(
            [x.flatten()[i], x.flatten()[j]], [y.flatten()[i], y.flatten()[j]], c="k", lw=0.25
        )
        ax.set_xlim(-125, 125)
        ax.set_ylim(-125, 125)


# contourf
ax = plt.subplot(gs[0, 2])
ax.contourf(data, vmin=-1e-3)
decorate(ax)
ax.set_title("contourf", fontsize=20)
ax = plt.subplot(gs[1, 2])
ax.contourf(data, vmin=-1e-3)
plot_delaunay_edges(ax, xi, yi, zi)
dot_pixel_centers(ax, xi, yi)
decorate(ax)

# contour
ax = plt.subplot(gs[0, 3])
ax.contour(data)
decorate(ax)
ax.set_title("contour", fontsize=20)
ax = plt.subplot(gs[1, 3])
ax.contour(data)
plot_delaunay_edges(ax, xi, yi, zi)
dot_pixel_centers(ax, xi, yi)
decorate(ax)

# label
ticks = [-100, 0, 100]
wt.artists.set_fig_labels(xlabel=data.d1.label, ylabel=data.d2.label, xticks=ticks, yticks=ticks)

# colorbar
cax = plt.subplot(gs[:, -1])
wt.artists.plot_colorbar(cax=cax, label="amplitude")
