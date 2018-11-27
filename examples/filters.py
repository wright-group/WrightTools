#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting Multiple Lines
=======================

A quick demonstration of how to plot multiple lines on the same
set of axes, using :meth:`create_figure` to have a set of axes
which can plot data objects directly.

The dataset is a set of optical filters transmission spectra.
"""

import WrightTools as wt
from WrightTools import datasets
from matplotlib import pyplot as plt

p = datasets.Cary.filters
col = wt.collection.from_Cary(p)

fig, gs = wt.artists.create_figure(width="double", default_aspect=.5)
ax = plt.subplot(gs[0])

for data in col.values():
    if data.natural_name in ("600LP", "550LP2"):
        continue
    data.convert("wn", verbose=False)
    ax.plot(data, label=data.natural_name)


ax.set_ylabel("%T")
ax.set_xlabel("Frequency (cm$^{-1}$)")
ax.legend()
