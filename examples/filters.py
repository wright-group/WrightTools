#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optical Filters
===============

A set of optical filters transmission spectra.
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
