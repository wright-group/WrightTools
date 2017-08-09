# -*- coding: utf-8 -*-
"""
Simple 1D
=========

A simple 1D plot.
"""

import WrightTools as wt
from WrightTools import datasets

ps = datasets.KENT.LDS821_TRSF
data = wt.data.from_KENT(ps, ignore=['d1', 'd2', 'wm'], verbose=False)
data.name = 'LDS'

artist = wt.artists.mpl_1D(data, 'w1', at={'w2': [1520, 'wn']}, verbose=False)
artist.plot()
