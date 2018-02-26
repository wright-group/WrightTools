# -*- coding: utf-8 -*-
"""
Quick 1D
========

A quick 1D plot.
"""

import WrightTools as wt
from WrightTools import datasets

ps = datasets.KENT.LDS821_TRSF
data = wt.data.from_KENT(ps, ignore=['d1', 'd2', 'wm'], verbose=False)
wt.artists.quick1D(data, 'w1', at={'w2': [1520, 'wn']}, verbose=False)
