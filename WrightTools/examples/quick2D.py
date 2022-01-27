# -*- coding: utf-8 -*-
"""
Quick 2D
========

A quick 2D plot.
"""

import WrightTools as wt
from WrightTools import datasets

ps = datasets.KENT.LDS821_TRSF
data = wt.data.from_KENT(ps, ignore=["d1", "d2", "wm"], verbose=False)
wt.artists.quick2D(data, verbose=False)
