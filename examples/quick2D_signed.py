# -*- coding: utf-8 -*-
"""
Quick 2D Signed
===============

A quick 2D plot of a signed channel.
"""

import WrightTools as wt
from WrightTools import datasets

p = datasets.wt5.v1p0p0_perovskite_TA
data = wt.open(p)
wt.artists.quick2D(data, 'w1=wm', 'w2', at={'d2': [0, 'fs']}, verbose=False)
