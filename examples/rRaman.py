# -*- coding: utf-8 -*-
"""
Resonance Raman
===============

A Resonance Raman plot.
"""

import WrightTools as wt
from WrightTools import datasets

p = datasets.BrunoldrRaman.LDS821_514nm_80mW
data = wt.data.from_BrunoldrRaman(p)

data.convert("wn", verbose=False)

wt.artists.quick1D(data)
