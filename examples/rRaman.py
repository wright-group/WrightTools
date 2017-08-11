# -*- coding: utf-8 -*-
"""
Resonance Raman
==========

A Resonance Raman plot.
"""

import WrightTools as wt
from WrightTools import datasets

p = datasets.BrunoldrRaman.LDS821_514nm_80mW
data = wt.data.from_BrunoldrRaman(p)
trash_pixels = 56
data = data.split(0, 843.0)[1]

data.convert('wn', verbose=False)

artist = wt.artists.mpl_1D(data)
d = artist.plot()
