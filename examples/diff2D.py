# -*- coding: utf-8 -*-
"""
Difference 2D
=============

Difference 2D.
"""

import WrightTools as wt
from WrightTools import datasets

ps = datasets.KENT.LDS821_TRSF
data = wt.data.from_KENT(ps, ignore=['d1', 'd2', 'wm'], verbose=False)

artist = wt.artists.mpl_2D(data, verbose=False)
artist.plot()
