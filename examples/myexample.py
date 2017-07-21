# -*- coding: utf-8 -*-
"""
My Example
==========

My example.
"""

import WrightTools as wt
from WrightTools import datasets

ps = datasets.KENT.LDS821_TRSF
data = wt.data.from_KENT(ps, ignore=['d1', 'd2', 'wm'])

artist = wt.artists.mpl_2D(data)
artist.plot()
