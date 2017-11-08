# -*- coding: utf-8 -*-
"""
Absorbance
==========

An absorbance plot.
"""

import WrightTools as wt
from WrightTools import datasets

p = datasets.JASCO.PbSe_batch_1
data = wt.data.from_JASCO(p)
data.natural_name = 'PbSe'

data.convert('wn', verbose=False)
data = data.split('wm', 10000)[0]
data = data.split('wm', 6000)[1]

artist = wt.artists.Absorbance(data)
d = artist.plot(n_smooth=50)
