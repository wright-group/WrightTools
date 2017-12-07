# -*- coding: utf-8 -*-
"""
Absorbance comparison
=====================

Comparison between two absorbance scans taken on different days.
Sample is PbSe quantum dots.
"""

import WrightTools as wt
from WrightTools import datasets

# 2012-02-21
p = datasets.JASCO.PbSe_batch_4_2012_02_21
data_02_21 = wt.data.from_JASCO(p)
data_02_21.convert('eV', verbose=False)

# 2012-03-15
p = datasets.JASCO.PbSe_batch_4_2012_03_15
data_03_15 = wt.data.from_JASCO(p)
data_03_15.convert('eV', verbose=False)

artist = wt.artists.Absorbance([data_02_21, data_03_15])
artist.plot(xlim=[0.75, 1.5])
