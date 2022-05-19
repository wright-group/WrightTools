#! /usr/bin/env python3


import WrightTools as wt
from WrightTools import datasets
from matplotlib import pyplot as plt
import shutil
import os

import copy


cmap = copy.copy(plt.cm.get_cmap("jet"))
cmap.set_under("k")
cmap.set_over("w")

# plt.figure()
# cax = plt.subplot(111)
# wt.artists.plot_colorbar(cmap=cmap, cax=cax, extend="both")
# plt.show()

fig, gs = wt.artists.create_figure(width="double", cols=[1])
cax = plt.subplot(gs[0])
plt.show()
