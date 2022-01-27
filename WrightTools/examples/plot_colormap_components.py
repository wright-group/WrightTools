# -*- coding: utf-8 -*-
"""
Plot colormap components
========================

Quickly plot the RGB components of a colormap.
"""

import WrightTools as wt

cmap = wt.artists.colormaps["default"]
wt.artists.plot_colormap_components(cmap)
