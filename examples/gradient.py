# -*- coding: utf-8 -*-
"""
Gradient
========

Demonstration of the gradient method.
"""

import numpy as np
import WrightTools as wt

data = wt.data.Data()
data.create_variable("w1", np.linspace(-10, 10, 100))
data.create_channel("sig", 1 / (np.pi * (1 + (data.w1[:] - 1) ** 2)))
data.transform("w1")
data.gradient("w1")

wt.artists.quick1D(data)
wt.artists.quick1D(data, channel="sig_w1_gradient")
