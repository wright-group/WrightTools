# -*- coding: utf-8 -*-
"""
Sin
===

Here is a plot.
"""

import numpy as np
import matplotlib.pyplot as plt

xi = np.linspace(0, 10)
yi = np.sin(xi)

plt.plot(xi, yi)