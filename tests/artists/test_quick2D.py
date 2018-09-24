#! /usr/bin/env python3

import numpy as np
import WrightTools as wt
from WrightTools import datasets


def test_perovskite():
    p = datasets.wt5.v1p0p0_perovskite_TA  # axes w1=wm, w2, d2
    data = wt.open(p)
    return wt.artists.quick2D(data, xaxis=0, yaxis=1, at={"d2": [0, "fs"]})


p = datasets.wt5.v1p0p0_perovskite_TA
data = wt.open(p)
chopped = data.chop(0, 1, at={"d2": [0, "fs"]}, verbose=False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")
    # store to variable to prevent garbage collection
    # t0 = test_perovskite()

    plt.show()
