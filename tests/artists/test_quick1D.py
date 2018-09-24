#! /usr/bin/env python3

import numpy as np
import WrightTools as wt
from WrightTools import datasets


def test_perovskite():
    p = datasets.wt5.v1p0p0_perovskite_TA  # axes w1=wm, w2, d2
    data = wt.open(p)
    return wt.artists.quick1D(data, axis=0, at={"w2": [1.7, "eV"], "d2": [0, "fs"]})


def test_2D():
    w1 = np.linspace(-3, 3, 51)
    w2 = np.linspace(-1, 1, 3)
    signal = w1[:, None] + w2[None, :]
    data = wt.data.Data(name="data")
    data.create_channel("signal", values=signal, signed=True)
    data.create_variable("w1", values=w1[:, None], units="wn", label="1")
    data.create_variable("w2", values=w2[None, :], units="wn", label="2")
    data.transform("w1", "w2")
    return wt.artists.quick1D(data)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")
    # store to variable to prevent garbage collection
    t0 = test_perovskite()
    t1 = test_2D()
    plt.show()
