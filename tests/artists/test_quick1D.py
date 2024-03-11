#! /usr/bin/env python3

import numpy as np
import WrightTools as wt
from WrightTools import datasets


def test_perovskite():
    p = datasets.wt5.v1p0p0_perovskite_TA  # axes w1=wm, w2, d2
    # A race condition exists where multiple tests access the same file in short order
    # this loop will open the file when it becomes available.
    while True:
        try:
            data = wt.open(p)
            break
        except:
            pass
    wt.artists.quick1D(data, axis=0, at={"w2": [1.7, "eV"], "d2": [0, "fs"]})


def test_2D():
    w1 = np.linspace(-3, 3, 51)
    w2 = np.linspace(-1, 1, 3)
    signal = w1[:, None] + w2[None, :]
    data = wt.data.Data(name="data")
    data.create_channel("signal", values=signal, signed=True)
    data.create_variable("w1", values=w1[:, None], units="wn", label="1")
    data.create_variable("w2", values=w2[None, :], units="wn", label="2")
    data.transform("w1", "w2")
    wt.artists.quick1D(data)


def test_moment_channel():
    w1 = np.linspace(-2, 2, 51)
    w2 = np.linspace(-1, 1, 3)
    w3 = np.linspace(1, 3, 5)
    signal = np.cos(w1[:, None, None] + w2[None, :, None] + w3[None, None, :])
    data = wt.data.Data(name="data")
    data.create_channel("signal", values=signal, signed=True)
    data.create_variable("w1", values=w1[:, None, None], units="wn", label="1")
    data.create_variable("w2", values=w2[None, :, None], units="wn", label="2")
    data.create_variable("w3", values=w3[None, None, :], units="wn", label="3")
    data.transform("w1", "w2", "w3")
    data.moment(2, 0, 0)
    wt.artists.quick1D(data, channel=-1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")
    test_perovskite()
    test_2D()
    test_moment_channel()
    plt.show()
