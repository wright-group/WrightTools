#! /usr/bin/env python3

import pytest
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
    return wt.artists.interact2D(data, xaxis=2, yaxis=1)


def test_MoS2():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie  # axes w2, w1=wm, d2
    data = wt.open(p)
    data.convert("eV")
    data.level(0, 2, -4)
    return wt.artists.interact2D(data, xaxis=0, yaxis=1, local=True)


def test_asymmetric():
    # 2 axes, easy to identify which is which
    x = np.arange(6)
    y = x[::2].copy()
    z = np.arange(x.size * y.size).reshape(x.size, y.size).astype("float")
    z[:, y < 2] *= 0
    data = wt.data.Data(name="data")
    data.create_channel("signal", values=z, signed=False)
    data.create_variable("x", values=x[:, None], units="wn")
    data.create_variable("y", values=y[None, :], units="wn")
    data.transform("x", "y")
    return wt.artists.interact2D(data, xaxis=1, yaxis=0)


def test_skewed():
    # non-orthogonal axes (still single variable)
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.convert("wn", convert_variables=True)
    data.transform("wm", "w1")  # wm = w1 + 2*w2
    return wt.artists.interact2D(data, xaxis=0, yaxis=1)


@pytest.mark.xfail(reason="sometimes fails for artificial reasons")
def test_4D():
    w1 = np.linspace(-5, 5, 31)
    w2 = w1[::2].copy()
    w3 = w1.copy()
    tau = np.linspace(-1, 3, 21)

    signal = (
        (w1[:, None, None, None] - 1j)
        * (w2[None, :, None, None] - 1j)
        * (w3[None, None, :, None] - 1j)
    ) ** -1 * np.exp(-tau[None, None, None, :])
    signal += (
        2
        * (
            (w1[:, None, None, None] - 1 - 1j)
            * (w2[None, :, None, None] - 1 - 1j)
            * (w3[None, None, :, None] - 1 - 1j)
        )
        ** -1
        * np.exp(-2 * tau[None, None, None, :])
    )
    signal[:, :, :, tau < 0] = 0
    signal[:, :, :, tau == 0] *= 0.5
    signal = np.imag(signal)

    data = wt.data.Data(name="data")
    data.create_channel("signal", values=signal, signed=True)
    data.create_variable("w_1", values=w1[:, None, None, None], units="wn")
    data.create_variable("w_2", values=w2[None, :, None, None], units="wn")
    data.create_variable("w_3", values=w3[None, None, :, None], units="wn")
    data.create_variable("d_1", values=tau[None, None, None, :], units="ps")

    data.transform("w_1", "w_2", "w_3", "d_1")
    return wt.artists.interact2D(data, xaxis=0, yaxis=1, local=True, use_imshow=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")
    out1 = test_perovskite()
    out2 = test_MoS2()
    out3 = test_asymmetric()
    out4 = test_skewed()
    out5 = test_4D()
    plt.show()
