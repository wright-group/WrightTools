import numpy as np
import WrightTools as wt
from WrightTools import datasets

if True:
    p = datasets.wt5.v1p0p0_perovskite_TA  # axes w1=wm, w2, d2
    data = wt.open(p)

if False:
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie  # axes w2, w1=wm, d2
    data = wt.open(p)
    data.convert("eV")

if False:
    x = np.arange(6)
    y = x[::2].copy()
    z = np.arange(x.size * y.size).reshape(x.size, y.size).astype("float")
    z[:, y < 2] *= 0
    data = wt.data.Data(name="data")
    data.create_channel("signal", values=z, signed=False)
    data.create_variable("x", values=x[:, None], units="wn")
    data.create_variable("y", values=y[None, :], units="wn")
    data.transform("x", "y")

if False:
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
    signal = np.abs(signal)

    data = wt.data.Data(name="data")
    data.create_channel("signal", values=signal, signed=False)
    data.create_variable("w1", values=w1[:, None, None, None], units="wn")
    data.create_variable("w2", values=w2[None, :, None, None], units="wn")
    data.create_variable("w3", values=w3[None, None, :, None], units="wn")
    data.create_variable("d1", values=tau[None, None, None, :], units="ps")

    data.transform("w1", "w2", "w3", "d1")

if __name__ == "__main__":
    # objects = wt.artists.interact2D(data)
    import matplotlib.pyplot as plt

    plt.close("all")
    objects = wt.artists.interact2D(data, xaxis=0, yaxis=1)
    plt.show()
