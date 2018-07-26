import numpy as np
import WrightTools as wt
from WrightTools import datasets
import os

here = os.path.dirname(__file__)

if False:
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)

if True:
    p = datasets.wt5.v1p0p0_perovskite_TA
    data = wt.open(p)

if False:
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    data = wt.open(p)
    data.convert("eV")

if False:
    w1 = np.linspace(-5, 5, 31)
    w2 = w1.copy()
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
    plt.close('all')
    objects = wt.artists.interact2D(data, xaxis=1, yaxis=0)
    plt.show()
