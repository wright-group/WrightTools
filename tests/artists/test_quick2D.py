#! /usr/bin/env python3

import numpy as np
import pathlib
import WrightTools as wt
from WrightTools import datasets




def test_save_arguments():
    p = datasets.wt5.v1p0p0_perovskite_TA  # axes w1=wm, w2, d2
    # A race condition exists where multiple tests access the same file in short order
    # this loop will open the file when it becomes available.
    while True:
        try:
            data = wt.open(p)
            break
        except:
            pass
    handler = wt.artists._quick._quick2D(data, 0, 2, autosave=True)
    assert handler.nD == 2
    assert handler.nfigs == 52
    assert handler.save_directory.parent == pathlib.Path.cwd()
    assert handler.filepath_seed == "{0:0>3}.png"
    handler = wt.artists._quick._quick2D(
        data, 0, 2, autosave=True, save_directory="some_filepath", fname="test"
    )
    assert handler.save_directory.parent == pathlib.Path("some_filepath")
    assert handler.filepath_seed == "test {0:0>3}.png"


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
    handler = wt.artists._quick._quick2D(data, xaxis=0, yaxis=2, at={"w2": [1.7, "eV"]})
    assert handler.nD == 2
    assert handler.autosave == False
    handler(True)


def test_4D():
    w1 = np.linspace(-3, 3, 3)
    w2 = np.linspace(-2, 2, 3)
    w3 = np.linspace(-1, 1, 3)
    tau = np.linspace(-1, 3, 2)
    signal = (
        w1[:, None, None, None]
        + w2[None, :, None, None]
        + w3[None, None, :, None]
        + tau[None, None, None, :]
    )
    data = wt.data.Data(name="data")
    data.create_channel("signal", values=signal, signed=True)
    data.create_variable("w1", values=w1[:, None, None, None], units="wn", label="1")
    data.create_variable("w2", values=w2[None, :, None, None], units="wn", label="2")
    data.create_variable("w3", values=w3[None, None, :, None], units="wn", label="3")
    data.create_variable("d1", values=tau[None, None, None, :], units="ps")
    data.transform("w1", "w2", "w3", "d1")
    wt.artists.quick2D(data, xaxis=0, yaxis=1)


def test_remove_uninvolved_dimensions():
    w1 = np.linspace(-2, 2, 51)
    w2 = np.linspace(-1, 1, 11)
    w3 = np.linspace(1, 3, 5)
    signal = np.cos(w1[:, None, None] + w2[None, :, None] + w3[None, None, :])
    data = wt.data.Data(name="data")
    data.create_channel("signal", values=signal, signed=True)
    data.create_variable("w1", values=w1[:, None, None], units="wn", label="1")
    data.create_variable("w2", values=w2[None, :, None], units="wn", label="2")
    data.create_variable("w3", values=w3[None, None, :], units="wn", label="3")
    data.transform("w1", "w2", "w3")
    data.moment(2, 0, 0)
    # moments bug(?): moment is not signed, even though the data is
    handler = wt.artists._quick._quick2D(data, channel=-1)
    handler(True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")
    test_save_arguments()
    test_perovskite()
    test_4D()
    test_remove_uninvolved_dimensions()
    plt.show()
