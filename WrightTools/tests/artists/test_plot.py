#! /usr/bin/env python3


import WrightTools as wt
from WrightTools import datasets
from matplotlib import pyplot as plt
import shutil
import os


def test_plot():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    p = shutil.copy(p, "./test_plot.wt5")
    data = wt.open(p)
    os.unlink(p)
    data.level(0, 2, -3)
    data.convert("eV")
    data.ai0.symmetric_root(2)
    chop = data.chop("w1=wm", at={"d2": [-600, "fs"], "w2": [2.0, "eV"]})[0]

    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.plot(chop)

    data.close()
    chop.close()


def test_plot_lower_rank():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    p = shutil.copy(p, "./test_plot_lower_rank.wt5")
    data = wt.open(p)
    os.unlink(p)
    data.transform("w1", "w2", "d2")
    data.level(0, 2, -3)
    data.convert("eV")
    data.ai0.symmetric_root(2)
    data.collapse("d2", method="sum")
    data.collapse("w2", method="sum")

    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.plot(data, channel="ai0_d2_sum_w2_sum")

    data.close()


if __name__ == "__main__":
    test_plot()
    test_plot_lower_rank()
    plt.show()
