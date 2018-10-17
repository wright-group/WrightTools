#! /usr/bin/env python3


import WrightTools as wt
from WrightTools import datasets
from matplotlib import pyplot as plt
import shutil
import os


def test_pcolor():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    p = shutil.copy(p, "./test_pcolor.wt5")
    data = wt.open(p)
    os.unlink(p)
    data.level(0, 2, -3)
    data.convert("eV")
    data.ai0.symmetric_root(0.5)
    chop = data.chop("w1=wm", "w2", at={"d2": [-600, "fs"]})[0]

    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.pcolor(chop)
    ax.pcolormesh(chop)

    data.close()
    chop.close()


def test_pcolor_lower_rank():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    p = shutil.copy(p, "./test_pcolor_lower_rank.wt5")
    data = wt.open(p)
    os.unlink(p)
    data.level(0, 2, -3)
    data.convert("eV")
    data.ai0.symmetric_root(0.5)
    data.collapse("d2", method="sum")

    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    ax.pcolor(data, channel="ai0_d2_sum")
    ax.pcolormesh(data, channel="ai0_d2_sum")

    data.close()


if __name__ == "__main__":
    test_pcolor()
    test_pcolor_lower_rank()
    plt.show()
