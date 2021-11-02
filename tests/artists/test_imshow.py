#! /usr/bin/env python3


import WrightTools as wt
from WrightTools import datasets
from matplotlib import pyplot as plt
import shutil
import os


def test_imshow():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    p = shutil.copy(p, "./test_imshow.wt5")
    data = wt.open(p)
    os.unlink(p)
    data.level(0, 2, -3)
    data.convert("eV")
    data.ai0.symmetric_root(2)
    chop = data.chop("w1=wm", "w2", at={"d2": [-600, "fs"]})[0]
    # chop = chop.split("w2", [2])[0]

    fig, gs = wt.artists.create_figure(cols=[1, 1])
    ax = plt.subplot(gs[0])
    ax.pcolormesh(chop)

    ax1 = plt.subplot(gs[1])
    ax1.imshow(chop)

    data.close()
    chop.close()


if __name__ == "__main__":
    test_imshow()
    plt.show()
