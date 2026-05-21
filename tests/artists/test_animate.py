"""smokescreen tests for animation functions"""

import WrightTools as wt
from WrightTools import datasets
from matplotlib import pyplot as plt
import logging

logging.basicConfig(level="INFO")


def test_animate2D():
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)
    d.channels[0].signed = True
    from matplotlib.colors import CenteredNorm
    from functools import partial

    ani = wt.artists.animate2D(
        d, interval=50, back_and_forth=True, norm=partial(CenteredNorm, vcenter=0)
    )
    return ani


def test_animate_interact2D():
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)
    d.channels[0].signed = True

    out = wt.artists.interact2D(d, local=True)
    ani = wt.artists.animate_interact2D(out, back_and_forth=True, interval=500)
    return ani


def test_animate_quick2D():
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)
    d.channels[0].signed = True

    quick2D = wt.artists.quick2Ds(d, autosave=True)

    ani = wt.artists.animate_quick(quick2D, interval=100)
    return ani


def test_animate_quick1D():
    d = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie).at(d2=[0, "fs"])
    quick_iter = wt.artists.quick1Ds(d, autosave=False, local=False)

    ani = wt.artists.animate_quick(quick_iter, interval=100)
    return ani


if __name__ == "__main__":
    _unused_ani1 = test_animate2D()
    _unused_ani2 = test_animate_interact2D()
    _unused_ani3 = test_animate_quick2D()
    _unused_ani4 = test_animate_quick1D()
    plt.show()
