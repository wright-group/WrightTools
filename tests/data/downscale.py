#! /usr/bin/env python3
"""Test downscale."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_downscale():
    p = datasets.Solis.wm_ypos_fluorescence_with_filter
    a = wt.data.from_Solis(p)
    b = a.downscale((3, 10))
    assert b.shape == (854, 216)
    assert b.axis_expressions == a.axis_expressions
    # wt.artists.quick2D(a)
    wt.artists.quick2D(b)
    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    test_downscale()
