#! /usr/bin/env python3
"""Test at."""


# --- import -------------------------------------------------------------------------------------


import numpy as np
import WrightTools as wt
from WrightTools import datasets


# --- tests --------------------------------------------------------------------------------------


def test_3D_to_1D():
    data = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)
    sliced = data.at(d2=[-50, "fs"], w2=[700, "nm"])
    assert sliced.axis_expressions == ("w1=wm",)
    sliced = data.at(w1__e__wm=[605, "nm"], w2=[700, "nm"])
    assert sliced.axis_expressions == ("d2",)
    data.close()
    sliced.close()


if __name__ == "__main__":
    test_3D_to_1D()
