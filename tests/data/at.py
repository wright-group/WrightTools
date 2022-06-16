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


def test_chop_equivalence():
    data = wt.open(datasets.wt5.v1p0p1_MoS2_TrEE_movie)
    at_data = data.at(d2=[-50, "fs"], w2=[700, "nm"])
    chop_data = data.chop("w1=wm", at={"d2":[-50, "fs"], "w2":[700, "nm"]})[0]
    assert at_data.shape == chop_data.shape
    assert at_data.axis_expressions == chop_data.axis_expressions


if __name__ == "__main__":
    test_3D_to_1D()
    test_chop_equivalence()
