"""Test units."""


# --- import -------------------------------------------------------------------------------------

import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ---------------------------------------------------------------------------------------


def test_axis_convert_exception():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    try:
        data.w2.convert("fs")
    except wt.exceptions.UnitsError:
        assert True
    else:
        assert False


def test_in_mm_conversion():
    assert np.isclose(wt.units.convert(25.4, "mm", "in"), 1.0)
    assert np.isclose(wt.units.convert(1.0, "in", "mm"), 25.4)
