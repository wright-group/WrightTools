"""Test units."""


# --- import -------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ---------------------------------------------------------------------------------------


def test_axis_convert_exception():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    try:
        data.w2.convert('fs')
    except wt.exceptions.UnitsError:
        assert True
