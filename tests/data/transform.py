"""Test transform."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- tests ---------------------------------------------------------------------------------------


def test_w2_w1():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.transform('w2', 'wm')
    assert data.axis_names == ('w2', 'wm',)
    assert data.w2.shape == (81, 1)
    assert data.wm.shape == (81, 81)
