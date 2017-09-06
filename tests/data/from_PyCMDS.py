"""test from_PyCMDS"""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_wm_w2_w1_000():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (35, 11, 11)
    assert data.axis_names == ['wm', 'w2', 'w1']


def test_wm_w2_w1_001():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (29, 11, 11)
    assert data.axis_names == ['wm', 'w2', 'w1']
