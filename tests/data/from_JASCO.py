"""test from_JASCO"""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_PbSe_batch_1():
    p = datasets.JASCO.PbSe_batch_1
    data = wt.data.from_JASCO(p)
    assert data.shape == (1801,)
    assert data.axis_names == ['wm']


def test_PbSe_batch_4_2012_02_21():
    p = datasets.JASCO.PbSe_batch_4_2012_02_21
    data = wt.data.from_JASCO(p)
    assert data.shape == (1251,)
    assert data.axis_names == ['wm']


def test_PbSe_batch_4_2012_03_15():
    p = datasets.JASCO.PbSe_batch_4_2012_03_15
    data = wt.data.from_JASCO(p)
    assert data.shape == (1251,)
    assert data.axis_names == ['wm']
