"""test from_KENT"""


# --- import --------------------------------------------------------------------------------------


import pytest

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_LDS821_TRSF():
    p = datasets.KENT.LDS821_TRSF
    ignore = ['wm', 'd1', 'd2']
    data = wt.data.from_KENT(p, ignore=ignore)
    assert data.shape == (71, 71)
    assert data.axis_names == ('w2', 'w1',)
    assert data.units == ('wn', 'wn')
    data.close()


@pytest.mark.skip(reason='unknown problems with data itself')
def test_PbSe_2D_delay_A():
    p = datasets.KENT.PbSe_2D_delay_A
    data = wt.data.from_KENT(p, delay_tolerance=0.01)
    assert data.shape == (101, 151)
    assert data.axis_names == ('d2', 'd1',)
    assert data.units == ('ps', 'ps')
    data.close()


def test_PbSe_2D_delay_B():
    p = datasets.KENT.PbSe_2D_delay_B
    data = wt.data.from_KENT(p, delay_tolerance=0.01)
    assert data.shape == (101, 101)
    assert data.axis_names == ('d2', 'd1',)
    assert data.units == ('ps', 'ps')
    data.close()
