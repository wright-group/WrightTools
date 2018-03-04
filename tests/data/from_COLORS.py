"""test from_COLORS"""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_v0p2_d1_d2_diagonal():
    p = datasets.COLORS.v0p2_d1_d2_diagonal
    data = wt.data.from_COLORS(p)
    assert data.shape == (21, 21)
    assert data.axis_expressions == ('d1', 'd2',)
    assert data.units == ('fs', 'fs')
    data.close()


def test_v0p2_d1_d2_off_diagonal():
    p = datasets.COLORS.v0p2_d1_d2_off_diagonal
    data = wt.data.from_COLORS(p)
    assert data.shape == (21, 21)
    assert data.axis_expressions == ('d1', 'd2',)
    assert data.units == ('fs', 'fs')
    data.close()


def test_v2p1_MoS2_TrEE_movie():
    ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
    data = wt.data.from_COLORS(ps)
    assert data.shape == (41, 41, 23)
    assert data.axis_expressions == ('w2', 'w1=wm', 'd2',)
    assert data.units == ('nm', 'nm', 'fs')
    data.close()


def test_v2p2_WL_wigner():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    assert data.shape == (241, 51)
    assert data.axis_expressions == ('wm', 'd1',)
    assert data.units == ('nm', 'fs')
    data.close()
