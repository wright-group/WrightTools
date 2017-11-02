"""test from_COLORS"""


# --- import --------------------------------------------------------------------------------------


import pytest

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


@pytest.mark.skip()
def test_v0p2_d1_d2_diagonal():
    p = datasets.COLORS.v0p2_d1_d2_diagonal
    data = wt.data.from_COLORS(p)
    assert data.shape == (21, 21)
    assert data.axis_names == ['d1', 'd2']
    data.close()


@pytest.mark.skip()
def test_v0p2_d1_d2_off_diagonal():
    p = datasets.COLORS.v0p2_d1_d2_off_diagonal
    data = wt.data.from_COLORS(p)
    assert data.shape == (21, 21)
    assert data.axis_names == ['d1', 'd2']
    data.close()


@pytest.mark.skip()
def test_v2p1_MoS2_TrEE_movie():
    ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
    data = wt.data.from_COLORS(ps)
    assert data.shape == (41, 41, 23)
    assert data.axis_names == ['w2', 'w1', 'd2']
    data.close()
