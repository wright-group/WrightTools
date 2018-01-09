"""Test data bring_to_front."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_integer():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.channel_names[0] == 'signal_diff'
    data.bring_to_front(1)
    assert data.channel_names[0] == 'signal_mean'
    data.bring_to_front(1)
    assert data.channel_names[0] == 'signal_diff'
    data.close()


def test_name():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    assert data.channel_names[0] == 'ai0'
    data.bring_to_front('ai3')
    assert data.channel_names[0] == 'ai3'
    data.close()
