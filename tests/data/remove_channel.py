"""Test remove channel."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_remove_first_channel():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.remove_channel('signal_diff')
    assert data.channel_names[0] == 'signal_mean'
    data.close()


def test_remove_trailing_channels():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    while len(data.channel_names) > 1:
        data.remove_channel(-1)
    assert data.channel_names == ('ai0',)
    data.close()
