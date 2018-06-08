"""Test dataset min method."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_JASCO_PbSe_batch_4_2012_02_21():
    p = datasets.JASCO.PbSe_batch_4_2012_02_21
    data = wt.data.from_JASCO(p)
    assert data.signal.argmin() == (11,)
    assert data["energy"].argmin() == (1250,)
    data.close()


def test_PyCMDS_wm_w2_w1_000():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data["wm"].argmin() == (0, 0, 0)
    assert data["w2"].argmin() == (0, 10, 0)
    assert data["w1"].argmin() == (0, 0, 0)
    assert data.signal_diff.argmin() == (0, 2, 0)
    assert data.signal_mean.argmin() == (0, 1, 3)
    data.close()
