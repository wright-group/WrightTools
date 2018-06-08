"""Test dataset max method."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_COLORS_v2p2_WL_wigner():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    arr = np.genfromtxt(p).T
    assert np.isclose(data["wm"].max(), arr[7].max())
    assert np.isclose(data["d1"].max(), arr[12].max(), rtol=0.1)
    assert np.isclose(data.ai0.max(), arr[16].max(), rtol=0.1)
    data.close()


def test_JASCO_PbSe_batch_4_2012_02_21():
    p = datasets.JASCO.PbSe_batch_4_2012_02_21
    data = wt.data.from_JASCO(p)
    assert np.isclose(data.signal.max(), 0.78552)
    assert np.isclose(data["energy"].max(), 2000.0)
    data.close()


def test_PyCMDS_wm_w2_w1_000():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert np.isclose(data["wm"].max(), 526.3259)
    assert np.isclose(data["w2"].max(), 1600.0)
    assert np.isclose(data["w1"].max(), 6451.612903)
    assert np.isclose(data.signal_diff.max(), 0.264612)
    assert np.isclose(data.signal_mean.max(), 0.07345)
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_COLORS_v2p2_WL_wigner()
    test_JASCO_PbSe_batch_4_2012_02_21()
    test_PyCMDS_wm_w2_w1_000()
