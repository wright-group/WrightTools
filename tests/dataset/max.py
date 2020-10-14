"""Test dataset max method."""


# --- import --------------------------------------------------------------------------------------


import h5py
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


def test_read_only():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    f = h5py.File(p, "r")
    d = wt.Data(f)
    assert np.isclose(d.w2.max(), 763.3587731728356)
    assert "max" not in d.ai0.attrs
    assert np.isclose(d.ai0.max(), 0.2560301283785622)
    assert "max" not in d.ai0.attrs
    d.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_COLORS_v2p2_WL_wigner()
    test_JASCO_PbSe_batch_4_2012_02_21()
    test_PyCMDS_wm_w2_w1_000()
    test_read_only()
