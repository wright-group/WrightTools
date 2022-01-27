"""Test dataset min method."""


# --- import --------------------------------------------------------------------------------------

import pathlib

import h5py
import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_COLORS_v2p2_WL_wigner():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    arr = np.genfromtxt(p).T
    assert np.isclose(data["wm"].min(), arr[7].min())
    assert np.isclose(data["d1"].min(), arr[12].min(), rtol=0.1)
    assert np.isclose(data.ai0.min(), arr[16].min(), rtol=0.1)
    data.close()


def test_JASCO_PbSe_batch_4_2012_02_21():
    p = datasets.JASCO.PbSe_batch_4_2012_02_21
    data = wt.data.from_JASCO(p)
    assert np.isclose(data.signal.min(), 0.02875)
    assert np.isclose(data["energy"].min(), 750.0)
    data.close()


def test_PyCMDS_wm_w2_w1_000():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert np.isclose(data["wm"].min(), 483.0833)
    assert np.isclose(data["w2"].min(), 1550.0)
    assert np.isclose(data["w1"].min(), 6250.0)
    assert np.isclose(data.signal_diff.min(), 0.000773)
    assert np.isclose(data.signal_mean.min(), 0.000563)
    data.close()


def test_read_only():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    f = h5py.File(p, "r")
    d = wt.Data(f)
    assert np.isclose(d.w2.min(), 584.7953254198521)
    assert "min" not in d.ai0.attrs
    assert np.isclose(d.ai0.min(), -0.008888)
    assert "min" not in d.ai0.attrs
    d.close()


def test_read_only_with_max_cached():
    p = pathlib.Path(__file__).parent / "max_cached.wt5"
    f = h5py.File(p, "r")
    d = wt.Data(f)
    assert "max" in d.x.attrs
    assert np.isclose(d.x.min(), -1)
    assert "min" not in d.x.attrs
    d.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_COLORS_v2p2_WL_wigner()
    test_JASCO_PbSe_batch_4_2012_02_21()
    test_PyCMDS_wm_w2_w1_000()
    test_read_only()
