"""Test dataset min method."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_COLORS_v2p2_WL_wigner():
    p = wt.datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    assert np.isclose(data['wm'].min(), 416.6946)
    assert np.isclose(data['d1'].min(), -500.12859918527766)
    assert np.isclose(data.ai0.min(), -1.3343566958049833)
    data.close()


def test_JASCO_PbSe_batch_4_2012_02_21():
    p = wt.datasets.JASCO.PbSe_batch_4_2012_02_21
    data = wt.data.from_JASCO(p)
    assert np.isclose(data.signal.min(), 0.02875)
    assert np.isclose(data['energy'].min(), 750.0)
    data.close()


def test_PyCMDS_wm_w2_w1_000():
    p = wt.datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert np.isclose(data['wm'].min(), 483.0833)
    assert np.isclose(data['w2'].min(), 1550.0)
    assert np.isclose(data['w1'].min(), 6250.0)
    assert np.isclose(data.signal_diff.min(), 0.000773)
    assert np.isclose(data.signal_mean.min(), 0.000563)
    data.close()
