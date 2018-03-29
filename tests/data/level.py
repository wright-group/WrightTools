#! /usr/bin/env python3
"""Test level."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets

import numpy as np


# --- test ----------------------------------------------------------------------------------------


def test_1D():
    p = datasets.Cary.CuPCtS_H2O_vis
    data = wt.collection.from_Cary(p)[0]
    data.level(0, 0, 5)
    assert np.isclose(data.abs[:5].mean(), 0.)
    data.close()


def test_2D():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    data.level('ai0', 1, -3)
    print(data.ai0[:, -3:].max())
    assert np.allclose(data.ai0[:, -3:], [0.], atol=5)  # very noisy data
    data.close()


def test_3D():
    p = datasets.COLORS.v2p1_MoS2_TrEE_movie
    data = wt.data.from_COLORS(p)
    data.level('ai0', 1, 1)
    assert np.allclose(data.ai0[:, :1], [0.], atol=1e-3)
    data.close()


def test_channels():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    data_copy = data.copy()
    data.bring_to_front(1)
    data.level(0, 0, 4)
    data_copy.level(1, 0, 4)
    a = data.channels[0][:]
    b = data_copy.channels[1][:]
    assert np.allclose(a, b)
    data.close()
    data_copy.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_1D()
    test_2D()
    test_3D()
    test_channels()
