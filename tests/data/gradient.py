#! /usr/bin/env python3
"""Test collapse."""


# --- import -------------------------------------------------------------------------------------


import WrightTools as wt
import numpy as np


# --- tests --------------------------------------------------------------------------------------


def test_gradient():
    data = wt.Data()
    data.create_variable("v1", np.arange(0, 6))
    data.create_variable("v2", np.arange(0, 6) * 2)
    data.create_channel("ch", np.array([1, 2, 4, 7, 11, 16]))
    data.transform("v1")
    data.gradient("v1")
    data.transform("v2")
    data.gradient("v2")

    assert data.ch_v1_gradient.shape == (6,)
    assert np.allclose(data.ch_v1_gradient.points, np.array([1.0, 1.5, 2.5, 3.5, 4.5, 5.0]))
    assert data.ch_v2_gradient.shape == (6,)
    assert np.allclose(data.ch_v2_gradient.points, np.array([0.5, 0.75, 1.25, 1.75, 2.25, 2.5]))


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_gradient()
