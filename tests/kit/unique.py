"""Test unique."""


# --- import -------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_5():
    arr = np.array([0, 1, 2, 3, 3 + 1e-7])
    assert np.isclose(wt.kit.unique(arr).all(), np.array([0, 1, 2, 3.000005]).all())


def test_5_tolerance():
    arr = np.array([0, 1, 2, 3, 3 + 1e-7])
    assert wt.kit.unique(arr, tolerance=1e-8).all() == arr.all()
