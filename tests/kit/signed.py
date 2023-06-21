"""Test signed data."""


# --- import -------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_5():
    arr = np.array([-1, 0, 1])
    assert wt.kit.signed(arr) == True


def test_5_multiple():
    arr = np.array([1, 3, 4, 11, 12])
    assert wt.kit.signed(arr) == False
