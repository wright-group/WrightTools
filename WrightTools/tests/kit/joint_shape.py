"""Test joint_shape."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_5x7():
    arr1 = np.empty((5, 1))
    arr2 = np.empty((1, 7))
    assert wt.kit.joint_shape(arr1, arr2) == (5, 7)


def test_3x4x5():
    arr1 = np.empty((1, 4, 1))
    arr2 = np.empty((3, 1, 5))
    arr3 = np.empty((1, 1, 1))
    assert wt.kit.joint_shape(arr1, arr2, arr3) == (3, 4, 5)
