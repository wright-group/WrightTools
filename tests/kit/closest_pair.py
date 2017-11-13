"""Test closest pair."""


# --- import -------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_5():
    arr = np.array([1, 3, 4, 6, 12])
    assert wt.kit.closest_pair(arr) == [[(1,), (2,)]]
    assert wt.kit.closest_pair(arr, 'distance') == 1


def test_5_multiple():
    arr = np.array([1, 3, 4, 11, 12])
    assert wt.kit.closest_pair(arr) == [[(1,), (2,)], [(3,), (4,)]]
    assert wt.kit.closest_pair(arr, 'distance') == 1


def test_example():
    arr = np.array([0, 1, 2, 3, 3, 4, 5, 6, 1])
    assert wt.kit.closest_pair(arr) == [[(1,), (8,)], [(3,), (4,)]]


def test_2x3():
    arr = np.array([[0, 1, 2], [3, 4, 4.5]])
    assert wt.kit.closest_pair(arr) == [[(1, 1), (1, 2)]]
    assert wt.kit.closest_pair(arr, 'distance') == 0.5
