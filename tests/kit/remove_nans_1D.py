#! /usr/bin/env python3
"""Test remove_nans_1D."""


# --- import -------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_simple():
    arr = np.arange(-4, 6, dtype=float)
    arr[arr < 0] = np.nan
    assert wt.kit.remove_nans_1D(arr)[0].all() == np.arange(0, 6, dtype=float).all()


def test_multiple():
    arrs = [np.random.random(21) for _ in range(5)]
    arrs[0][0] = np.nan
    arrs[1][-1] = np.nan
    arrs = wt.kit.remove_nans_1D(*arrs)
    for arr in arrs:
        assert arr.size == 19


def test_list():
    assert np.all(wt.kit.remove_nans_1D([np.nan, 1, 2, 3])[0] == np.array([1, 2, 3]))


if __name__ == "__main__":
    test_simple()
    test_multiple()
    test_list()
