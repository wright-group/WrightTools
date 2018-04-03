#! /usr/bin/env python3
"""Test share_nans."""


# --- import -------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_5():
    arrs = [np.random.random(5) for _ in range(12)]
    arrs[2][2] = np.nan
    arrs = wt.kit.share_nans(*arrs)
    for arr in arrs:
        assert np.isnan(arr[2])


def test_broadcast():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, np.nan]])

    ao, bo = wt.kit.share_nans(a, b)

    assert np.allclose(ao, np.array([[1, np.nan], [3, np.nan]]), equal_nan=True)
    assert np.allclose(bo, np.array([[5, np.nan], [5, np.nan]]), equal_nan=True)

    bo, ao = wt.kit.share_nans(b, a)

    assert np.allclose(ao, np.array([[1, np.nan], [3, np.nan]]), equal_nan=True)
    assert np.allclose(bo, np.array([[5, np.nan], [5, np.nan]]), equal_nan=True)


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_5()
    test_broadcast()
