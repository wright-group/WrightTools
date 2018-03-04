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


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_5()
