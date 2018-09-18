"""Test kit.smooth_1D."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_basic_smoothing_functionality():
    # create arrays
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    np.random.seed(seed=12)
    r = np.random.rand(1000) - .5
    yr = y + r
    # iterate through window types
    windows = ["flat", "hanning", "hamming", "bartlett", "blackman"]
    for w in windows:
        out = wt.kit.smooth_1D(yr, n=101, smooth_type=w)
        check_arr = out - y
        check_arr = check_arr[50:-50]  # get rid of edge effects
        assert np.allclose(check_arr, 0, rtol=.2, atol=.2)


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_basic_smoothing_functionality()
