"""Test timer."""


# --- import -------------------------------------------------------------------------------------


import time

import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_wait():
    with wt.kit.Timer():
        time.sleep(1)


def test_instance():
    t = wt.kit.Timer(verbose=False)
    with t:
        time.sleep(1)
    assert np.isclose(t.interval, 1, atol=1)
