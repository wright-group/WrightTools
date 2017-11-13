"""Test nm_width."""


# --- import -------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_float():
    assert np.isclose(wt.kit.nm_width(1300, 100), 592.593, atol=0.1)
