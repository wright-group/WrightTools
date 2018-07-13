"""Test fluence."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_0():
    out = wt.kit.fluence(1, 2, .1, 1000, 1, "eV", "cm", "ps_t")
    assert np.isclose(31.8309, out[0], rtol=1e-3)
