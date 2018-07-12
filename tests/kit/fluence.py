"""Test fluence."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_0():
    out = wt.kit.fluence(1, 2, .1, 1000, 1, 'eV', 'cm', 'ps_t')
    assert isinstance(out[0], float)
