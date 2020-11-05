#! /usr/bin/env python3
"""Test data __init__"""

# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
import numpy as np


# --- test ----------------------------------------------------------------------------------------


def test_axes_NoneType_units():
    a = wt.Data()
    a.create_variable("x", np.array([0]), units=None)
    a.transform("x")
    a = a.copy()
    assert a.x.units is None
    a.close()
