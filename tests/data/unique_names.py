"""Test that names are kept unique in data."""


# --- import -------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_exception():
    d = wt.Data()
    points = np.linspace(0, 1, 51)
    d.create_axis(name='w1', points=points, units='eV')
    try:
        d.create_channel(name='w1') 
    except RuntimeError:
        assert True
    else:
        assert False