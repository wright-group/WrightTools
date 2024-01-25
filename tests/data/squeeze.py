#! /usr/bin/env python3
"""Test squeeze."""


# --- import -------------------------------------------------------------------------------------


import numpy as np
import WrightTools as wt
from WrightTools import datasets


# --- tests --------------------------------------------------------------------------------------


def test_squeeze():
    d = wt.Data(name="test")
    d.create_variable("x", values=np.arange(5)[:, None, None])
    d.create_variable("y", values=np.arange(4)[None, :, None])
    d.create_variable("redundant_array", values=np.tile(np.arange(3), (5, 4, 1)))

    d.create_channel("keep", values=d.x[:] + d.y[:])
    d.create_channel("throw_away", values=np.zeros((5, 4, 3)))

    d.transform("x", "y")
    d = d.squeeze()  # make sure it runs error free
    assert d.ndim == 2
    assert d.shape == (5, 4)

def test_constants():
    d = wt.Data(name="test")
    d.create_variable("x", values=np.array([1]).reshape(1,1))
    d.create_constant("x")
    d.create_variable("y", values=np.linspace(3,5,4).reshape(-1,1))
    d.create_variable("z", values=np.linspace(0,1,6).reshape(1,-1))
    d.transform("y")
    ds = d.squeeze()
    assert "x" in ds.constant_expressions
    d.print_tree()
    ds.print_tree()


if __name__ == "__main__":
    test_squeeze()
    test_constants()
