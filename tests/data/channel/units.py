"""Tests to do with null."""

# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
import numpy as np


def test_assignment():
    d = wt.Data()
    d.create_channel("x", values=np.arange(5), units="Hz")
    assert d["x"].units == "Hz"


if __name__ == "__main__":
    test_assignment()