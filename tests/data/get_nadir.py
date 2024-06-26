#! /usr/bin/env python3
"""Test get_nadir."""

import WrightTools as wt
import numpy as np
from WrightTools import datasets


def test_nadir():
    p = wt.datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.wm.convert("nm")
    assert np.allclose(list(data.get_nadir()), [483.0833, 1590.0, 1600.0])


if __name__ == "__main__":
    test_nadir()
