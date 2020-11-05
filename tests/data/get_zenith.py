#! /usr/bin/env python3
"""Test get_zenith."""

import WrightTools as wt
import numpy as np
from WrightTools import datasets


def test_zenith():
    p = wt.datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.wm.convert("nm")
    data.w1.convert("nm")
    assert np.allclose(list(data.get_zenith()), [495.0566, 1560.0, 6369.426752])


if __name__ == "__main__":
    test_zenith()
