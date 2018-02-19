#! /usr/bin/env python3
"""Test get_zenith."""

import WrightTools as wt
import numpy as np
from WrightTools import datasets


def test_zenith():
    p = wt.datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert np.allclose(list(data.get_zenith()), [499.9893, 1575., 6349.206349])


if __name__ == '__main__':
    test_zenith()
