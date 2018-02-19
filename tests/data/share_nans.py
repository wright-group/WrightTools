#! /usr/bin/env python3
"""Test share_nans."""

import WrightTools as wt
import numpy as np
from WrightTools import datasets


def test_share_nans():
    p = wt.datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.channels[0][3] = np.nan
    data.share_nans()
    for ch in data.channels:
        assert np.all(np.isnan(ch[3]))


if __name__ == '__main__':
    test_share_nans()
