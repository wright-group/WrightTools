#! /usr/bin/env python3
"""Test join."""


# --- import --------------------------------------------------------------------------------------


import posixpath

import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_wm_w2_w1():
    col = wt.Collection()
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    p = datasets.PyCMDS.wm_w2_w1_001
    b = wt.data.from_PyCMDS(p)
    joined = wt.data.join([a, b], parent=col, name='join')
    assert posixpath.basename(joined.name) == 'join'
    assert joined.natural_name == 'join'
    assert joined.shape == (63, 11, 11)
    assert joined.d2.shape == (1, 1, 1)
    assert not np.isnan(joined.channels[0][:]).any()
    assert joined['w1'].label == '1'
    joined.print_tree(verbose=True)
    a.close()
    b.close()
    joined.close()


if __name__ == '__main__':
    test_wm_w2_w1()
