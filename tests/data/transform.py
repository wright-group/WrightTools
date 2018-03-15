"""Test transform."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- tests ---------------------------------------------------------------------------------------


def test_w2_w1():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.transform('w2', 'wm')
    assert data.axis_names == ('w2', 'wm',)
    assert data.w2.shape == (81, 1)
    assert data.wm.shape == (81, 81)
    data.close()


def test_w2_w1_multiply():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.transform('3*w2', '0.5*wm')
    assert data.axis_names == ('_3__t__w2', '_0_5__t__wm',)
    assert np.allclose(data._3__t__w2[:], data['w2'][:] * 3)
    assert np.allclose(data._0_5__t__wm[:], data['wm'][:] * 0.5)
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_w2_w1()
    test_w2_w1_multiply()
