"""Test transform."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- tests ---------------------------------------------------------------------------------------


def test_add():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.transform('w1+w2', 'wm')
    assert data.axis_names == ('w1__p__w2', 'wm',)
    assert data.w1__p__w2.shape == (81, 81)
    assert data.wm.shape == (81, 81)
    data.close()


def test_multiply_by_constant():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.transform('3*w2', '0.5*wm')
    assert data.axis_names == ('_3__t__w2', '_0_5__t__wm',)
    assert np.allclose(data._3__t__w2[:], data['w2'][:] * 3)
    assert np.allclose(data._0_5__t__wm[:], data['wm'][:] * 0.5)
    data.close()


def test_simple():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.transform('w2', 'wm')
    assert data.axis_names == ('w2', 'wm',)
    assert data.w2.shape == (81, 1)
    assert data.wm.shape == (81, 81)
    data.close()


def test_subtract():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.transform('w1-w2', 'wm')
    assert data.axis_names == ('w1__m__w2', 'wm',)
    assert np.allclose(data.w1__m__w2[:], data['w1'][:] - data['w2'][:])
    assert np.allclose(data.wm[:], data['wm'][:])
    data.close()


def test_subtract_constant():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.transform('w2-7000', 'wm')
    assert data.axis_names == ('w2__m__7000', 'wm',)
    assert np.allclose(data.w2__m__7000[:], data['w2'][:] - 7000.)
    assert np.allclose(data.wm[:], data['wm'][:])
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_add()
    test_multiply_by_constant()
    test_simple()
    test_subtract()
    test_subtract_constant()
