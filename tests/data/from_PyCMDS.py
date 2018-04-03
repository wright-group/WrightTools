#! /usr/bin/env python3
"""test from_PyCMDS"""


# --- import --------------------------------------------------------------------------------------


import os
import numpy as np

import WrightTools as wt
from WrightTools import datasets

here = os.path.abspath(os.path.dirname(__file__))


# --- test ----------------------------------------------------------------------------------------


def test_w1_000():
    p = datasets.PyCMDS.w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (51,)
    assert data.axis_expressions == ('w1',)
    data.close()


def test_w1_wa_000():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (25, 256)
    assert data.axis_expressions == ('w1=wm', 'wa',)
    data.close()


def test_w2_w1_000():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (81, 81)
    assert data.axis_expressions == ('w2', 'w1',)
    data.close()


def test_wm_w2_w1_000():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (35, 11, 11)
    assert data.axis_expressions == ('wm', 'w2', 'w1',)
    data.close()


def test_wm_w2_w1_001():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (29, 11, 11)
    assert data.axis_expressions == ('wm', 'w2', 'w1',)
    data.close()


def test_incomplete():
    p = os.path.join(here, 'test_data', 'incomplete.data')
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (9, 9)
    assert data.axis_expressions == ('d1', 'd2')
    assert np.allclose(data.d1.points, np.array([-1., -1.125, -1.25, -1.375, -1.5,
                                                -1.625, -1.75, -1.875, -2.]))
    data.close()


def test_ps_delay():
    p = os.path.join(here, 'test_data', 'ps_delay.data')
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (11, 15, 15)
    assert data.axis_expressions == ('d1', 'w2', 'w1')
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_w1_000()
    test_w1_wa_000()
    test_w2_w1_000()
    test_wm_w2_w1_000()
    test_wm_w2_w1_001()
    test_incomplete()
    test_ps_delay()
