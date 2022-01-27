"""Test in place division."""


# --- import --------------------------------------------------------------------------------------


import random

import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_d1_d2():
    p = datasets.COLORS.v0p2_d1_d2_diagonal
    data = wt.data.from_COLORS(p)
    value = random.randint(0, 1e5)
    original_max = data.ai0.max()
    original_min = data.ai0.min()
    data.ai0 /= value
    assert data.ai0.max() == original_max / value
    assert data.ai0.min() == original_min / value
    data.close()


def test_d1_d2_array():
    p = datasets.COLORS.v0p2_d1_d2_diagonal
    data = wt.data.from_COLORS(p)
    value = np.random.random(data.shape)
    original = data.ai0[:]
    data.ai0 /= value
    assert np.array_equal(data.ai0, original / value)
    data.close()


def test_w1():
    p = datasets.PyCMDS.w1_000
    data = wt.data.from_PyCMDS(p)
    value = random.randint(0, 1e5)
    original_max = data.signal.max()
    original_min = data.signal.min()
    data.signal /= value
    assert data.signal.max() == original_max / value
    assert data.signal.min() == original_min / value
    data.close()


def test_w1_array():
    p = datasets.PyCMDS.w1_000
    data = wt.data.from_PyCMDS(p)
    value = np.random.random(data.shape)
    original = data.signal[:]
    data.signal /= value
    assert np.array_equal(data.signal, original / value)
    data.close()


def test_w1_wa():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    value = random.randint(0, 1e5)
    original_max = data.array_signal.max()
    original_min = data.array_signal.min()
    data.array_signal /= value
    assert data.array_signal.max() == original_max / value
    assert data.array_signal.min() == original_min / value
    data.close()


def test_w1_wa_array():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    value = np.random.random((256,))
    original = data.array_signal[:]
    data.array_signal /= value
    assert np.array_equal(data.array_signal, original / value)
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_d1_d2()
    test_d1_d2_array()
    test_w1()
    test_w1_array()
    test_w1_wa()
    test_w1_wa_array()
