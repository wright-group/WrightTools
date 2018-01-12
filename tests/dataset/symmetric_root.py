"""Test symmetric root."""


# --- import --------------------------------------------------------------------------------------


import random

import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_d1_d2():
    p = datasets.COLORS.v0p2_d1_d2_diagonal
    data = wt.data.from_COLORS(p)
    root = random.randint(1, 10)
    original = data.ai0[:]
    data.ai0.symmetric_root(root)
    assert np.array_equal(np.sign(data.ai0[:]), np.sign(original))
    assert original.max() ** (1 / root) == data.ai0.max()
    data.close()


def test_w1():
    p = datasets.PyCMDS.w1_000
    data = wt.data.from_PyCMDS(p)
    root = random.randint(1, 10)
    original = data.signal[:]
    data.signal.symmetric_root(root)
    assert np.array_equal(np.sign(data.signal[:]), np.sign(original))
    assert original.max() ** (1 / root) == data.signal.max()
    data.close()


def test_w1_wa():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    root = random.randint(1, 10)
    original = data.array_signal[:]
    data.array_signal.symmetric_root(root)
    assert np.array_equal(np.sign(data.array_signal[:]), np.sign(original))
    assert original.max() ** (1 / root) == data.array_signal.max()
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_d1_d2()
    test_w1()
    test_w1_wa()
