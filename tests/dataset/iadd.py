"""Test in place addition."""


# --- import --------------------------------------------------------------------------------------


import random

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_d1_d2():
    p = datasets.COLORS.v0p2_d1_d2_diagonal
    data = wt.data.from_COLORS(p)
    value = random.randint(-1e5, 1e5)
    original_max = data.ai0.max()
    original_min = data.ai0.min()
    data.ai0 += value
    assert data.ai0.max() == original_max + value
    assert data.ai0.min() == original_min + value
    data.close()


def test_w1():
    p = datasets.PyCMDS.w1_000
    data = wt.data.from_PyCMDS(p)
    value = random.randint(-1e5, 1e5)
    original_max = data.signal.max()
    original_min = data.signal.min()
    data.signal += value
    assert data.signal.max() == original_max + value
    assert data.signal.min() == original_min + value
    data.close()


def test_w1_wa():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    value = random.randint(-1e5, 1e5)
    original_max = data.array_signal.max()
    original_min = data.array_signal.min()
    data.array_signal += value
    assert data.array_signal.max() == original_max + value
    assert data.array_signal.min() == original_min + value
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_d1_d2()
    test_w1()
    test_w1_wa()
