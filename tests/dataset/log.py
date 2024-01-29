"""Test log."""

# --- import --------------------------------------------------------------------------------------


import random

import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_log():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    target = np.log(data.array_signal[:])
    data.array_signal.log()
    assert np.array_equal(data.array_signal[:], target)
    data.close()


def test_log_floor():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    floor = random.random() * -5
    data.array_signal.log(floor=floor)
    assert data.array_signal.min() >= floor
    data.close()


def test_log10():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    target = np.log10(data.array_signal[:])
    data.array_signal.log10()
    assert np.array_equal(data.array_signal[:], target)
    data.close()


def test_log10_floor():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    floor = random.random() * -3
    data.array_signal.log10(floor=floor)
    assert data.array_signal.min() >= floor
    data.close()


def test_log2():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    target = np.log2(data.array_signal[:])
    data.array_signal.log2()
    assert np.array_equal(data.array_signal[:], target)
    data.close()


def test_log2_floor():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    floor = random.random() * -6
    data.array_signal.log2(floor=floor)
    assert data.array_signal.min() >= floor
    data.close()


def test_log_strange_base():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    base = random.random()
    target = np.log(data.array_signal[:]) / np.log(base)
    data.array_signal.log(base=base)
    assert np.array_equal(data.array_signal[:], target)
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_log()
    test_log_floor()
    test_log10()
    test_log10_floor()
    test_log2()
    test_log2_floor()
    test_log_strange_base()
