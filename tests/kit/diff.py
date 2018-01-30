"""Test diff."""


# --- import -------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------------------------------


def test_ascending_1():
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    d = wt.kit.diff(x, y)
    assert np.all((np.abs(d - np.cos(x)) < 0.0001)[:-1])


def test_ascending_2():
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    d = wt.kit.diff(x, y, 2)
    assert np.all((np.abs(d + np.sin(x)) < 0.0001)[1:-2])


def test_ascending_3():
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    d = wt.kit.diff(x, y, 3)
    assert np.all((np.abs(d + np.cos(x)) < 0.0001)[2:-3])


def test_ascending_4():
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    d = wt.kit.diff(x, y, 4)
    assert np.all((np.abs(d - np.sin(x)) < 0.0001)[3:-4])


def test_descending_1():
    x = np.linspace(10, 0, 1000)
    y = np.sin(x)
    d = wt.kit.diff(x, y)
    assert np.all((np.abs(d - np.cos(x)) < 0.0001)[1:-1])


def test_descending_3():
    x = np.linspace(10, 0, 1000)
    y = np.sin(x)
    d = wt.kit.diff(x, y, 3)
    assert np.all((np.abs(d + np.cos(x)) < 0.0001)[3:-3])
