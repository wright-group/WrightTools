"""test from_KENT"""


# --- import --------------------------------------------------------------------------------------


import pytest

import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- helper functions ----------------------------------------------------------------------------


def isclose(p, data, ignore=[], ss=(slice(None))):
    if isinstance(p, list):
        for i, f in enumerate(p):
            dat = np.genfromtxt(f).T
            if i == 0:
                arr = dat
            else:
                arr = np.append(arr, dat, axis=1)
    else:
        arr = np.genfromtxt(p).T
    for i, key in enumerate(['w1', 'w2', 'wm', 'd1', 'd2', 'signal', 'OPA1', 'OPA2']):
        if key in ignore:
            continue
        raw = arr[i]
        raw.shape = data.shape
        assert np.isclose(raw.all(), data[key].full[ss].all())


# --- test ----------------------------------------------------------------------------------------


def test_LDS821_TRSF():
    p = datasets.KENT.LDS821_TRSF
    ignore = ['wm', 'd1', 'd2']
    data = wt.data.from_KENT(p, ignore=ignore)
    assert data.shape == (71, 71)
    assert data.axis_names == ('w2', 'w1',)
    assert data.units == ('wn', 'wn')
    ss = (slice(None, None, -1), slice(None, None, None))
    isclose(p, data, ignore=ignore, ss=ss)
    data.close()


def test_PbSe_2D_delay_B():
    p = datasets.KENT.PbSe_2D_delay_B
    data = wt.data.from_KENT(p, delay_tolerance=0.01)
    assert data.shape == (101, 101)
    assert data.axis_names == ('d2', 'd1',)
    assert data.units == ('ps', 'ps')
    ss = (slice(None, None, None), slice(None, None, -1))
    isclose(p, data, ss=ss, ignore=['d1', 'd2'])
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_LDS821_TRSF()
    test_PbSe_2D_delay_B()
