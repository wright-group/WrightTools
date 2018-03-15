"""test from_COLORS"""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- helper functions ----------------------------------------------------------------------------


def isclose(p, data, **kwargs):
    if isinstance(p, list):
        for i, f in enumerate(p):
            dat = np.genfromtxt(f).T
            if i == 0:
                arr = dat
            else:
                arr = np.append(arr, dat, axis=1)
    else:
        arr = np.genfromtxt(p).T
    for key, index in kwargs.items():
        raw = arr[index]
        raw.shape = data.shape
        assert np.isclose(raw.all(), data[key].full.all())


# --- test ----------------------------------------------------------------------------------------


def test_v0p2_d1_d2_diagonal():
    p = datasets.COLORS.v0p2_d1_d2_diagonal
    data = wt.data.from_COLORS(p)
    assert data.shape == (21, 21)
    assert data.axis_expressions == ('d1', 'd2',)
    assert data.units == ('fs', 'fs')
    isclose(p, data, d1=6, d2=8, ai0=10, ai1=11, ai2=12)
    data.close()


def test_v2p1_MoS2_TrEE_movie():
    ps = datasets.COLORS.v2p1_MoS2_TrEE_movie
    data = wt.data.from_COLORS(ps)
    assert data.shape == (41, 41, 23)
    assert data.axis_expressions == ('w2', 'w1=wm', 'd2',)
    assert data.units == ('nm', 'nm', 'fs')
    isclose(ps, data, wm=7)
    data.close()


def test_v2p2_WL_wigner():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    assert data.shape == (241, 51)
    assert data.axis_expressions == ('wm', 'd1',)
    assert data.units == ('nm', 'fs')
    isclose(p, data, wm=7, d1=12, ai0=16, ai1=17, ai2=18)
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_v0p2_d1_d2_diagonal()
    test_v2p1_MoS2_TrEE_movie()
    test_v2p2_WL_wigner()
