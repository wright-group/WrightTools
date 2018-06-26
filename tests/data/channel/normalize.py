"""Test channel normalize."""


# --- import --------------------------------------------------------------------------------------


import random

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_LDS821():
    p = datasets.BrunoldrRaman.LDS821_514nm_80mW
    data = wt.data.from_BrunoldrRaman(p)
    data.signal.normalize()
    assert data.signal.null == 0.
    assert data.signal.max() == 1.
    data.close()


def test_LDS821_mag_random():
    p = datasets.BrunoldrRaman.LDS821_514nm_80mW
    data = wt.data.from_BrunoldrRaman(p)
    mag = random.random() + 0.5
    data.signal.normalize(mag)
    assert data.signal.null == 0.
    assert data.signal.max() == mag
    assert data.signal.mag() == mag
    data.close()


def test_wigner():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    data.ai0.normalize()
    assert data.ai0.null == 0.
    assert data.ai0.max() == 1.
    data.close()


def test_negative_wigner():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    data.ai0 *= -1
    data.ai0.signed = True
    data.ai0.normalize()
    assert data.ai0.null == 0.
    assert data.ai0.min() == -1.
    assert data.ai0.mag() == 1.
    data.close()


def test_negative_wigner_mag_1p1():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    data.ai0 *= -1
    data.ai0.signed = True
    data.ai0.normalize(1.1)
    assert data.ai0.null == 0.
    assert data.ai0.min() == -1.1
    assert data.ai0.mag() == 1.1
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_LDS821()
    test_LDS821_mag_random()
    test_wigner()
    test_negative_wigner()
    test_negative_wigner_mag_1p1()
