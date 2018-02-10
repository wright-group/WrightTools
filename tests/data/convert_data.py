#! /usr/bin/env python3
"""Test data unit conversion."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- define --------------------------------------------------------------------------------------


def test_convert_variables():
    p = datasets.KENT.LDS821_TRSF
    ignore = ['d1', 'd2']
    data = wt.data.from_KENT(p, ignore=ignore)
    data.convert('meV', convert_variables=True)
    assert data.w1.units == 'meV'
    assert data.w2.units == 'meV'
    assert data['w2'].units == 'meV'
    assert data['w2'].units == 'meV'
    # tests that 'inactive' variable is converted
    assert data['wm'].units == 'meV'
    data.close()


def test_w1_wa():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    assert data.wa.units == 'nm'
    data.convert('eV')
    assert data.wa.units == 'eV'
    assert np.isclose(data.wa.max(), 1.5802564757220569)
    assert np.isclose(data.wa.min(), 0.6726385958618104)
    assert data['wa'].units == 'nm'
    data.close()


def test_wigner():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    data.convert('ns')
    assert data.d1.units == 'ns'
    assert data['d1'].units == 'fs'
    assert data.wm.units == 'nm'
    assert data['wm'].units == 'nm'
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_convert_variables()
    test_w1_wa()
    test_wigner()
