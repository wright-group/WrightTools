#! /usr/bin/env python3
"""Test remove_variable."""


# --- import --------------------------------------------------------------------------------------


import warnings

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_exception():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    try:
        data.remove_variable("d1")
    except RuntimeError:
        assert True
    else:
        assert False
    data.close()


def test_constant_warning():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.create_constant("d1")
    with warnings.catch_warnings(record=True) as w:
        data.remove_variable("d1")
        assert len(w) == 1
        assert issubclass(w[-1].category, wt.exceptions.WrightToolsWarning)
    data.close()


def test_implied():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    names = data.variable_names
    data.remove_variable("d0")
    for n in names:
        if n.startswith("d0"):
            assert n not in data.variable_names
    data.close()


def test_not_implied():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    names = data.variable_names
    data.remove_variable("d0", implied=False)
    for n in names:
        if n != "d0":
            assert n in data.variable_names
        else:
            assert n not in data.variable_names
    data.close()


def test_index():
    p = datasets.KENT.LDS821_TRSF
    ignore = ["wm", "d1", "d2"]
    data = wt.data.from_KENT(p, ignore=ignore)
    assert data.variable_names == ("w2", "w1", "wm", "d1", "d2")
    data.remove_variable(-2)
    assert data.variable_names == ("w2", "w1", "wm", "d2")
    data.close()


if __name__ == "__main__":
    test_exception()
    test_implied()
    test_not_implied()
    test_index()
