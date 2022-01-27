"""Test dataset unit conversion."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- define --------------------------------------------------------------------------------------


def test_exception():
    p = datasets.PyCMDS.w1_000
    data = wt.data.from_PyCMDS(p)
    try:
        data["w1"].convert("fs")
    except wt.exceptions.UnitsError:
        assert True
    else:
        assert False
    assert data["w1"].units == "nm"
    data.close()


def test_w1_wa():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    assert data["wa"].units == "nm"
    data["wa"].convert("eV")
    assert np.isclose(data["wa"].max(), 1.5802564757220569)
    assert np.isclose(data["wa"].min(), 0.6726385958618104)
    data.close()


def test_wigner():
    p = datasets.COLORS.v2p2_WL_wigner
    data = wt.data.from_COLORS(p)
    data["d1"].convert("ns")
    assert data["d1"].units == "ns"
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_exception()
    test_w1_wa()
    test_wigner()
