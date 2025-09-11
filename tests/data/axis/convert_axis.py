"""Test axis unit conversion."""

# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- define --------------------------------------------------------------------------------------


def test_convert_variables():
    p = datasets.KENT.LDS821_TRSF
    ignore = ["d1", "d2", "wm"]
    data = wt.data.from_KENT(p, ignore=ignore)
    data.w2.convert("meV", convert_variables=True)
    assert data.w2.units == "meV"
    assert data["w2"].units == "meV"
    data.close()


def test_units_preserved_on_copy():
    d1 = wt.Data()
    d1.create_variable(name="color", values=np.array([1]), units="nm")
    d1.create_variable(name="length", values=np.array([1]), units="nm")
    d1.transform("color", "length")

    d1.color.convert("wn")
    d2 = d1.copy()
    assert d2.units == d1.units
    assert d2.units == ("wn", "nm")
    d2.close()
    d1.close()


def test_exception():
    p = datasets.PyCMDS.w1_000
    data = wt.data.from_PyCMDS(p)
    try:
        data.w1.convert("fs")
    except wt.exceptions.UnitsError:
        assert True
    else:
        assert False
    assert data.w1.units == "wn"
    assert data["w1"].units == "nm"
    data.close()


def test_w1_wm():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    data = wt.open(p).at(d2=[0, "fs"])
    assert data.w2.units == "nm"
    data.w2.convert("eV")
    assert data.w2.units == "eV"
    assert np.isclose(data.w2.max(), 2.120129779494837)
    assert np.isclose(data.w2.min(), 1.6241930110774845)
    assert data["w2"].units == "nm"
    data.close()


def test_wigner():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    data = wt.open(p).at(w2=[680, "nm"])
    data.d2.convert("ns")
    assert data.d2.units == "ns"
    assert data["d2"].units == "fs"
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_convert_variables()
    test_units_preserved_on_copy()
    test_exception()
    test_w1_wm()
    test_wigner()
