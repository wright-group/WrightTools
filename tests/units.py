"""Test units."""


# --- import -------------------------------------------------------------------------------------

import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ---------------------------------------------------------------------------------------


def test_axis_convert_exception():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    try:
        data.w2.convert("fs")
    except wt.exceptions.UnitsError:
        assert True
    else:
        assert False


def test_in_mm_conversion():
    assert np.isclose(wt.units.convert(25.4, "mm", "in"), 1.0)
    assert np.isclose(wt.units.convert(1.0, "in", "mm"), 25.4)


def unit_registry_test1():
    values = np.linspace(-1, 1, 51)
    d = wt.Data(name="test")
    try:
        d.create_variable("Bgood", values=values, units="tesla")
        d.transform("Bgood")
    except ValueError:
        assert False
    else:
        assert True


def unit_registry_test2():
    values = np.linspace(-1, 1, 51)
    d = wt.Data(name="test")
    try:
        d.create_variable("Bbad", values=values, units="Tesla")
        d.transform("Bbad")
    except ValueError:
        assert True
    else:
        assert False
