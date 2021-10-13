"""Test units."""


# --- import -------------------------------------------------------------------------------------

import numpy as np
import pytest

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


def test_unit_registry():
    values = np.linspace(-1, 1, 51)
    d = wt.Data(name="test")
    d.create_variable("Bgood", values=values, units="tesla")
    d.transform("Bgood")


def test_bad_unit_registry():
    values = np.linspace(-1, 1, 51)
    d = wt.Data(name="test")
    with pytest.raises(ValueError):
        d.create_variable("Bbad", values=values, units="Tesla")
        d.transform("Bbad")


def test_0_inf():
    assert wt.units.convert(0, "wn", "nm") == np.inf
    assert wt.units.convert(0, "nm", "wn") == np.inf
