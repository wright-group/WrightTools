"""test from_JASCO"""

# --- import --------------------------------------------------------------------------------------


import pathlib

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_PbSe_batch_1():
    p = datasets.JASCO.PbSe_batch_1
    data = wt.data.from_JASCO(p)
    assert data.shape == (1801,)
    assert data.axis_expressions == ("energy",)
    assert data.units == ("nm",)
    data.close()


def test_PbSe_batch_4_2012_02_21():
    p = datasets.JASCO.PbSe_batch_4_2012_02_21
    data = wt.data.from_JASCO(p)
    assert data.shape == (1251,)
    assert data.axis_expressions == ("energy",)
    assert data.units == ("nm",)
    data.close()


def test_PbSe_batch_4_2012_03_15():
    p = datasets.JASCO.PbSe_batch_4_2012_03_15
    p = pathlib.Path(p)
    data = wt.data.from_JASCO(p)
    assert data.shape == (1251,)
    assert data.axis_expressions == ("energy",)
    assert data.units == ("nm",)
    data.close()


def test_remote():
    p = "https://osf.io/download/hzsjp/"
    data = wt.data.from_JASCO(p)
    assert data.shape == (851,)
    assert data.axis_expressions == ("energy",)
    assert data.units == ("nm",)
