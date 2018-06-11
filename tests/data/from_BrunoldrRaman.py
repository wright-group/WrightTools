"""test from_BrunoldrRaman"""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_LDS821_514nm_80mW():
    p = datasets.BrunoldrRaman.LDS821_514nm_80mW
    data = wt.data.from_BrunoldrRaman(p)
    assert data.shape == (1340,)
    assert data.axis_expressions == ("energy",)
    assert data.units == ("wn",)
    data.close()
