"""test from_shimadzu"""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_MoS2_fromCzech2015():
    p = datasets.Shimadzu.MoS2_fromCzech2015
    data = wt.data.from_shimadzu(p)
    assert data.shape == (819,)
    assert data.axis_expressions == ('energy',)
    assert data.units == ('nm',)
    data.close()
