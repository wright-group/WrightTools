"""test from_JASCO"""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_wm_ypos_fluorescence_with_filter():
    p = datasets.Solis.wm_ypos_fluorescence_with_filter
    data = wt.data.from_Solis(p)
    assert data.shape == (2560, 2160)
    assert data.axis_expressions == ("wm", "yindex")
    assert data.units == ("nm", None)
    data.close()


def test_xpos_ypos_fluorescence():
    p = datasets.Solis.xpos_ypos_fluorescence
    data = wt.data.from_Solis(p)
    assert data.shape == (2560, 2160)
    assert data.axis_expressions == ("xindex", "yindex")
    assert data.units == (None, None)
    data.close()


if __name__ == "__main__":
    test_wm_ypos_fluorescence_with_filter()
    test_xpos_ypos_fluorescence()
