"""Test from_ocean_optics."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_tsunami_scope():
    p = datasets.ocean_optics.tsunami
    data = wt.data.from_ocean_optics(p)
    assert data.axis_names == ('energy',)
    assert data.shape == (2048,)
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_tsunami_scope()
