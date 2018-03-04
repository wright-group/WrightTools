"""Test from_spcm."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_test_data():
    p = datasets.spcm.test_data
    data = wt.data.from_spcm(p)
    assert data.shape == (1024,)
    assert data.axis_names == ('time',)
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_test_data()
