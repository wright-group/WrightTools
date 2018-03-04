"""test from_Tensor27"""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_CuPCtS_powder_ATR():
    p = datasets.Tensor27.CuPCtS_powder_ATR
    data = wt.data.from_Tensor27(p)
    assert data.shape == (7259,)
    assert data.axis_names == ('energy',)
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_CuPCtS_powder_ATR()
