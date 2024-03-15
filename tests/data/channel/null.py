"""Tests to do with null."""

# --- import --------------------------------------------------------------------------------------


import WrightTools as wt

from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_setter():
    p = datasets.BrunoldrRaman.LDS821_514nm_80mW
    data = wt.data.from_BrunoldrRaman(p)
    assert data.signal.null == 0
    data.signal.null = 5
    assert data.signal.null == 5
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_setter()
