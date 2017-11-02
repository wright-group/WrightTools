"""test from_BrunoldrRaman"""


# --- import --------------------------------------------------------------------------------------


import pytest

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_LDS821_514nm_80mW():
    p = datasets.BrunoldrRaman.LDS821_514nm_80mW
    data = wt.data.from_BrunoldrRaman(p)
    import inspect
    data.attrs['test'] = inspect.stack()[0][3]
    assert data.shape == (1340,)
    assert data.axis_names == ['wm']
    data.file.flush()
    data.close()
