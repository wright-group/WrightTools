"""test from_Tensor27"""


# --- import --------------------------------------------------------------------------------------


import pytest

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_CuPCtS_powder_ATR():
    p = datasets.Tensor27.CuPCtS_powder_ATR
    data = wt.data.from_Tensor27(p)
    import inspect
    data.attrs['test'] = inspect.stack()[0][3]
    assert data.shape == (7259,)
    assert data.axis_names == ['wm']
    data.file.flush()
    data.close()
