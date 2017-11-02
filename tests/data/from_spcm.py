"""test from_spcm"""


# --- import --------------------------------------------------------------------------------------


import pytest

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_test_data():
    p = datasets.spcm.test_data
    data = wt.data.from_spcm(p)
    import inspect
    data.attrs['test'] = inspect.stack()[0][3]
    print(inspect.stack()[0][3], data.filepath)
    assert data.shape == (1024,)
    assert data.axis_names == ['time']
    data.file.flush()
    data.close()
