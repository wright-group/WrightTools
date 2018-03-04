"""Test map_variable."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_array():
    p = datasets.JASCO.PbSe_batch_1
    data = wt.data.from_JASCO(p)
    assert data.shape == (1801,)
    new = np.linspace(6000, 8000, 55)
    mapped = data.map_variable('energy', new, 'wn')
    assert data.axes[0][:].all() == new.all()
    data.close()


def test_int():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (35, 11, 11)
    mapped = data.map_variable('w2', 5)
    assert mapped.shape == (35, 5, 11)
    mapped = data.map_variable('w1', 25)
    assert mapped.shape == (35, 11, 25)
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_array()
    test_int()
