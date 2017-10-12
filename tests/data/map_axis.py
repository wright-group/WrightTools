"""test map_axis"""


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
    data.map_axis(0, new, 'wn')
    assert data.axes[0].points.all() == new.all()


def test_int():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (35, 11, 11)
    data.map_axis(1, 5)
    assert data.shape == (35, 5, 11)
    data.map_axis(2, 25)
    assert data.shape == (35, 5, 25)
