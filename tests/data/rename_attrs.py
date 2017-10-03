"""test rename_attrs."""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_rename():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    data.rename_attrs(w1='w2', w2='w1')
    assert data.shape == (35, 11, 11)
    assert data.axis_names == ['wm', 'w1', 'w2']


def test_error():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    try:
        data.rename_attrs(w1='w2')
    except wt.exceptions.NameNotUniqueError:
        assert True
    else:
        assert False
