"""Test chop."""


# --- import -------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- tests --------------------------------------------------------------------------------------


def test_2D_to_1D():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    chop = data.chop('w2')
    assert len(chop) == 81
    for d in chop.values():
        assert d.w2.size == 81
    data.close()
    chop.close()


def test_3D_to_1D():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    chop = data.chop('w2')
    assert len(chop) == 385
    for d in chop.values():
        assert d.w2.size == 11
    data.close()
    chop.close()


def test_3D_to_2D():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    chop = data.chop('wm', 'w2')
    assert len(chop) == 11
    for d in chop.values():
        assert d.wm.size == 29
        assert d.w2.size == 11
    data.close()
    chop.close()