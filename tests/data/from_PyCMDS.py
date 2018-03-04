"""test from_PyCMDS"""


# --- import --------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_w1_000():
    p = datasets.PyCMDS.w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (51,)
    assert data.axis_expressions == ('w1',)
    data.close()


def test_w1_wa_000():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (25, 256)
    assert data.axis_expressions == ('w1=wm', 'wa',)
    data.close()


def test_w2_w1_000():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (81, 81)
    assert data.axis_expressions == ('w2', 'w1',)
    data.close()


def test_wm_w2_w1_000():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (35, 11, 11)
    assert data.axis_expressions == ('wm', 'w2', 'w1',)
    data.close()


def test_wm_w2_w1_001():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    assert data.shape == (29, 11, 11)
    assert data.axis_expressions == ('wm', 'w2', 'w1',)
    data.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_w1_000()
    test_w1_wa_000()
    test_w2_w1_000()
    test_wm_w2_w1_000()
    test_wm_w2_w1_001()
