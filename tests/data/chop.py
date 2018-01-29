#! /usr/bin/env python3
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
        assert d.axis_expressions == ('w2',)
        for k in data.variable_names:
            assert d[k].label == data[k].label
    data.close()
    chop.close()


def test_3D_to_1D():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    chop = data.chop('w2')
    assert len(chop) == 385
    for d in chop.values():
        assert d.w2.size == 11
        assert d.axis_expressions == ('w2',)
    data.close()
    chop.close()


def test_3D_to_1D_at():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    chop = data.chop('w2', at={'w1': [1590, 'wn']})
    assert len(chop) == 29
    for d in chop.values():
        assert d.w2.size == 11
        assert d.axis_expressions == ('w2',)
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
        assert d.axis_expressions == ('wm', 'w2',)
    data.close()
    chop.close()


def test_3D_to_2D_at():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    chop = data.chop('wm', 'w2', at={'w1': [1590, 'wn']})
    assert len(chop) == 1
    assert chop[0].wm.size == 29
    assert chop[0].w2.size == 11
    assert chop[0].axis_expressions == ('wm', 'w2',)
    data.close()
    chop.close()


def test_3D_to_2D_units():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    data.convert('eV')
    chop = data.chop('wm', 'w2')
    assert len(chop) == 11
    for d in chop.values():
        assert d.wm.size == 29
        assert d.w2.size == 11
        assert d.axis_expressions == ('wm', 'w2',)
        assert d.units == ('eV', 'eV')
    data.close()
    chop.close()


def test_parent():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    parent = wt.Collection()
    chop = data.chop('wm', 'w2', parent=parent)
    assert chop.parent is parent


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_2D_to_1D()
    test_3D_to_1D()
    test_3D_to_2D_units()
