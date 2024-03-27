#! /usr/bin/env python3
"""
Test ichop.

Many tests of `chop` also validate ichop functionality, so tests are less
encompasing here and focus on iterator behavior.
"""


# --- import -------------------------------------------------------------------------------------


import WrightTools as wt
from WrightTools import datasets
import pathlib


# --- tests --------------------------------------------------------------------------------------


def test_2D_to_1D():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    for i, d in enumerate(data.ichop("w2")):
        assert d.w2.size == 81
        assert d.axis_expressions == ("w2",)
        assert d.constant_expressions == ("w1",)
        for k in data.variable_names:
            assert d[k].label == data[k].label
    assert i == data.w1.size - 1
    data.close()


def test_3D_to_1D():
    p = datasets.PyCMDS.wm_w2_w1_000
    data = wt.data.from_PyCMDS(p)
    for i, d in enumerate(data.ichop("w2")):
        assert d.w2.size == 11
        assert d.axis_expressions == ("w2",)
    assert i == (data.wm.size * data.w1.size) - 1
    data.close()


def test_3D_to_2D():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    for i, d in enumerate(data.ichop("wm", "w2")):
        assert d.wm.size == 29
        assert d.w2.size == 11
        assert d.axis_expressions == ("wm", "w2")
    assert i == data.w1.size - 1
    data.close()


def test_3D_to_2D_at():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    ichop = data.ichop("wm", "w2", at={"w1": [1590, "wn"]})
    d = ichop.__next__()
    assert d.wm.size == 29
    assert d.w2.size == 11
    assert d.axis_expressions == ("wm", "w2")

    # no next because only one data object
    try:
        d = ichop.__next__()
    except StopIteration:
        pass
    else:
        raise AssertionError
    data.close()


def test_closing_context():
    """
    the iterator uses the close method on the last iteration,
    so we shouldn't have persistent files
    """
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    ichop = data.ichop("wm", "w2")
    d1 = ichop.__next__()
    assert pathlib.Path(d1.filepath).exists()
    d2 = ichop.__next__()
    assert not pathlib.Path(d1.filepath).exists()
    d2.close()
    ichop = data.ichop("wm", "w2", autoclose=False)
    d1 = ichop.__next__()
    d2 = ichop.__next__()
    assert pathlib.Path(d1.filepath).exists()
    d1.close()
    d2.close()
    data.close()


def test_filtering():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    data.print_tree()
    mean = data.channels[0][:].mean()
    is_interesting = lambda d: d.channels[0][:].mean() > mean
    interesting = {
        f"{i}": d
        for i, d in enumerate(filter(is_interesting, data.ichop("w1", "w2", autoclose=False)))
    }
    assert len(interesting) == 19


def test_collection_like():
    p = datasets.PyCMDS.w2_w1_000
    data = wt.data.from_PyCMDS(p)
    collection = {f"chop{i:0>3}": d for i, d in enumerate(data.ichop("w2", autoclose=False))}
    assert len(collection) == data.w1.size
    collection["chop002"].__repr__()
    wt.close()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_2D_to_1D()
    test_3D_to_1D()
    test_3D_to_2D()
    test_3D_to_2D_at()
    test_closing_context()
    test_collection_like()
    test_filtering()
