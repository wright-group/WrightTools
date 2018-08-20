#! /usr/bin/env python3
"""Test join."""


# --- import --------------------------------------------------------------------------------------


import posixpath

import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_wm_w2_w1():
    col = wt.Collection()
    p = datasets.PyCMDS.wm_w2_w1_000
    a = wt.data.from_PyCMDS(p)
    p = datasets.PyCMDS.wm_w2_w1_001
    b = wt.data.from_PyCMDS(p)
    joined = wt.data.join([a, b], parent=col, name="join")
    assert posixpath.basename(joined.name) == "join"
    assert joined.natural_name == "join"
    assert joined.shape == (63, 11, 11)
    assert joined.d2.shape == (1, 1, 1)
    assert not np.isnan(joined.channels[0][:]).any()
    assert joined["w1"].label == "1"
    joined.print_tree(verbose=True)
    a.close()
    b.close()
    joined.close()


def test_1D_no_overlap():
    a = wt.Data()
    b = wt.Data()

    a.create_variable("x", np.linspace(0, 10, 11))
    b.create_variable("x", np.linspace(11, 21, 11))
    a.transform("x")
    b.transform("x")

    joined = wt.data.join([a, b])

    assert joined.shape == (22,)
    assert np.allclose(joined.x.points, np.linspace(0, 21, 22))

    a.close()
    b.close()
    joined.close()


def test_1D_overlap_identical():
    a = wt.Data()
    b = wt.Data()

    a.create_variable("x", np.linspace(0, 10, 11))
    b.create_variable("x", np.linspace(5, 15, 11))
    a.transform("x")
    b.transform("x")

    joined = wt.data.join([a, b])

    assert joined.shape == (16,)
    assert np.allclose(joined.x.points, np.linspace(0, 15, 16))

    a.close()
    b.close()
    joined.close()


def test_1D_overlap_offset():
    a = wt.Data()
    b = wt.Data()

    a.create_variable("x", np.linspace(0, 10, 11))
    b.create_variable("x", np.linspace(5.5, 15.5, 11))
    a.transform("x")
    b.transform("x")

    joined = wt.data.join([a, b])

    assert joined.shape == (22,)
    assert np.allclose(
        joined.x.points,
        np.sort(np.concatenate([np.linspace(0, 10, 11), np.linspace(5.5, 15.5, 11)])),
    )

    a.close()
    b.close()


def test_2D_no_overlap_aligned():
    pass


def test_2D_no_overlap_offset():
    pass


def test_2D_overlap_identical():
    pass


def test_2D_overlap_offset():
    pass


def test_2D_some_same_some_offset():
    pass


def test_1D_to_2D_aligned():
    pass


def test_1D_to_2D_not_aligned():
    pass


def test_2D_plus_1D():
    pass


def test_3D_no_overlap_aligned():
    pass


def test_3D_no_overlap_offset():
    pass


def test_3D_overlap_identical():
    pass


def test_3D_overlap_offset():
    pass


def test_3D_plus_2D():
    pass


def test_1D_plus_2D_plus_3D():
    pass


def test_overlap_first():
    pass


def test_overlap_last():
    pass


def test_overlap_sum():
    pass


def test_overlap_max():
    pass


def test_overlap_min():
    pass


def test_overlap_mean():
    pass


if __name__ == "__main__":
    test_wm_w2_w1()
    test_1D_no_overlap()
    test_1D_overlap_identical()
    test_1D_overlap_offset()
    test_2D_no_overlap_aligned()
    test_2D_no_overlap_offset()
    test_2D_overlap_identical()
    test_2D_overlap_offset()
    test_2D_some_same_some_offset()
    test_1D_to_2D_aligned()
    test_1D_to_2D_not_aligned()
    test_2D_plus_1D()
    test_3D_no_overlap_aligned()
    test_3D_no_overlap_offset()
    test_3D_overlap_identical()
    test_3D_overlap_offset()
    test_3D_plus_2D()
    test_1D_plus_2D_plus_3D()
    test_overlap_first()
    test_overlap_last()
    test_overlap_sum()
    test_overlap_max()
    test_overlap_min()
    test_overlap_mean()
