#! /usr/bin/env python3
"""Test group copy and save."""


# --- import --------------------------------------------------------------------------------------


import os
import pathlib
import warnings

import numpy as np

import WrightTools as wt
from WrightTools import datasets


# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


# --- helpers -------------------------------------------------------------------------------------


def assert_equal(a, b):
    if hasattr(a, "__iter__"):
        if hasattr(a, "dtype") and a.ndim > 1:
            assert np.isclose(a[:].all(), b[:].all())
        else:
            for ai, bi in zip(a, b):
                assert ai == bi
    else:
        assert a == b


# --- test ----------------------------------------------------------------------------------------


def test_copy_data():
    root = wt.Collection()
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p, parent=root, name="data")
    new = data.copy(parent=root, name="copy")
    assert len(root.item_names) == 2
    for k, v in data.attrs.items():
        if k == "name":
            continue
        assert_equal(new.attrs[k], v)
    for k, v in data.items():
        assert_equal(new[k], v)
    for axis in new.axes:
        assert getattr(new, axis.natural_name) is axis
    for channel in new.channels:
        assert getattr(new, channel.natural_name) is channel
    data.close()
    new.close()


def test_save_data():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    data.create_constant("w3")
    p = os.path.join(here, "data")
    p = data.save(p)
    assert os.path.isfile(p)
    assert p.endswith(".wt5")
    new = wt.open(p)
    for k, v in data.attrs.items():
        assert_equal(new.attrs[k], v)
    for k, v in data.items():
        assert_equal(new[k], v)
    for axis in new.axes:
        assert getattr(new, axis.natural_name) is axis
    for channel in new.channels:
        assert getattr(new, channel.natural_name) is channel
    assert new.constant_names == ("w3",)
    data.close()
    new.close()
    os.remove(p)


def test_save_nested():
    root = wt.Collection(name="root")
    one = wt.Collection(name="one", parent=root)
    wt.Collection(name="two", parent=one)
    p = os.path.join(here, "nested")
    p = one.save(p)
    assert os.path.isfile(p)
    assert p.endswith(".wt5")
    new = wt.open(p)
    for k, v in one.attrs.items():
        assert_equal(new.attrs[k], v)
    for k, v in one.items():
        assert_equal(new[k], v)
    root.close()
    new.close()
    os.remove(p)


def test_simple_copy():
    original = wt.Collection(name="blaise")
    new = original.copy()
    assert original.fullpath != new.fullpath
    for k, v in original.attrs.items():
        print(k)
        assert_equal(new.attrs[k], v)
    for k, v in original.items():
        print(k)
        assert_equal(new[k], v)
    original.close()
    new.close()


def test_simple_save():
    original = wt.Collection(name="blaise")
    p = os.path.join(here, "simple")
    p = original.save(p)
    assert os.path.isfile(p)
    assert p.endswith(".wt5")
    new = wt.open(p)
    for k, v in original.attrs.items():
        assert_equal(new.attrs[k], v)
    for k, v in original.items():
        assert_equal(new[k], v)
    original.close()
    new.close()
    os.remove(p)


def test_propagate_units():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    data.convert("eV")
    p = os.path.join(here, "units")
    p = data.save(p)
    new = wt.open(p)
    assert new.units == ("eV", "eV", "eV")
    data.close()
    new.close()
    os.remove(p)


def test_propagate_expressions():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)
    data.transform("w1-wm", "w2", "w1")
    p = os.path.join(here, "expressions")
    p = data.save(p)
    new = wt.open(p)
    assert data.axis_expressions == new.axis_expressions
    data.close()
    new.close()
    os.remove(p)


def test_save_pathlib():
    p = datasets.PyCMDS.wm_w2_w1_001
    data = wt.data.from_PyCMDS(p)

    p = pathlib.Path(__file__).parent / "pathlib"
    p = data.save(p)
    p = pathlib.Path(p)
    assert p.exists()
    assert p.suffix == ".wt5"
    with warnings.catch_warnings(record=True) as w:
        new = wt.open(p)
        assert len(w) == 0
    for k, v in data.attrs.items():
        assert_equal(new.attrs[k], v)
    for k, v in data.items():
        assert_equal(new[k], v)
    for axis in new.axes:
        assert getattr(new, axis.natural_name) is axis
    for channel in new.channels:
        assert getattr(new, channel.natural_name) is channel
    data.close()
    new.close()
    p.unlink()


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_copy_data()
    test_save_data()
    test_save_nested()
    test_simple_copy()
    test_simple_save()
    test_propagate_units()
    test_propagate_expressions()
    test_save_pathlib()
