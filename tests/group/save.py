"""Test group copy and save."""


# --- import --------------------------------------------------------------------------------------


import os

import WrightTools as wt
from WrightTools import datasets


# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


# --- helpers -------------------------------------------------------------------------------------


def assert_equal(a, b):
    if hasattr(a, '__iter__'):
        for ai, bi in zip(a, b):
            assert ai == bi
    else:
        assert a == b


# --- test ----------------------------------------------------------------------------------------


def test_save_nested():
    root = wt.Collection(name='root')
    one = wt.Collection(name='one', parent=root)
    wt.Collection(name='two', parent=one)
    p = os.path.join(here, 'nested')
    p = one.save(p)
    assert os.path.isfile(p)
    assert p.endswith('.wt5')
    new = wt.open(p)
    for k, v in one.attrs.items():
        assert_equal(new.attrs[k], v)
    for k, v in one.items():
        assert_equal(new[k], v)
    root.close()
    new.close()
    os.remove(p)


def test_simple_copy():
    original = wt.Collection(name='blaise')
    new = original.copy()
    assert original.fullpath != new.fullpath
    for k, v in original.attrs.items():
        assert_equal(new.attrs[k], v)
    for k, v in original.items():
        assert_equal(new[k], v)
    original.close()
    new.close()


def test_simple_save():
    original = wt.Collection(name='blaise')
    p = os.path.join(here, 'simple')
    p = original.save(p)
    assert os.path.isfile(p)
    assert p.endswith('.wt5')
    new = wt.open(p)
    for k, v in original.attrs.items():
        assert_equal(new.attrs[k], v)
    for k, v in original.items():
        assert_equal(new[k], v)
    original.close()
    new.close()
    os.remove(p)
