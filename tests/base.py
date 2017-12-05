"""Test basic instantiation and handling."""


# --- import --------------------------------------------------------------------------------------


import os

import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_named_root_collection():
    c = wt.Collection(name='blaise')
    assert c.natural_name == 'blaise'
    assert c.attrs['name'] == 'blaise'


def test_named_root_data():
    d = wt.Data(name='blaise')
    assert d.natural_name == 'blaise'
    assert d.attrs['name'] == 'blaise'


def test_parent_child():
    parent = wt.Collection(name='mother')
    child = wt.Collection(parent=parent, name='goose')
    assert child.filepath == parent.filepath
    assert child.parent is parent


def test_single_instance_collection():
    c1 = wt.Collection()
    c2 = wt.Collection(filepath=c1.filepath, edit_local=True)
    assert c1 is c2


def test_single_instance_data():
    d1 = wt.Data()
    d2 = wt.Data(filepath=d1.filepath, edit_local=True)
    assert d1 is d2


def test_tempfile_cleanup():
    c = wt.Collection()
    path = c.filepath
    assert os.path.isfile(path)
    c.close()
    assert not os.path.isfile(path)
