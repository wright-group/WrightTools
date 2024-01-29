"""Test basic instantiation and handling."""

# --- import --------------------------------------------------------------------------------------


import os
import pytest
import h5py

import WrightTools as wt
from WrightTools import datasets


# --- test ----------------------------------------------------------------------------------------


def test_named_root_collection():
    c = wt.Collection(name="blaise")
    assert c.natural_name == "blaise"
    assert c.attrs["name"] == "blaise"
    assert c.name == "/"
    c.natural_name = "kyle"
    assert c.natural_name == "kyle"
    assert c.attrs["name"] == "kyle"
    assert c.name == "/"


def test_named_root_data():
    d = wt.Data(name="blaise")
    assert d.natural_name == "blaise"
    assert d.attrs["name"] == "blaise"
    assert d.name == "/"
    d.natural_name = "kyle"
    assert d.natural_name == "kyle"
    assert d.attrs["name"] == "kyle"
    assert d.name == "/"


def test_parent_child():
    parent = wt.Collection(name="mother")
    child = wt.Collection(parent=parent, name="goose")
    grandchild = wt.Collection(parent=child, name="hen")
    assert child.filepath == parent.filepath
    assert child.parent is parent
    assert grandchild.parent is child
    assert grandchild.fullpath in wt._group.Group._instances.keys()
    assert child.fullpath in wt._group.Group._instances.keys()
    assert parent.fullpath in wt._group.Group._instances.keys()
    child.natural_name = "duck"
    assert grandchild.fullpath.endswith("/duck/hen")
    assert grandchild.fullpath in wt._group.Group._instances.keys()


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


def test_nested():
    c = wt.Collection()
    cc = c.create_collection()
    assert c.fid.id == cc.fid.id
    c.close()
    assert not os.path.isfile(c.filepath)
    assert c.id.valid == 0
    assert cc.id.valid == 0


def test_open_context():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    with wt.open(p) as d:
        assert os.path.isfile(d.filepath)
    assert not os.path.isfile(d.filepath)
    assert d.id.valid == 0


def test_open_readonly():
    p = datasets.wt5.v1p0p1_MoS2_TrEE_movie
    f = h5py.File(p, "r")
    d = wt.Data(f)
    assert d.file.mode == "r"
    d.close()


def test_close():
    d = wt.Data()
    path = d.filepath
    wt.close()
    assert not os.path.isfile(path)
    with pytest.raises(ValueError):
        d.file


if __name__ == "__main__":
    test_named_root_collection()
    test_named_root_data()
    test_parent_child()
    test_single_instance_collection()
    test_single_instance_data()
    test_tempfile_cleanup()
    test_nested()
    test_open_context()
    test_close()
