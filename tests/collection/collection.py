#! /usr/bin/env python3
import WrightTools as wt
import pytest


def test_create_collection():
    col = wt.Collection()
    child1 = col.create_collection()
    with pytest.warns(wt.exceptions.ObjectExistsWarning):
        child2 = col.create_collection()
    child3 = col.create_collection("path/to/collection")
    assert child3.natural_name == "collection"
    assert child3.parent.natural_name == "to"
    assert "path" in col.item_names
    assert isinstance(child3.parent, wt.Collection)
    assert child1 == child2
    assert child1.natural_name == "collection"


def test_create_data():
    col = wt.Collection()
    child1 = col.create_data()
    with pytest.warns(wt.exceptions.ObjectExistsWarning):
        child2 = col.create_data()
    child3 = col.create_data("path/to/data")
    assert child3.natural_name == "data"
    assert child3.parent.natural_name == "to"
    assert "path" in col.item_names
    assert isinstance(child3.parent, wt.Collection)
    assert child1 == child2
    assert child1.natural_name == "data"


if __name__ == "__main__":
    test_create_collection()
    test_create_data()
