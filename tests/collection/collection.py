#! /usr/bin/env python3
import WrightTools as wt
import pytest


def test_create_collection():
    col = wt.Collection()
    child1 = col.create_collection()
    with pytest.warns(wt.exceptions.ObjectExistsWarning):
        child2 = col.create_collection()
    assert child1 == child2
    assert child1.natural_name == "collection"


def test_create_data():
    col = wt.Collection()
    child1 = col.create_data()
    with pytest.warns(wt.exceptions.ObjectExistsWarning):
        child2 = col.create_data()
    assert child1 == child2
    assert child1.natural_name == "data"


if __name__ == "__main__":
    test_create_collection()
    test_create_data()
