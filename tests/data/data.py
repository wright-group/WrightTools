#! /usr/bin/env python3

import WrightTools as wt
import pytest


def test_create_variable():
    data = wt.Data()
    child1 = data.create_variable("hi", shape=(10, 10))
    with pytest.warns(wt.exceptions.ObjectExistsWarning):
        child2 = data.create_variable("hi", shape=(10, 1))
    assert child1 == child2
    assert child2.shape == (10, 10)


def test_create_channel():
    data = wt.Data()
    child1 = data.create_channel("hi", shape=(10, 10))
    with pytest.warns(wt.exceptions.ObjectExistsWarning):
        child2 = data.create_channel("hi", shape=(10, 1))
    assert child1 == child2
    assert child2.shape == (10, 10)


if __name__ == "__main__":
    test_create_variable()
    test_create_channel()
