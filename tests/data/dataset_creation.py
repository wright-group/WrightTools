#! /usr/bin/env python3

import WrightTools as wt
import numpy as np
import pytest

import os


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


def test_exception():
    d = wt.Data()
    points = np.linspace(0, 1, 51)
    d.create_variable(name="w1", points=points, units="eV")
    with pytest.raises(wt.exceptions.NameNotUniqueError):
        d.create_channel(name="w1")


def test_create_compressed_channel():
    data = wt.Data()
    child1 = data.create_channel("hi", shape=(1024, 1024), compression="gzip")
    data["hi"][:] = 0
    assert os.path.getsize(data.filepath) < 1e6


if __name__ == "__main__":
    test_create_variable()
    test_create_channel()
    test_exception()
