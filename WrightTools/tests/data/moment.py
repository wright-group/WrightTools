#! /usr/bin/env python3
import numpy as np
import WrightTools as wt
from WrightTools import datasets


def test_moment_lognormal():
    data = wt.Data()
    data.create_variable("x", np.linspace(0, 1000, 100000))
    data.transform("x")
    x = data.x[:]
    data.create_channel("y", 1 / (x * np.sqrt(2 * np.pi)) * np.exp(-np.log(x) ** 2 / 2))

    data.moment(0, moment=(0, 1, 2, 3, 4))

    vals = [
        1,
        np.exp(1 / 2),
        (np.exp(1) - 1) * np.exp(1),
        (np.exp(1) + 2) * np.sqrt(np.exp(1) - 1),
        np.exp(4) + 2 * np.exp(3) + 3 * np.exp(2) - 6,
    ]

    for i, val in zip(range(5), vals):
        assert np.isclose(data[f"y_0_moment_{i}"][0], val, rtol=10 ** (-4 + i))


def test_moment_nd():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    data.convert("nm")
    data.create_variable("x", (data.wa[:] - data.w1[:]).mean(axis=0, keepdims=True), units="nm")

    data.transform("w1", "x")

    data.level(0, 1, -10)
    data.moment(1, moment=0)
    assert data.channels[-1][10, 0] > 0  # check sign is as expected
    assert data.channels[-1].shape == (25, 1)
    data.moment(1, moment=1)
    assert data.channels[-1].shape == (25, 1)
    data.moment(1, moment=2)
    assert data.channels[-1].shape == (25, 1)
    data.close()


def test_moment_resultant():
    p = datasets.PyCMDS.w1_wa_000
    data = wt.data.from_PyCMDS(p)
    data.convert("nm")

    data.level(0, 1, -10)
    data.moment("wa", moment=0, resultant=data.w1.shape)
    assert data.channels[-1][10, 0] > 0  # check sign is as expected
    assert data.channels[-1].shape == (25, 1)
    data.moment("wa", moment=1, resultant=data.w1.shape)
    assert data.channels[-1].shape == (25, 1)
    data.moment("wa", moment=2, resultant=data.w1.shape)
    assert data.channels[-1].shape == (25, 1)
    data.close()


if __name__ == "__main__":
    test_moment_nd()
    test_moment_lognormal()
