#! /usr/bin/env python3
"""Test channel.trim."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt

import matplotlib.pyplot as plt


# --- test ----------------------------------------------------------------------------------------


def test_trim_2Dgauss():
    # create original arrays
    x = np.linspace(-3, 3, 31)[:, None]
    y = np.linspace(-3, 3, 31)[None, :]
    arr = np.exp(-1 * (x ** 2 + y ** 2))
    # create damaged array
    arr2 = arr.copy()
    np.random.seed(11)  # set seed for reproducibility
    arr2[np.random.random(arr2.shape) < .05] = 2
    # create data object
    d = wt.data.Data()
    d.create_variable("x", values=x)
    d.create_variable("y", values=y)
    d.create_channel("original", arr)
    d.create_channel("damaged1", arr2)
    d.create_channel("damaged2", arr2)
    d.create_channel("damaged3", arr2)
    d.transform("x", "y")
    # trim
    d.damaged1.trim([2, 2], factor=2)
    d.damaged2.trim([2, 2], factor=2, replace="mean")
    d.damaged3.trim([2, 2], factor=2, replace=0.5)
    # now heal
    d.create_channel("healed_linear", d.damaged1[:])
    d.heal(channel="healed_linear", fill_value=0, method="linear")
    # check
    assert np.allclose(d.original[:], d.healed_linear[:], rtol=1e-1, atol=1e-1)
    assert np.allclose(d.original[:], d.damaged2[:], rtol=1e-1, atol=9e-1)
    assert np.allclose(d.original[:], d.damaged3[:], rtol=1e-1, atol=5e-1)


def test_trim_3Dgauss():
    # create original arrays
    x = np.linspace(-3, 3, 31)[:, None, None]
    y = np.linspace(-3, 3, 31)[None, :, None]
    z = np.linspace(-3, 3, 31)[None, None, :]
    arr = np.exp(-1 * (x ** 2 + y ** 2 + z ** 2))
    # create damaged array
    arr2 = arr.copy()
    np.random.seed(11)  # set seed for reproducibility
    arr2[np.random.random(arr2.shape) < .05] = 1
    # create data object
    d = wt.data.Data()
    d.create_variable("x", values=x)
    d.create_variable("y", values=y)
    d.create_variable("z", values=z)
    d.create_channel("original", arr)
    d.create_channel("damaged", arr2)
    d.transform("x", "y", "z")
    # trim
    d.damaged.trim([2, 2, 2], factor=2, replace="mean")
    # check
    assert np.allclose(d.original[:], d.damaged[:], rtol=1e-1, atol=9e-1)


if __name__ == "__main__":
    test_trim_2Dgauss()
    test_trim_3Dgauss()
