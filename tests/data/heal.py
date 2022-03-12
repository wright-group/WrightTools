#! /usr/bin/env python3
"""Test data.heal."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_heal_gauss():
    # create original arrays
    x = np.linspace(-3, 3, 31)[:, None]
    y = np.linspace(-3, 3, 31)[None, :]
    arr = np.exp(-1 * (x ** 2 + y ** 2))
    # create damaged array
    arr2 = arr.copy()
    np.random.seed(11)  # set seed for reproducibility
    arr2[np.random.random(arr2.shape) < 0.2] = np.nan
    # create data object
    d = wt.data.Data()
    d.create_variable("x", values=x)
    d.create_variable("y", values=y)
    d.create_channel("original", arr)
    d.create_channel("damaged", arr2)
    d.create_channel("healed_linear", arr2)
    d.create_channel("healed_nearest", arr2)
    d.create_channel("healed_cubic", arr2)
    d.transform("x", "y")
    # heal
    d.heal(channel="healed_linear", fill_value=0, method="linear")
    d.heal(channel="healed_nearest", fill_value=0, method="nearest")
    d.heal(channel="healed_cubic", fill_value=0, method="cubic")
    # check
    assert np.allclose(d.original[:], d.healed_linear[:], rtol=1e-1, atol=1e-1)
    assert np.allclose(d.original[:], d.healed_nearest[:], rtol=5e-1, atol=5e-1)
    assert np.allclose(d.original[:], d.healed_cubic[:], rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    test_heal_gauss()
