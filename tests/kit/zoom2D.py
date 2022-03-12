#! /usr/bin/env python3
"""Test smooth2D function."""

# --- import --------------------------------------------------------------------------------------

import WrightTools as wt
import numpy as np


# --- tests ---------------------------------------------------------------------------------------


def test_default():
    xi = np.linspace(-10, 10, 100)
    yi = np.linspace(-10, 10, 100)

    zi = np.sin(xi[:, None]) * np.cos(yi[None, :])

    xo, yo, zo = wt.kit.zoom2D(xi, yi, zi)

    zcheck = np.sin(xo[:, None]) * np.cos(yo[None, :])

    assert xo.shape == (300,)
    assert yo.shape == (300,)
    assert zo.shape == (300, 300)

    assert np.all(np.isclose(zo, zcheck, 0.01))  # all values within 1 percent of "actual"


def test_non_default():
    xi = np.linspace(-10, 10, 100)
    yi = np.linspace(-10, 10, 100)

    zi = np.sin(xi[:, None]) * np.cos(yi[None, :])

    xo, yo, zo = wt.kit.zoom2D(xi, yi, zi, 2, 4)

    zcheck = np.sin(xo[:, None]) * np.cos(yo[None, :])

    assert xo.shape == (200,)
    assert yo.shape == (400,)
    assert zo.shape == (200, 400)

    assert np.all(np.isclose(zo, zcheck, 0.01))  # all values within 1 percent of "actual"


# --- run -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    test_default()
    test_non_default()
