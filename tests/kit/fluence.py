"""Test fluence."""


# --- import -------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ---------------------------------------------------------------


def test_even():
    out = wt.kit.fluence(1, 2, 0.1, 1000, 1, "eV", "cm", "ps_t")
    checks = (31.83098, 99336493460095.2, 0.03183098)
    assert np.isclose(checks[0], out[0], rtol=1e-3)
    assert np.isclose(checks[1], out[1], rtol=1e-3)
    assert np.isclose(checks[2], out[2], rtol=1e-3)


def test_average():
    out = wt.kit.fluence(1, 2, 0.1, 1000, 1, "eV", "cm", "ps_t", "average")
    checks = (22.95969209043682, 71651412732766.67, 0.02295969209043682)
    assert np.isclose(checks[0], out[0], rtol=1e-3)
    assert np.isclose(checks[1], out[1], rtol=1e-3)
    assert np.isclose(checks[2], out[2], rtol=1e-3)


if __name__ == "__main__":
    test_even()
    test_average()
