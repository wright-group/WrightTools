"""Test lineshapes."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_gaussian():
    x = np.linspace(-2, 2, 1001)
    x0 = 0
    FWHM = 1
    y = wt.kit.gaussian(x, x0, FWHM, norm="area")
    assert np.isclose(np.trapz(y, x), 1, rtol=1e-3, atol=1e-3)
    y = wt.kit.gaussian(x, x0, FWHM, norm="height")
    assert np.isclose(y.max(), 1, rtol=1e-3, atol=1e-3)


def test_lorentzian_complex():
    x = np.linspace(-50, 50, 100001)
    x0 = 0
    G = 0.1
    y = wt.kit.lorentzian_complex(x, x0, G, norm="area_int")
    assert np.isclose(np.trapz(np.abs(y) ** 2, x), 1, rtol=1e-3, atol=1e-3)
    y = wt.kit.lorentzian_complex(x, x0, G, norm="height_imag")
    assert np.isclose(y.imag.max(), 1, rtol=1e-3, atol=1e-3)


def test_lorentzian_real():
    x = np.linspace(-50, 50, 100001)
    x0 = 0
    G = 0.1
    y = wt.kit.lorentzian_real(x, x0, G, norm="area")
    assert np.isclose(np.trapz(y, x), 1, rtol=1e-3, atol=1e-3)
    y = wt.kit.lorentzian_real(x, x0, G, norm="height")
    assert np.isclose(y.max(), 1, rtol=1e-3, atol=1e-3)


def test_voight():
    x = np.linspace(-2, 2, 1001)
    x0 = 0
    G = 0.5
    FWHM = 1
    gauss = wt.kit.gaussian(x, x0, FWHM, norm="height")
    lor = wt.kit.lorentzian_real(x, x0, G, norm="height")
    y = wt.kit.voight(x, x0, FWHM, 0)
    assert np.all(np.isclose(y / y.max(), gauss, rtol=1e-3, atol=1e-3))
    y = wt.kit.voight(x, x0, 1e-6, G)
    assert np.all(np.isclose(y / y.max(), lor, rtol=1e-3, atol=1e-3))
