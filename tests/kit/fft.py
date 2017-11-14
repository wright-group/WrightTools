"""Test fft."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_1_sin():
    t = np.linspace(-10, 10, 10000)
    z = np.sin(2 * np.pi * t)
    wi, zi = wt.kit.fft(t, z)
    freq = np.abs(wi[np.argmax(zi)])
    assert np.isclose(freq, 1, rtol=1e-3, atol=1e-3)


def test_5_sines():
    t = np.linspace(-20, 20, 10000)
    freqs = np.linspace(1, 5, 5)
    z = np.sin(2 * np.pi * freqs[None, :] * t[:, None])
    wi, zi = wt.kit.fft(t, z, axis=0)
    freq = np.abs(wi[np.argmax(zi, axis=0)])
    assert np.all(np.isclose(freq, freqs, rtol=1e-3, atol=1e-3))


def test_dimensionality_error():
    try:
        t = np.linspace(-20, 20, 10000)
        freqs = np.linspace(1, 5, 5)
        z = np.sin(2 * np.pi * freqs[None, :] * t[:, None])
        t = t[:, None]
        wi, zi = wt.kit.fft(t, z, axis=0)
    except wt.exceptions.DimensionalityError:
        assert True


def test_even_spacing_error():
    try:
        xi = np.logspace(0, 2, 50)
        yi = 0
        wt.kit.fft(xi, yi)
    except RuntimeError:
        assert True
