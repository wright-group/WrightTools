"""Test fft."""


# --- import --------------------------------------------------------------------------------------


import pytest

import numpy as np

import WrightTools as wt


# --- test ----------------------------------------------------------------------------------------


def test_1_sin():
    t = np.linspace(-10, 10, 10000)
    z = np.sin(2 * np.pi * t)
    wi, zi = wt.kit.fft(t, z)
    freq = np.abs(wi[np.argmax(zi)])
    assert np.isclose(freq, 1, rtol=1e-3, atol=1e-3)


def test_plancherel():
    t = np.linspace(-10, 10, 10000)
    z = np.sin(2 * np.pi * t)
    wi, zi = wt.kit.fft(t, z)
    intensity_time = (z**2).sum() * (t[1] - t[0])
    intensity_freq = (zi * zi.conjugate()).real.sum() * (wi[1] - wi[0])
    rel_error = np.abs(intensity_time - intensity_freq) / (intensity_time + intensity_freq)
    print(intensity_time, intensity_freq, rel_error)
    assert rel_error < 1e-12


def test_5_sines():
    t = np.linspace(-20, 20, 10000)
    freqs = np.linspace(1, 5, 5)
    z = np.sin(2 * np.pi * freqs[None, :] * t[:, None])
    wi, zi = wt.kit.fft(t, z, axis=0)
    freq = np.abs(wi[np.argmax(zi, axis=0)])
    assert np.all(np.isclose(freq, freqs, rtol=1e-3, atol=1e-3))


def test_dimensionality_error():
    with pytest.raises(wt.exceptions.DimensionalityError):
        t = np.linspace(-20, 20, 10000)
        freqs = np.linspace(1, 5, 5)
        z = np.sin(2 * np.pi * freqs[None, :] * t[:, None])
        t = t[:, None]
        wt.kit.fft(t, z, axis=0)


def test_even_spacing_error():
    with pytest.raises(RuntimeError):
        xi = np.logspace(0, 2, 50)
        yi = 0
        wt.kit.fft(xi, yi)
