"""Lineshapes."""


# --- import -------------------------------------------------------------


import numpy as np

from scipy.special import wofz


# --- define -------------------------------------------------------------


__all__ = ["gaussian", "lorentzian_complex", "lorentzian_real", "voigt"]


# --- functions ----------------------------------------------------------


def gaussian(x, x0, FWHM, norm="height"):
    """Calculate a normalized Gaussian lineshape

    Parameters
    ----------
    x : array_like or number
        Input array.
    x0 : array_like or number
        Center of lineshape.
    FHWM : array_like or number
        Full-width-at-half-maximum of lineshape.
    norm : string (optional):
        Type of normalization.
        height specifies that the maximum value is 1.
        area specifies that the lineshape integrates to 1.
        Default is height.

    Returns
    -------
    array_like or number
        Gaussian lineshape.
    """
    c = FWHM / (2 * np.sqrt(2 * np.log(2)))
    arr = np.exp(-1 * (x - x0) ** 2 / (2 * c ** 2))
    if norm == "height":
        return arr
    elif norm == "area":
        return arr / (c * np.sqrt(2 * np.pi))


def lorentzian_complex(x, x0, G, norm="height_imag"):
    """Calculate a normalized complex Lorentzian lineshape

    Parameters
    ----------
    x : array_like or number
        Input array.
    x0 : array_like or number
        Center of lineshape.
    G : array_like or number
        Half-width-at-half-maximum of lineshape.
    norm : string (optional):
        Type of normalization.
        height_imag specifies that the maximum value of the imaginary component is 1.
        area_int specifies that the square magnitude of the lineshape integrates to 1.
        Default is height_imag.

    Returns
    -------
    array_like or number
        Complex Lorentzian lineshape.
    """
    arr = 1 / (x0 - x - 1j * G)
    if norm == "height_imag":
        return arr * G
    elif norm == "area_int":
        return arr * np.sqrt(G / np.pi)


def lorentzian_real(x, x0, G, norm="height"):
    """Calculate a normalized Lorentzian lineshape

    Parameters
    ----------
    x : array_like or number
        Input array.
    x0 : array_like or number
        Center of lineshape.
    G : array_like or number
        Half-width-at-half-maximum of lineshape.
    norm : string (optional):
        Type of normalization.
        height specifies that the maximum value is 1.
        area specifies that the lineshape integrates to 1.
        Default is height.

    Returns
    -------
    array_like or number
        Lorentzian lineshape.
    """
    arr = G ** 2 / ((x - x0) ** 2 + G ** 2)
    if norm == "height":
        return arr
    if norm == "area":
        return arr / (G * np.pi)


def voigt(x, x0, FWHM, G):
    """Calculate an unnormalized Voigt lineshape using Scipy's Faddeeva function

    `Link to Voigt article on Wikipedia`__

    __ https://en.wikipedia.org/wiki/Voigt_profile

    Parameters
    ----------
    x : array_like or number
        Input array.
    x0 : array_like or number
        Center of lineshape.
    FHWM : array_like or number
        Full-width-at-half-maximum of Gaussian part of lineshape.
    G : array_like or number
        Half-width-at-half-maximum of Lorentzian part of lineshape.

    Returns
    -------
    array_like or number
        Voigt lineshape.
    """
    c = FWHM / (2 * np.sqrt(2 * np.log(2)))
    arr = (x - x0 + 1j * G) / (c * np.sqrt(2))
    w = wofz(arr)
    return np.real(w) / (c * np.sqrt(2 * np.pi))
