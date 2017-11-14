"""Calculate."""


# --- import --------------------------------------------------------------------------------------


import numpy as np

from .. import units as wt_units


# --- define --------------------------------------------------------------------------------------


__all__ = ['mono_resolution', 'nm_width', 'symmetric_sqrt']


# --- functions -----------------------------------------------------------------------------------


def mono_resolution(grooves_per_mm, slit_width, focal_length, output_color, output_units='wn'):
    """Calculate the resolution of a monochromator.

    Parameters
    ----------
    grooves_per_mm : number
        Grooves per millimeter.
    slit_width : number
        Slit width in microns.
    focal_length : number
        Focal length in mm.
    output_color : number
        Output color in nm.
    output_units : string (optional)
        Output units. Default is wn.

    Returns
    -------
    float
        Resolution.
    """
    d_lambda = 1e6 * slit_width / (grooves_per_mm * focal_length)  # nm
    upper = output_color + d_lambda / 2  # nm
    lower = output_color - d_lambda / 2  # nm
    return abs(wt_units.converter(upper, 'nm', output_units) -
               wt_units.converter(lower, 'nm', output_units))


def nm_width(center, width, units='wn'):
    """Given a center and width, in energy units, get back a width in nm.

    Parameters
    ----------
    center : number
        Center (in energy units).
    width : number
        Width (in energy units).
    units : string (optional)
        Input units. Default is wn.

    Returns
    -------
    number
        Width in nm.
    """
    red = wt_units.converter(center - width / 2., units, 'nm')
    blue = wt_units.converter(center + width / 2., units, 'nm')
    return red - blue


def symmetric_sqrt(x, out=None):
    """Compute the 'symmetric' square root: sign(x) * sqrt(abs(x)).

    Parameters
    ----------
    x : array_like or number
        Input array.
    out : ndarray, None, or tuple of ndarray and None (optional)
        A location into which the result is stored. If provided, it must
        have a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.

    Returns
    -------
    np.ndarray
        Symmetric square root of arr.
    """
    factor = np.sign(x)
    out = np.sqrt(np.abs(x), out=out)
    return out * factor
