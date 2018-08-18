"""Calculate."""


# --- import -------------------------------------------------------------


import numpy as np

from .. import units as wt_units


# --- define -------------------------------------------------------------


__all__ = ["fluence", "mono_resolution", "nm_width", "symmetric_sqrt"]

# --- functions ----------------------------------------------------------


def fluence(
    power_mW,
    color,
    beam_radius,
    reprate_Hz,
    pulse_width,
    color_units="wn",
    beam_radius_units="mm",
    pulse_width_units="fs_t",
    area_type="even",
) -> tuple:
    """Calculate the fluence of a beam.

    Parameters
    ----------
    power_mW : number
        Time integrated power of beam.
    color : number
        Color of beam in units.
    beam_radius : number
        Radius of beam in units.
    reprate_Hz : number
        Laser repetition rate in inverse seconds (Hz).
    pulse_width : number
        Pulsewidth of laser in units
    color_units : string (optional)
        Valid wt.units color unit identifier. Default is wn.
    beam_radius_units : string (optional)
        Valid wt.units distance unit identifier. Default is mm.
    pulse_width_units : number
        Valid wt.units time unit identifier. Default is fs.
    area_type : string (optional)
        Type of calculation to accomplish for Gaussian area.
        Currently nothing other than the default of even is implemented.

    Returns
    -------
    tuple
        Fluence in uj/cm2, photons/cm2, and peak intensity in GW/cm2

    """
    # calculate beam area
    if area_type == "even":
        radius_cm = wt_units.converter(beam_radius, beam_radius_units, "cm")
        area_cm2 = np.pi * radius_cm ** 2  # cm^2
    else:
        raise NotImplementedError
    # calculate fluence in uj/cm^2
    ujcm2 = power_mW / reprate_Hz  # mJ
    ujcm2 *= 1e3  # uJ
    ujcm2 /= area_cm2  # uJ/cm^2
    # calculate fluence in photons/cm^2
    energy = wt_units.converter(color, color_units, "eV")  # eV
    photonscm2 = ujcm2 * 1e-6  # J/cm2
    photonscm2 /= 1.60218e-19  # eV/cm2
    photonscm2 /= energy  # photons/cm2
    # calculate peak intensity in GW/cm^2
    pulse_width_s = wt_units.converter(pulse_width, pulse_width_units, "s_t")  # seconds
    GWcm2 = ujcm2 / 1e6  # J/cm2
    GWcm2 /= pulse_width_s  # W/cm2
    GWcm2 /= 1e9
    # finish
    return ujcm2, photonscm2, GWcm2


def mono_resolution(
    grooves_per_mm, slit_width, focal_length, output_color, output_units="wn"
) -> float:
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
    return abs(
        wt_units.converter(upper, "nm", output_units)
        - wt_units.converter(lower, "nm", output_units)
    )


def nm_width(center, width, units="wn") -> float:
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
    red = wt_units.converter(center - width / 2., units, "nm")
    blue = wt_units.converter(center + width / 2., units, "nm")
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
