"""Unit and label handling in WrightTools."""


# --- import --------------------------------------------------------------------------------------


import collections
import warnings

import numpy as np
import numexpr

import pint

ureg = pint.UnitRegistry()
ureg.enable_contexts("spectroscopy")


# --- define --------------------------------------------------------------------------------------

ureg.define("[fluence] = [energy] / [area]")

ureg.define("OD = [] ")

ureg.define("wavenumber = 1 / cm = wn")


# Aliases for backwards compatability
ureg.define("@alias s = s_t")
ureg.define("@alias min = m_t")
ureg.define("@alias hour = h_t")
ureg.define("@alias d = d_t")

ureg.define("@alias degC = deg_C")
ureg.define("@alias degF = deg_F")
ureg.define("@alias degR = deg_R")

# delay units (native: fs)
fs_per_mm = 3336.0
delay = {
    "fs": ["x", "x", r"fs"],
    "ps": ["x*1e3", "x/1e3", r"ps"],
    "mm_delay": ["x*2*fs_per_mm", "x/(2*fs_per_mm)", r"mm"],
}

# --- functions -----------------------------------------------------------------------------------


def converter(val, current_unit, destination_unit):
    """Convert from one unit to another.

    Parameters
    ----------
    val : number
        Number to convert.
    current_unit : string
        Current unit.
    destination_unit : string
        Destination unit.

    Returns
    -------
    number
        Converted value.
    """
    try:
        val = ureg.Quantity(val, current_unit).to(destination_unit).magnitude
    except:
        warnings.warn(
            "conversion {0} to {1} not valid: returning input".format(
                current_unit, destination_unit
            )
        )
    return val


convert = converter


def get_symbol(units) -> str:
    """Get default symbol type.

    Parameters
    ----------
    units_str : string
        Units.

    Returns
    -------
    string
        LaTeX formatted symbol.
    """
    dimensionality = str(ureg.Unit(units).dimensionality)
    if dimensionality == "[length]":
        return r"\lambda"
    if dimensionality == "1 / [length]":
        return r"\bar\nu"
    if dimensionality == "[length] ** 2 * [mass] / [time] ** 2":
        return r"\hslash\omega"
    if dimensionality == "1 / [time]":
        return "f"
    if dimensionality == "[time]":
        return r"\tau"
    if dimensionality == "[mass] / [time] ** 2":
        return r"\mathcal{F}"
    if dimensionality == "[temperature]":
        return "T"
    else:
        return kind(units)


def get_valid_conversions(units) -> tuple:
    return ()
    try:
        valid = list(dicts[kind(units)])
    except KeyError:
        return ()
    valid.remove(units)
    return tuple(valid)


def is_valid_conversion(a, b) -> bool:
    if a is None:
        return b is None
    try:
        return ureg.Unit(a).is_compatible_with(b, "spectroscopy")
    except pint.UndefinedUnitError:
        return False


def kind(units):
    """Find the kind of given units.

    Parameters
    ----------
    units : string
        The units of interest

    Returns
    -------
    string
        The kind of the given units. If no match is found, returns None.
    """
    if units is None:
        return
    return str(ureg.Unit(units).dimensionality)
