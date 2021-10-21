"""Unit and label handling in WrightTools."""


# --- import --------------------------------------------------------------------------------------


import warnings

import pint


# --- define --------------------------------------------------------------------------------------

# Thise "blessed" units are here primarily for backwards compatibility, in particular
# to enable the behavior of `data.convert` which will convert freely between the energy units
# but does not go to time (where delay will)
# Since both of these context can convert to [length] units, they are interconvertible, but we
# do not want them to automatically do so.
# This list is (at creation time) purely reflective of historical units supported pre pint
# There is nothing preventing other units from being used and converted to, only to enable
# expected behavior
# 2021-01-29 KFS
blessed_units = (
    # angle
    "rad",
    "deg",
    # delay
    "fs",
    "ps",
    "ns",
    "mm_delay",
    # energy
    "nm",
    "wn",
    "eV",
    "meV",
    "Hz",
    "THz",
    "GHz",
    # optical density
    "mOD",
    # position
    "nm_p",
    "um",
    "mm",
    "cm",
    "in",
    # absolute temperature
    "K",
    "deg_C",
    "deg_F",
    "deg_R",
    # time
    "fs_t",
    "ps_t",
    "ns_t",
    "us_t",
    "ns_t",
    "s_t",
    "m_t",
    "h_t",
    "d_t",
)

ureg = pint.UnitRegistry()
ureg.define("[fluence] = [energy] / [area]")

ureg.define("OD = [] ")

ureg.define("wavenumber = 1 / cm = cm^{-1} = wn")


# Aliases for backwards compatability
ureg.define("@alias s = s_t")
ureg.define("@alias min = m_t")
ureg.define("@alias hour = h_t")
ureg.define("@alias d = d_t")

ureg.define("@alias degC = deg_C")
ureg.define("@alias degF = deg_F")
ureg.define("@alias degR = deg_R")

ureg.define("@alias m = m_delay")

delay = pint.Context("delay", defaults={"n": 1, "num_pass": 2})
delay.add_transformation(
    "[length]", "[time]", lambda ureg, x, n=1, num_pass=2: num_pass * x / ureg.speed_of_light * n
)
delay.add_transformation(
    "[time]", "[length]", lambda ureg, x, n=1, num_pass=2: x / num_pass * ureg.speed_of_light / n
)
ureg.enable_contexts("spectroscopy", delay)

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
    except (pint.errors.DimensionalityError, pint.errors.UndefinedUnitError, AttributeError):
        warnings.warn(
            f"conversion {current_unit} to {destination_unit} not valid: returning input"
        )
    except ZeroDivisionError:
        warnings.warn(
            f"conversion {current_unit} to {destination_unit} resulted in ZeroDivisionError: returning inf"
        )
        return float("inf")
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
    quantity = ureg.Quantity(1, ureg[units])
    if quantity.check("[length]"):
        return r"\lambda"
    elif quantity.check("1 / [length]"):
        return r"\bar\nu"
    elif quantity.check("[energy]"):
        return r"\hslash\omega"
    elif quantity.check("1 / [time]"):
        return "f"
    elif quantity.check("[time]"):
        return r"\tau"
    elif quantity.check("[fluence]"):
        return r"\mathcal{F}"
    elif quantity.check("[temperature]"):
        return "T"
    elif ureg[units] in (ureg.deg, ureg.radian):
        return r"\omega"
    else:
        return None


def get_valid_conversions(units, options=blessed_units) -> tuple:
    return tuple(i for i in options if is_valid_conversion(units, i) and units != i)


def is_valid_conversion(a, b, blessed=True) -> bool:
    if a is None:
        return b is None
    if blessed and a in blessed_units and b in blessed_units:
        blessed_energy_units = {"nm", "wn", "eV", "meV", "Hz", "THz", "GHz"}
        if a in blessed_energy_units:
            return b in blessed_energy_units
        blessed_delay_units = {"fs", "ps", "ns", "mm_delay"}
        if a in blessed_delay_units:
            return b in blessed_delay_units
        return ureg.Unit(a).dimensionality == ureg.Unit(b).dimensionality
    try:
        return ureg.Unit(a).is_compatible_with(b, "spectroscopy")
    except pint.UndefinedUnitError:
        return False


def kind(units):
    """Find the dimensionality of given units.

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
        return None
    return str(ureg.Unit(units).dimensionality)
