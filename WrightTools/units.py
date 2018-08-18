"""Unit and label handling in WrightTools."""


# --- import --------------------------------------------------------------------------------------


import collections

import numpy as np
import warnings


# --- define --------------------------------------------------------------------------------------


# units are stored in dictionaries of like kind. format:
#     unit : to native, from native, units_symbol, units_label

# angle units (native: rad)
angle = {"rad": ["x", "x", r"rad"], "deg": ["x/57.2958", "57.2958*x", r"deg"]}

# delay units (native: fs)
fs_per_mm = 3336.
delay = {
    "fs": ["x", "x", r"fs"],
    "ps": ["x*1e3", "x/1e3", r"ps"],
    "ns": ["x*1e6", "x/1e6", r"ns"],
    "mm_delay": ["x*2*fs_per_mm", "x/(2*fs_per_mm)", r"mm"],
}

# energy units (native: nm)
energy = {
    "nm": ["x", "x", r"nm"],
    "wn": ["1e7/x", "1e7/x", r"cm^{-1}"],
    "eV": ["1240./x", "1240./x", r"eV"],
    "meV": ["1240000./x", "1240000./x", r"meV"],
    "Hz": ["2.99792458e17/x", "2.99792458e17/x", r"Hz"],
    "THz": ["2.99792458e5/x", "2.99792458e5/x", r"THz"],
    "GHz": ["2.99792458e8/x", "2.99792458e8/x", r"GHz"],
}

# fluence units (native: uJ per sq. cm)
fluence = {"uJ per sq. cm": ["x", "x", r"\frac{\mu J}{cm^{2}}"]}

# optical density units (native: od)
od = {"mOD": ["1e3*x", "x/1e3", r"mOD"], "OD": ["x", "x", r"OD"]}

# position units (native: mm)
position = {
    "nm_p": ["x/1e6", "1e6/x", r"nm"],
    "um": ["x/1000.", "1000.*x", r"um"],
    "mm": ["x", "x", r"mm"],
    "cm": ["10.*x", "x/10.", r"cm"],
    "in": ["x*0.039370", "0.039370*x", r"in"],
}

# pulse width units (native: FWHM)
pulse_width = {"FWHM": ["x", "x", r"FWHM"]}

# time units (native: s)
time = {
    "fs_t": ["x/1e15", "x*1e15", r"fs"],
    "ps_t": ["x/1e12", "x*1e12", r"ps"],
    "ns_t": ["x/1e9", "x*1e9", r"ns"],
    "us_t": ["x/1e6", "x*1e6", r"us"],
    "ms_t": ["x/1000.", "x*1000.", r"ms"],
    "s_t": ["x", "x", r"s"],
    "m_t": ["x*60.", "x/60.", r"m"],
    "h_t": ["x*3600.", "x/3600.", r"h"],
    "d_t": ["x*86400.", "x/86400.", r"d"],
}

dicts = collections.OrderedDict()
dicts["angle"] = angle
dicts["delay"] = delay
dicts["energy"] = energy
dicts["time"] = time
dicts["position"] = position
dicts["pulse_width"] = pulse_width
dicts["fluence"] = fluence
dicts["od"] = od


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
    x = val
    for dic in dicts.values():
        if current_unit in dic.keys() and destination_unit in dic.keys():
            try:
                native = eval(dic[current_unit][0])
            except ZeroDivisionError:
                native = np.inf
            x = native  # noqa: F841
            try:
                out = eval(dic[destination_unit][1])
            except ZeroDivisionError:
                out = np.inf
            return out
    # if all dictionaries fail
    if current_unit is None and destination_unit is None:
        pass
    else:
        warnings.warn(
            "conversion {0} to {1} not valid: returning input".format(
                current_unit, destination_unit
            )
        )
    return val


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
    if kind(units) == "energy":
        d = {}
        d["nm"] = r"\lambda"
        d["wn"] = r"\bar\nu"
        d["eV"] = r"\hslash\omega"
        d["Hz"] = r"f"
        d["THz"] = r"f"
        d["GHz"] = r"f"
        return d.get(units, "E")
    elif kind(units) == "delay":
        return r"\tau"
    elif kind(units) == "fluence":
        return r"\mathcal{F}"
    elif kind(units) == "pulse_width":
        return r"\sigma"
    else:
        return kind(units)


def get_valid_conversions(units) -> tuple:
    try:
        valid = list(dicts[kind(units)])
    except KeyError:
        return ()
    valid.remove(units)
    return tuple(valid)


def is_valid_conversion(a, b) -> bool:
    for dic in dicts.values():
        if a in dic.keys() and b in dic.keys():
            return True
    if a is None and b is None:
        return True
    else:
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
    for k, v in dicts.items():
        if units in v.keys():
            return k
