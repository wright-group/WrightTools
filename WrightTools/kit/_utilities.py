"""Utilities."""


# --- import --------------------------------------------------------------------------------------


import time
import string
from .._axes import operator_to_identifier, operators


# --- define --------------------------------------------------------------------------------------


__all__ = [
    "string2identifier",
    "data_from_slice", 
    "Timer",
]


# --- functions -----------------------------------------------------------------------------------


def string2identifier(s):
    """Turn a string into a valid Axis identifier.

    Currently only allows ASCII letters and underscore. Operators are replaced 
    with dunder string representations. Other illegal characters are replaced
    with underscore. This is slightly more opinionated than python 3 itself,
    and may be refactored in future (see PEP 3131).

    Parameters
    ----------
    s : string
        string to convert

    Returns
    -------
    str
        valid Axis identifier.
    """
    # https://docs.python.org/3/reference/lexical_analysis.html#identifiers
    # https://www.python.org/dev/peps/pep-3131/
    if len(s) == 0:
        return "_"
    if s[0] not in string.ascii_letters:
        s = "_" + s
    valids = string.ascii_letters + string.digits + "_"
    out = ""
    for i, char in enumerate(s):
        if char in operators:
            out += operator_to_identifier(char)
        elif char in valids:
            out += char
        else:
            out += "_"
    return out


def data_from_slice(data, idx, name=None, parent=None):
    """create data from an array slice of the parent data"""
    if parent is None:
        out = Data(name=name)
    else:
        out = parent.create_data(name=name)

    for v in data.variables:
        kwargs = {}
        kwargs["name"] = v.natural_name
        kwargs["values"] = v[idx]
        kwargs["units"] = v.units
        kwargs["label"] = v.label
        kwargs.update(v.attrs)
        out.create_variable(**kwargs)
    for c in data.channels:
        kwargs = {}
        kwargs["name"] = c.natural_name
        kwargs["values"] = c[idx]
        kwargs["units"] = c.units
        kwargs["label"] = c.label
        kwargs["signed"] = c.signed
        kwargs.update(c.attrs)
        out.create_channel(**kwargs)

    new_axis_units = [a.units for a in data.axes]
    out.transform(*data.axis_expressions)

    for const in data.constant_expressions:
        out.create_constant(const, verbose=False)
    for j, units in enumerate(new_axis_units):
        out.axes[j].convert(units)

    return out


# --- classes -------------------------------------------------------------------------------------


class Timer:
    """Context manager for timing code.

    >>> with Timer():
    ...     your_code()
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self, progress=None):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.verbose:
            print("elapsed time: {0} sec".format(self.interval))
