"""Utilities."""


# --- import --------------------------------------------------------------------------------------


import time
import string


# --- define --------------------------------------------------------------------------------------


__all__ = ["string2identifier", "Timer", "data_from_slice"]


# --- functions -----------------------------------------------------------------------------------


def string2identifier(s):
    """Turn a string into a valid python identifier.

    Currently only allows ASCII letters and underscore. Illegal characters
    are replaced with underscore. This is slightly more opinionated than
    python 3 itself, and may be refactored in future (see PEP 3131).

    Parameters
    ----------
    s : string
        string to convert

    Returns
    -------
    str
        valid python identifier.
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
        if char in valids:
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

    new_axes = [a.expression for a in data._axes]
    new_axis_units = [a.units for a in data._axes]
    out.transform(*new_axes)

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
