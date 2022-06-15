"""Utilities."""


# --- import --------------------------------------------------------------------------------------


import time
import string


# --- define --------------------------------------------------------------------------------------


__all__ = ["string2identifier", "Timer"]


# --- functions -----------------------------------------------------------------------------------


def string2identifier(s, replace=None):
    """Turn a string into a valid python identifier.

    This method restricts identifier characters to ASCII letters, numbers, and
    underscore. The characters are slightly more restrictive than python 3
    itself, and may be refactored in future (see PEP 3131).

    For non-valid characters, the default replacement is "_". Replacement
    assignments can be customized with the replace kwarg.

    Parameters
    ----------
    s : string
        string to convert
    replace: dictionary[str, str] (optional)
        dictionary of characters (keys) and their replacements (values). Values
        should be ASCII or underscore. Unspecified non-ascii characters are
        converted to underscore.

    Returns
    -------
    str
        valid python identifier.
    """
    if len(s) == 0:
        return "_"
    if s[0] not in string.ascii_letters:
        s = "_" + s
    valids = string.ascii_letters + string.digits + "_"
    out = ""
    for i, char in enumerate(s):
        if replace and (char in replace.keys()):
            out += replace[char]
        elif char in valids:
            out += char
        else:
            out += "_"
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
