"""Utilities."""


# --- import --------------------------------------------------------------------------------------


import os
import string


# --- define --------------------------------------------------------------------------------------


__all__ = ['get_methods', 'string2identifier', 'Suppress', 'Timer']


# --- functions -----------------------------------------------------------------------------------


def get_methods(the_class, class_only=False, instance_only=False,
                exclude_internal=True):
    """Get a list of strings corresponding to the names of the methods of an object."""
    import inspect

    def acceptMethod(tup):
        # internal function that analyzes the tuples returned by getmembers
        # tup[1] is the actual member object
        is_method = inspect.ismethod(tup[1])
        if is_method:
            bound_to = tup[1].im_self
            internal = (tup[1].im_func.func_name[:2] == '__' and
                        tup[1].im_func.func_name[-2:] == '__')
            if internal and exclude_internal:
                include = False
            else:
                include = (bound_to == the_class and not instance_only) or (
                    bound_to is None and not class_only)
        else:
            include = False
        return include

    # filter to return results according to internal function and arguments
    tups = filter(acceptMethod, inspect.getmembers(the_class))
    return [tup[0] for tup in tups]


def string2identifier(s):
    """Turn a string into a valid python identifier.

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
    if s[0] not in string.ascii_letters:
        s = '_' + s
    valids = string.ascii_letters + string.digits + '_'
    out = ''
    for i, char in enumerate(s):
        if char in valids:
            out += char
        else:
            out += '_'
    return out


# --- classes -------------------------------------------------------------------------------------


class Suppress(object):
    """Context manager for doing a "deep suppression" of stdout and stderr in Python.

    i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    `Source`__

    __ http://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions


    >>> with WrightTools.kit.Supress():
    ...     rogue_function()

    """

    def __init__(self):
        """init."""
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        """enter."""
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        """exit."""
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class Timer:
    """Context manager for timing code.

    >>> with Timer():
    ...     your_code()
    """

    def __init__(self, verbose=True):
        """init."""
        self.verbose = verbose

    def __enter__(self, progress=None):
        """enter."""
        self.start = clock()

    def __exit__(self, type, value, traceback):
        """exit."""
        self.end = clock()
        self.interval = self.end - self.start
        if self.verbose:
            print('elapsed time: {0} sec'.format(self.interval))
