"""Custom exception types."""


# --- import --------------------------------------------------------------------------------------


import os

import warnings


# --- custom exceptions ---------------------------------------------------------------------------

class WrightToolsError(Exception):
    """WrightTools Base Exception."""
    pass

class DimensionalityError(WrightToolsError):
    """DimensionalityError."""

    def __init__(self, expected, recieved):
        """Dimensionality error.

        Parameters
        ----------
        expected : object
            Expected dimensionalit(ies).
        recieved : object
            Recieved dimensionality.
        """
        message = "dimensionality must be {0} (recieved {1})".format(expected, recieved)
        WrightToolsError.__init__(self, message)


class NameNotUniqueError(WrightToolsError):
    """NameNotUniqueError."""

    def __init__(self, name=None):
        """Format a Name Not Unique Error.

        Parameters
        ----------
        name : string
            Name of an attribute which causes a duplication.
        """
        if name is not None:
            message = 'Name {} results in a duplicate'.format(name)
        else:
            message = "Names must be unique"
        WrightToolsError.__init__(self, message)


class UnitsError(WrightToolsError):
    """Units Error."""

    def __init__(self, expected, recieved):
        """Units error.

        Parameters
        ----------
        expected : object
            Expected units.
        recieved : object
            Recieved units.
        """
        message = "expected units of {0} (recieved {1})".format(expected, recieved)
        WrightToolsError.__init__(self, message)


# --- custom warnings -----------------------------------------------------------------------------

class WrightToolsWarning(Warning):
    """WrightTools Base Warning."""
    pass

class VisibleDeprecationWarning(WrightToolsWarning):
    """VisibleDepreciationWarning."""

    pass


class WrongFileTypeWarning(WrightToolsWarning):
    """WrongFileTypeWarning."""

    def warn(filepath, expected):
        """Raise warning.

        Parameters
        ----------
        filepath : string
            Given filepath.
        expected : string
            Expected file suffix.
        """
        filesuffix = os.path.basename(filepath).split('.')[-1]
        message = 'file {0} has type {1} (expected {2})'.format(filepath, filesuffix, 'txt')
        warnings.warn(message, WrongFileTypeWarning)
