"""Custom exception types."""


# --- import --------------------------------------------------------------------------------------


import os

import warnings


# --- custom exceptions ---------------------------------------------------------------------------


class DimensionalityError(Exception):
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
        Exception.__init__(self, message)


class FileNotFound(Exception):
    """FileNotFound."""

    def __init__(self, path):
        """Format a file not found exception.

        Parameters
        ----------
        path : string
            Given path.
        """
        message = 'no file was found at {}'.format(path)
        Exception.__init__(self, message)


class NameNotUniqueError(Exception):
    """NameNotUniqueError."""

    def __init__(self, name):
        """Format a Name Not Unique Error.

        Parameters
        ----------
        name : string
            Name of an attribute which causes a duplication.
        """
        message = 'Name {} results in a duplicate'.format(name)
        Exception.__init__(self, message)


# --- custom warnings -----------------------------------------------------------------------------


class VisibleDeprecationWarning(Warning):
    """VisibleDepreciationWarning."""

    pass


class WrongFileTypeWarning(Warning):
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
