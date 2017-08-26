"""Custom exception types."""


# --- import --------------------------------------------------------------------------------------


import os

import warnings


# --- custom exceptions ---------------------------------------------------------------------------


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
