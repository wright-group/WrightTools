"""Custom exception types."""


# --- import --------------------------------------------------------------------------------------


import os

import warnings


# --- custom exceptions ---------------------------------------------------------------------------


class WrightToolsException(Exception):
    """WrightTools Base Exception."""

    pass


class DimensionalityError(WrightToolsException):
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
        super().__init__(self, message)


class NameNotUniqueError(WrightToolsException):
    """NameNotUniqueError."""

    def __init__(self, name=None):
        """Format a Name Not Unique Error.

        Parameters
        ----------
        name : string
            Name of an attribute which causes a duplication.
        """
        if name is not None:
            message = "Name {} results in a duplicate".format(name)
        else:
            message = "Names must be unique"
        super().__init__(self, message)


class MultidimensionalAxisError(WrightToolsException):
    """Error for when operation does not support Multidimensional Axes."""

    def __init__(self, axis, operation):
        """Multidimesional Axis error.

        Parameters
        ----------
        axis : str
            Name of axis which causes the error.
        operation : str
            Name of operation which cannot handle multidimensional axes.
        """
        message = "{} can not handle multidimensional axis: {}".format(operation, axis)
        super().__init__(self, message)


class ValueError(ValueError, WrightToolsException):
    """Raised when an argument has the right type but an inappropriate value."""

    pass


class FileExistsError(FileExistsError, WrightToolsException):
    """Raised when trying to create a file or directory which already exists.

    Corresponds to errno ``EEXIST``.
    """

    pass


class TypeError(TypeError, WrightToolsException):
    """Raised when an operation or function is applied to an object of inappropriate type.

    The associated value is a string giving details about the type mismatch.
    """

    pass


class UnitsError(WrightToolsException):
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
        super().__init__(self, message)


# --- custom warnings -----------------------------------------------------------------------------


class WrightToolsWarning(Warning):
    """WrightTools Base Warning."""

    pass


class EntireDatasetInMemoryWarning(WrightToolsWarning):
    """Warn when an entire dataset is taken into memory at once.

    Such operations may lead to memory overflow errors for large datasets.

    Warning ignored by default.
    """

    pass


warnings.simplefilter("ignore", category=EntireDatasetInMemoryWarning)  # ignore by default


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
        filesuffix = os.path.basename(filepath).split(".")[-1]
        message = "file {0} has type {1} (expected {2})".format(filepath, filesuffix, expected)
        warnings.warn(message, WrongFileTypeWarning)


class ObjectExistsWarning(WrightToolsWarning):
    """Warn that an HDF5 object already exists when a new one is requested."""

    def warn(name):
        message = "object '{0}' already exists, returning existing copy".format(name)
        warnings.warn(message, ObjectExistsWarning)
