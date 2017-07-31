"""
Custom exception types
"""


# --- import --------------------------------------------------------------------------------------


import os

import warnings


# --- custom exceptions ---------------------------------------------------------------------------


class FileNotFound(Exception):

    def __init__(self, path):
        message = 'no file was found at {}'.format(path)
        Exception.__init__(self, message)


# --- custom warnings -----------------------------------------------------------------------------


class WrongFileTypeWarning(Warning):

    def warn(filepath, expected):
        filesuffix = os.path.basename(filepath).split('.')[-1]
        message = 'file {0} has type {1} (expected {2})'.format(filepath, filesuffix, 'txt')
        warnings.warn(message, WrongFileTypeWarning)
