"""JASCO."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

from ._data import Axis, Channel, Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_JASCO']


# --- from function -------------------------------------------------------------------------------


def from_JASCO(filepath, name=None, kind='absorbance', verbose=True):
    """Create a data object from a JASCO UV-VIS NIR file.

    Parameters
    ----------
    filepath : string
        Path to JASCO output file (.txt).
    name : string (optional)
        Name to give to the created data object. If None, filename is used.
        Default is None.
    kind : {'absorbance', 'diffuse reflectance'} (optional)
        Kind of data taken. Default is absorbance.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        New data object.
    """
    # check filepath
    if not os.path.isfile(filepath):
        raise wt_exceptions.FileNotFound(path=filepath)
    filesuffix = os.path.basename(filepath).split('.')[-1]
    if filesuffix != 'txt':
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'txt')
    # import array
    arr = np.genfromtxt(filepath, skip_header=18).T
    # name
    if not name:
        name = filepath
    # construct data
    axis = Axis(arr[0], 'nm', name='wm')
    signal = Channel(arr[1], kind, signed=False)
    data = Data([axis], [signal], source='JASCO', name=name)
    # finish
    if verbose:
        print(data)
    return data
