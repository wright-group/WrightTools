"""Brunold."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

from ._data import Axis, Channel, Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_BrunoldrRaman']


# --- from function -------------------------------------------------------------------------------


def from_BrunoldrRaman(filepath, name=None, verbose=True):
    """Create a data object from the Brunold rRaman instrument.

    Expects one energy (in wavenumbers) and one counts value.

    Parameters
    ----------
    filepath : string, list of strings, or array of strings
        Path to .txt file.
    name : string (optional)
        Name to give to the created data object. If None, filename is used.
        Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        New data object(s).
    """
    if not os.path.isfile(filepath):
        raise wt_exceptions.FileNotFound(path=filepath)
    if not filepath.endswith('txt'):
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'txt')
    # import array
    arr = np.genfromtxt(filepath, delimiter='\t').T
    # chew through all scans
    axis = Axis(arr[0], 'wn', name='wm')
    signal = Channel(arr[1], name='signal', label='counts', signed=False)
    if name:
        data = Data([axis], [signal], source='Brunold rRaman', name=name)
    else:
        name = filepath.split('//')[-1].split('.')[0]
        data = Data([axis], [signal], source='Brunold rRaman', name=name)
    # finish
    if verbose:
        print('1 data object successfully created from file')
    return data
