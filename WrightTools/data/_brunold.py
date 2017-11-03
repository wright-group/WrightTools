"""Brunold."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

from ._data import Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_BrunoldrRaman']


# --- from function -------------------------------------------------------------------------------


def from_BrunoldrRaman(filepath, name=None, collection=None, verbose=True):
    """Create a data object from the Brunold rRaman instrument.

    Expects one energy (in wavenumbers) and one counts value.

    Parameters
    ----------
    filepath : string, list of strings, or array of strings
        Path to .txt file.
    name : string (optional)
        Name to give to the created data object. If None, filename is used.
        Default is None.
    collection : WrightTools.Collection (optional)
        Collection to place new data object within. Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        New data object(s).
    """
    # parse filepath
    if not os.path.isfile(filepath):
        raise wt_exceptions.FileNotFound(path=filepath)
    if not filepath.endswith('txt'):
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'txt')
    # parse name
    if not name:
        name = os.path.basename(filepath).split('.')[0]
    # create data
    kwargs = {'name': name, 'kind': 'BrunoldrRaman', 'source': filepath}
    if collection is None:
        data = Data(**kwargs)
    else:
        data = collection.create_data(**kwargs)
    # array
    arr = np.genfromtxt(filepath, delimiter='\t').T
    # chew through all scans
    data.create_axis(name='wm', points=arr[0], units='wn')
    data.create_channel(name='counts', values=arr[1])
    # finish
    if verbose:
        print('data created at {0}'.format(data.fullpath))
        print('  kind: {0}'.format(data.kind))
        print('  range: {0} to {1} (wn)'.format(data.wm[0], data.wm[-1]))
        print('  size: {0}'.format(data.size))
    return data
