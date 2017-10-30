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


def from_JASCO(filepath, name=None, collection=None, verbose=True):
    """Create a data object from a JASCO UV-VIS NIR file.

    Parameters
    ----------
    filepath : string
        Path to JASCO output file (.txt).
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
        New data object.
    """
    # parse filepath
    if not os.path.isfile(filepath):
        raise wt_exceptions.FileNotFound(path=filepath)
    filesuffix = os.path.basename(filepath).split('.')[-1]
    if filesuffix != 'txt':
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'txt')
    # parse name
    if not name:
        name = filepath
    # create data
    kwargs = {'name': name, 'kind': 'JASCO', 'source': filepath}
    if collection is None:
        data = Data(**kwargs)
    else:
        data = collection.create_data(**kwargs)
    # array
    arr = np.genfromtxt(filepath, skip_header=18).T
    # construct data
    data.create_axis(name='wm', points=arr[0], units='nm')
    data.create_channel(name='channel', values=arr[1])
    # finish
    if verbose:
        print('data created at {0}'.format(data.fullpath))
        print('  kind: {0}'.format(data.kind))
        print('  range: {0} to {1} (nm)'.format(data.wm[0], data.wm[-1]))
        print('  size: {0}'.format(data.size))
    return data
