"""Ocean Optics."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

from ._data import Axis, Channel, Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_scope']


# --- from function -------------------------------------------------------------------------------


def from_scope(filepath, name=None, verbose=True):
    """Create a Data object from an Ocean Optics .scope file.

    Parameters
    ----------
    filepath : path
        Filepath to .scope file.
    name : string (optional)
        Name of Data object. Default is None (filename).
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    WrightTools.data.Data object
    """
    # check filepath
    if os.path.isfile(filepath):
        if verbose:
            print('found the file!')
    else:
        raise wt_exceptions.FileNotFound('{0}'.format(filepath))
    # import
    skip_header = 14
    skip_footer = 1
    arr = np.genfromtxt(filepath, skip_header=skip_header,
                        skip_footer=skip_footer, delimiter='\t').T
    # construct data
    a = Axis(arr[0], 'nm', name='wm')
    c = Channel(arr[1], name='intensity', signed=False)
    data = Data([a], [c], source='scope', name=name)
    # finish
    return data
