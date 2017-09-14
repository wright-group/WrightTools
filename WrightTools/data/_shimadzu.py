"""Shimadzu."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

from ._data import Axis, Channel, Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_shimadzu']


# --- from function -------------------------------------------------------------------------------


def from_shimadzu(filepath, name=None, verbose=True):
    """Create a Data object from a Shimadzu txt file.

    Parameters
    ----------
    filepath : path
        Path to Shimadzu dataset.
    name : string (optional)
        Data name. Default is None (filename).
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    WrightTools.data.Data object
    """
    # check filepath ------------------------------------------------------------------------------
    if os.path.isfile(filepath):
        if verbose:
            print('found the file!')
    else:
        raise FileNotFoundError(filepath)
    # is the file suffix one that we expect?  warn if it is not!
    filesuffix = os.path.basename(filepath).split('.')[-1]
    if filesuffix != 'txt':
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'txt')
    # import data ---------------------------------------------------------------------------------
    # now import file as a local var--18 lines are just txt and thus discarded
    data = np.genfromtxt(filepath, skip_header=2, delimiter=',').T
    # construct data
    x_axis = Axis(data[0], 'nm', name='wm')
    signal = Channel(data[1], 'sig', file_idx=1, signed=False)
    data = Data([x_axis], [signal], source='Shimadzu', name=name)
    # return --------------------------------------------------------------------------------------
    return data
