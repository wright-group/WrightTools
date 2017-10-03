"""Tensor 27."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

from ._data import Axis, Channel, Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_Tensor27']


# --- from function -------------------------------------------------------------------------------


def from_Tensor27(filepath, name=None, verbose=True):
    """Create a data object from a Tensor27 FTIR file.

    .. plot::

        >>> import WrightTools as wt
        >>> import matplotlib
        >>> from WrightTools import datasets
        >>> p = datasets.Tensor27.CuPCtS_powder_ATR
        >>> data = wt.data.from_Tensor27(p)
        >>> artist = wt.artists.mpl_1D(data)
        >>> artist.plot()
        >>> matplotlib.pyplot.xlim(1300,1700)
        >>> matplotlib.pyplot.ylim(-0.005,.02)

    Parameters
    ----------
    filepath : string
        Path to Tensor27 output file (.dpt).
    name : string (optional)
        Name to give to the created data object. If None, filename is used.
        Default is None.
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
    if filesuffix != 'dpt':
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'dpt')
    # import array
    arr = np.genfromtxt(filepath, skip_header=0).T
    # name
    if not name:
        name = os.path.basename(filepath)
    # construct data
    axis = Axis(arr[0], 'wn', name='w')
    signal = Channel(arr[1], name='absorbance', label='absorbance', signed=False)
    data = Data([axis], [signal], source='Tensor 27', name=name)
    # finish
    if verbose:
        print('data object successfully created from Tensor 27 file')
    return data
