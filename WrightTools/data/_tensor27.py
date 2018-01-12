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


def from_Tensor27(filepath, name=None, collection=None, verbose=True):
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
    filesuffix = os.path.basename(filepath).split('.')[-1]
    if filesuffix != 'dpt':
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'dpt')
    # parse name
    if name is None:
        name = os.path.basename(filepath).split('.')[0]
    # create data
    kwargs = {'name': name, 'kind': 'Tensor27', 'source': filepath}
    if collection:
        data = collection.create_data(**kwargs)
    else:
        data = Data(**kwargs)
    # array
    arr = np.genfromtxt(filepath, skip_header=0).T
    # construct data
    data.create_axis(name='wm', points=arr[0], units='wn')
    data.create_channel(name='signal', values=arr[1])
    # finish
    if verbose:
        print('data created at {0}'.format(data.fullpath))
        print('  kind: {0}'.format(data.kind))
        print('  range: {0} to {1} (wn)'.format(data.wm[0], data.wm[-1]))
        print('  size: {0}'.format(data.size))
    return data
