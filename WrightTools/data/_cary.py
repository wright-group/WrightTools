"""Cary."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

from ._data import Axis, Channel, Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_Cary50']


# --- from function -------------------------------------------------------------------------------


def from_Cary50(filepath, verbose=True):
    """Create a data object from a Cary 50 UV VIS absorbance file.

    .. plot::

        >>> import WrightTools as wt
        >>> from WrightTools import datasets
        >>> p = datasets.Cary50.CuPCtS_H2O_vis
        >>> data = wt.data.from_Cary50(p)[0]
        >>> artist = wt.artists.mpl_1D(data)
        >>> artist.plot()

    Parameters
    ----------
    filepath : string
        Path to Cary50 output file (.csv).
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
    if filesuffix != 'csv':
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'csv')
    # import array
    lines = []
    with open(filepath, 'r') as f:
        header = f.readline()
        while True:
            line = f.readline()
            if line == '\n' or line == '':
                break
            else:
                clean = line[:-2]  # lines end with ',/n'
                lines.append(np.fromstring(clean, sep=','))
    header = header.split(',')
    lines = [i for i in lines if len(i) > 0]
    arr = np.array(lines).T
    # chew through all scans
    datas = []
    indicies = np.arange(len(header) // 2) * 2
    for i in indicies:
        axis = Axis(arr[i], 'nm', name='wm')
        signal = Channel(arr[i + 1], name='absorbance', label='absorbance', signed=False)
        data = Data([axis], [signal], source='Cary 50', name=header[i])
        datas.append(data)
    # finish
    if verbose:
        print('{0} data objects successfully created from Cary 50 file:'.format(len(indicies)))
        for i, data in enumerate(datas):
            print('  {0}: {1}'.format(i, data.name))
    return datas
