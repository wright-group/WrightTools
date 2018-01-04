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


def from_shimadzu(filepath, name=None, parent=None, verbose=True):
    """Create a data object from Shimadzu .txt file.

    Parameters
    ----------
    filepath : string
        Path to .txt file.
    name : string (optional)
        Name to give to the created data object. If None, filename is used.
        Default is None.
    parent : WrightTools.Collection (optional)
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
    if not filepath.endswith('txt'):
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'txt')
    # parse name
    if not name:
        name = os.path.basename(filepath).split('.')[0]
    # create data
    kwargs = {'name': name, 'kind': 'Shimadzu', 'source': filepath}
    if parent is None:
        data = Data(**kwargs)
    else:
        data = parent.create_data(**kwargs)
    # array
    arr = np.genfromtxt(filepath, skip_header=2, delimiter=',').T
    # chew through all scans
    data.create_variable(name='energy', values=arr[0], units='nm')
    data.create_channel(name='signal', values=arr[1])
    data.transform(['energy'])
    # finish
    if verbose:
        print('data created at {0}'.format(data.fullpath))
        print('  range: {0} to {1} (nm)'.format(data.energy[0], data.energy[-1]))
        print('  size: {0}'.format(data.size))
    return data












if False:

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
