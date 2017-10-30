"""SPCM."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import collections
import warnings

import numpy as np

from ._data import Axis, Channel, Data
from .. import exceptions as wt_exceptions


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_spcm']


# --- from function -------------------------------------------------------------------------------


def from_spcm(filepath, name=None, delimiter=',', format=None, collection=None, verbose=True):
    """Create a data object from Becker & Hickl `spcm`__ software.

    __ http://www.becker-hickl.com/software/spcm.htm

    Parameters
    ----------
    filepath : string
        Path to SPC-130 .asc file.
    name : string (optional)
        Name to give to the created data object. If None, filename is used.
        Default is None.
    delimiter : string (optional)
        The string used to separate values. Default is ','.
    format : {'ascii'} (optional)
        Force file to be interpreted as a specific format. Default is None
        (autorecognized).
    collection : WrightTools.Collection (optional)
        Collection to place new data object within. Default is None.
    verbose : boolean (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    WrightTools.data.Data object
    """
    # check filepath
    if not os.path.isfile(filepath):
        raise wt_exceptions.FileNotFound(path=filepath)
    if not filepath.endswith('asc'):
        wt_exceptions.WrongFileTypeWarning.warn(filepath, 'asc')
    # parse name
    if not name:
        name = os.path.basename('filepath').split('.')[0]
    # create data
    kwargs = {'name': name, 'kind': 'spcm', 'source': filepath}
    if collection:
        data = collection.create_data(**kwargs)
    else:
        data = Data(**kwargs)
    # create headers dictionary
    headers = collections.OrderedDict()
    with open(filepath) as f:
        while True:
            line = f.readline().strip()
            if len(line) == 0:
                break
            else:
                key, value = line.split(':', 1)
                if key.strip() == 'Revision':
                    headers['resolution'] = int(value.strip(' bits ADC'))
                else:
                    headers[key.strip()] = value.strip()
    # import data
    arr = np.genfromtxt(filepath, skip_header=(len(headers) + 2), skip_footer=1,
                        delimiter=delimiter).T
    # unexpected delimiter handler
    if np.any(np.isnan(arr)):
        # delimiter warning dictionary
        delim_dict = {',': 'comma',
                      ' ': 'space',
                      '\t': 'tab',
                      ';': 'semicolon',
                      ':': 'colon'}
        warnings.warn('file is not %s-delimited! Trying other delimiters.' % delim_dict[delimiter])
        for delimiter in delim_dict.keys():
            arr = np.genfromtxt(filepath, skip_header=len(headers) + 2, skip_footer=1,
                                delimiter=delimiter).T
            if not np.any(np.isnan(arr)):
                print('file is %s-delimited.' % delim_dict[delimiter])
                break
        else:
            error = '''Unable to load data file.
                       Data object not created!
                       Please check that your file is formatted properly.'''
            raise RuntimeError(error)
    # construct data
    data.create_axis(name='time', points=arr[0], units='ns')
    data.create_channel(name='counts', values=arr[1])
    if verbose:
        print('data created at {0}'.format(data.fullpath))
        print('  kind: {0}'.format(data.kind))
        print('  range: {0} to {1} (ns)'.format(data.time[0], data.time[-1]))
        print('  size: {0}'.format(data.size))
    # return
    return data
