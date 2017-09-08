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


def from_spcm(filepath, name=None, delimiter=',', format=None, verbose=True):
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
        w = wt_exceptions.WrongFileTypeWarning.warn(filepath, 'asc')
        warnings.warn(w)
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
    # now import file as a local var as comma-delimited .asc file
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
    x_axis = Axis(arr[0], 'ns', name='time')
    signal = Channel(arr[1], name='counts', signed=False)
    data = Data([x_axis], [signal], source='SPC_130', name=name, **headers)
    if verbose:
        print('data object created!')
    # return
    return data
