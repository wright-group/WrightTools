"""PyCMDS."""


# --- import --------------------------------------------------------------------------------------


import collections
import warnings

import numpy as np

from scipy.interpolate import griddata

import tidy_headers

from ._data import Data
from .. import kit as wt_kit
from .. import units as wt_units


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_PyCMDS']


# --- from function -------------------------------------------------------------------------------


def from_PyCMDS(filepath, name=None, parent=None, verbose=True):
    """Create a data object from a single PyCMDS output file.

    Parameters
    ----------
    filepath : str
        The file to load. Can accept .data, .fit, or .shots files.
    name : str or None (optional)
        The name to be applied to the new data object. If None, name is read
        from file.
    parent : WrightTools.Collection (optional)
        Collection to place new data object within. Default is None.
    verbose : bool (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        A Data instance.
    """
    # header
    headers = tidy_headers.read(filepath)
    # name
    if name is None:  # name not given in method arguments
        data_name = headers['data name']
    else:
        data_name = name
    if data_name == '':  # name not given in PyCMDS
        data_name = headers['data origin']
    # create data object
    kwargs = {'name': data_name, 'kind': 'PyCMDS', 'source': filepath,
              'created': headers['file created'],
              }
    if parent is not None:
        data = parent.create_data(**kwargs)
    else:
        data = Data(**kwargs)
    # array
    arr = np.genfromtxt(filepath).T
    # get axes and scanned variables
    axes = []
    for name, identity, units in zip(headers['axis names'],
                                     headers['axis identities'],
                                     headers['axis units']):
        # points and centers
        points = np.array(headers[name + ' points'])
        if name + ' centers' in headers.keys():
            centers = headers[name + ' centers']
        else:
            centers = None
        # create
        axis = {'points': points, 'units': units, 'name': name, 'identity': identity,
                'centers': centers}
        axes.append(axis)
    shape = tuple([a['points'].size for a in axes])
    # get assorted remaining things
    # variables and channels
    for index, kind, name in zip(range(len(arr)), headers['kind'], headers['name']):
        if name == 'time':
            values = np.reshape(arr[index], shape)
            data.create_variable(name='labtime', values=values)
        if kind == 'hardware':
            # sadly, recorded tolerances are not reliable
            # so a bit of hard-coded hacking is needed
            # if this ends up being too fragile, we might have to use the points arrays
            # ---Blaise 2018-01-09
            values = np.reshape(arr[index], shape)
            if 'w' in name and name.startswith(tuple(data.variable_names)):
                inherited_shape = data[name.split('_')[0]].shape
                for i, s in enumerate(inherited_shape):
                    if s == 1:
                        values = np.mean(values, axis=i)
                        values = np.expand_dims(values, i)
            else:
                tolerance = headers['tolerance'][index]
                for i in range(len(shape)):
                    if tolerance is None:
                        break
                    if 'd' in name:
                        tolerance = 3.
                    if 'zero' in name:
                        tolerance = 1e-10
                    mean = np.mean(values, axis=i)
                    mean = np.expand_dims(mean, i)
                    if np.allclose(mean, values, atol=tolerance):
                        values = mean
            units = headers['units'][index]
            label = headers['label'][index]
            data.create_variable(name, values=values, units=units, label=label)
        if kind == 'channel':
            values = np.reshape(arr[index], shape)
            data.create_channel(name=name, values=values, shape=values.shape)
    # axes
    for a in axes:
        expression = a['identity']
        if expression.startswith('D'):
            expression = expression[1:]
        expression.replace('=D', '=')
        a['expression'] = expression
    data.transform(*[a['expression'] for a in axes])
    # return
    if verbose:
        print('data created at {0}'.format(data.fullpath))
        print('  axes: {0}'.format(data.axis_names))
        print('  shape: {0}'.format(data.shape))
    return data
