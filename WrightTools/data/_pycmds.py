"""PyCMDS."""


# --- import --------------------------------------------------------------------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import warnings

import numpy as np

from scipy.interpolate import griddata

from ._data import Axis, Channel, Data
from .. import kit as wt_kit
from .. import units as wt_units


# --- define --------------------------------------------------------------------------------------


__all__ = ['from_PyCMDS']


# --- from function -------------------------------------------------------------------------------


def from_PyCMDS(filepath, name=None,
                shots_processing_module='mean_and_std', verbose=True):
    """Create a data object from a single PyCMDS output file.

    Parameters
    ----------
    filepath : str
        The file to load. Can accept .data, .fit, or .shots files.
    name : str or None (optional)
        The name to be applied to the new data object. If None, name is read
        from file.
    shots_processing_module : str (optional)
        The module used to process .shots files, if provided. Must be the name
        of a module in the shots_processing directory.
    verbose : bool (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    data
        A Data instance.
    """
    # header
    headers = wt_kit.read_headers(filepath)
    # name
    if name is None:  # name not given in method arguments
        data_name = headers['data name']
    else:
        data_name = name
    if data_name == '':  # name not given in PyCMDS
        data_name = headers['data origin']
    # array
    arr = np.genfromtxt(filepath).T
    # get axes
    axes = []
    for name, identity, units in zip(headers['axis names'],
                                     headers['axis identities'],
                                     headers['axis units']):
        points = np.array(headers[name + ' points'])
        # label seed (temporary implementation)
        try:
            index = headers['name'].index(name)
            label = headers['label'][index]
            label_seed = [label]
        except ValueError:
            label_seed = ['']
        # create axis
        kwargs = {'identity': identity}
        if 'D' in identity:
            kwargs['centers'] = headers[name + ' centers']
        axis = Axis(points, units, name=name, label_seed=label_seed, **kwargs)
        axes.append(axis)
    # get indicies arrays
    indicies = arr[:len(axes)].T
    indicies = indicies.astype('int64')
    # get interpolation toggles
    if 'axis interpolate' in headers.keys():
        interpolate_toggles = headers['axis interpolate']
    else:
        # old data files may not have interpolate toggles in headers
        # assume no interpolation, unless the axis is the array detector
        interpolate_toggles = [True if name == 'wa' else False for name in headers['axis names']]
    # get assorted remaining things
    shape = tuple([a.points.size for a in axes])
    tols = [wt_kit.closest_pair(a.points, give='distance') / 2. for a in axes]
    # prepare points for interpolation
    points_dict = collections.OrderedDict()
    for i, axis in enumerate(axes):
        # TODO: math and proper full recognition...
        axis_col_name = [name for name in headers['name'][::-1] if name in axis.identity][0]
        axis_index = headers['name'].index(axis_col_name)
        # shape array acording to recorded coordinates
        points = np.full(shape, np.nan)
        for j, idx in enumerate(indicies):
            lis = arr[axis_index]
            points[tuple(idx)] = lis[j]
        # convert array
        points = wt_units.converter(points, headers['units'][axis_index], axis.units)
        # take case of scan about center
        if axis.identity[0] == 'D':
            # transpose so this axis is first
            transpose_order = list(range(len(axes)))
            transpose_order.insert(0, transpose_order.pop(i))
            points = points.transpose(transpose_order)
            # subtract out centers
            centers = np.array(headers[axis.name + ' centers'])
            points -= centers
            # transpose out
            transpose_order = list(range(len(axes)))
            transpose_order.insert(i, transpose_order.pop(0))
            points = points.transpose(transpose_order)
        points = points.flatten()
        points_dict[axis.name] = points
        # check, coerce non-interpolated axes
        if not interpolate_toggles[i]:
            for j, idx in enumerate(indicies):
                actual = points[j]
                expected = axis.points[idx[i]]
                if abs(actual - expected) > tols[i]:
                    warnings.warn('at least one point exceded tolerance ' +
                                  'in axis {}'.format(axis.name))
                points[j] = expected
    all_points = tuple(points_dict.values())
    # prepare values for interpolation
    values_dict = collections.OrderedDict()
    for i, kind in enumerate(headers['kind']):
        if kind == 'channel':
            values_dict[headers['name'][i]] = arr[i]
    # create grid to interpolate onto
    if len(axes) == 1:
        meshgrid = tuple([axes[0].points])
    else:
        meshgrid = tuple(np.meshgrid(*[a.points for a in axes], indexing='ij'))
    if any(interpolate_toggles):
        # create channels through linear interpolation
        channels = []
        for i in range(len(arr)):
            if headers['kind'][i] == 'channel':
                # unpack
                units = headers['units'][i]
                signed = headers['channel signed'][len(channels)]
                name = headers['name'][i]
                label = headers['label'][i]
                # interpolate
                values = values_dict[name]
                zi = griddata(all_points, values, meshgrid, rescale=True,
                              method='linear', fill_value=np.nan)
                # assemble
                channel = Channel(zi, units=units, signed=signed, name=name, label=label)
                channels.append(channel)
    else:
        # if none of the axes are interpolated onto,
        # simply fill zis based on recorded axis index
        num_channels = headers['kind'].count('channel')
        channel_indicies = [i for i, kind in enumerate(headers['kind']) if kind == 'channel']
        zis = [np.full(shape, np.nan) for _ in range(num_channels)]
        # iterate through each row of the array, filling zis
        for i in range(len(arr[0])):
            idx = tuple(indicies[i])  # yes, this MUST be a tuple >:(
            for zi_index, arr_index in enumerate(channel_indicies):
                zis[zi_index][idx] = arr[arr_index, i]
        # assemble channels
        channels = []
        for zi_index, arr_index in zip(range(len(zis)), channel_indicies):
            zi = zis[zi_index]
            units = headers['units'][arr_index]
            signed = headers['channel signed'][zi_index]
            name = headers['name'][arr_index]
            label = headers['label'][arr_index]
            channel = Channel(zi, units=units, signed=signed, name=name, label=label)
            channels.append(channel)
    # get constants
    constants = []
    for name, identity in zip(headers['constant names'], headers['constant identities']):
        # TODO: handle PyCMDS constants
        pass
    # create data object
    data = Data(axes, channels, constants, name=data_name, source=filepath)
    # return
    if verbose:
        print('data object succesfully created')
        print('  axes:', data.axis_names)
        print('  shape:', data.shape)
    return data
