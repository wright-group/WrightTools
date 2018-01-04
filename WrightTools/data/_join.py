"""Join multiple data objects together."""


# --- import --------------------------------------------------------------------------------------


import collections

import numpy as np

from .. import units as wt_units
from ._data import Data


# --- define --------------------------------------------------------------------------------------


__all__ = ['join']


# --- functions -----------------------------------------------------------------------------------


def join(datas, method='first', parent=None, verbose=True, **kwargs):
    """Join a list of data objects together.

    For now datas must have identical dimensionalities (order and identity).

    Parameters
    ----------
    datas : list of data
        The list of data objects to join together.
    method : {'first', 'sum', 'max', 'min', 'mean'} (optional)
        The method for how overlapping points get treated. Default is first,
        meaning that the data object that appears first in datas will take
        precedence.
    verbose : bool (optional)
        Toggle talkback. Default is True.

    Returns
    -------
    WrightTools.data.Data
        A new Data instance.
    """
    # TODO: fill value
    print('data.join! ----------------------------------------------------')
    datas = list(datas)
    # check if variables are valid
    axis_expressions = datas[0].axis_expressions
    variable_units = []
    variable_names = []
    for a in datas[0].axes:
        for v in a.variables:
            variable_names.append(a.natural_name)
            variable_units.append(a.units)
    # TODO: check if all other datas have the same variable names
    # check if channels are valid
    # TODO: this is a hack
    channel_units = []
    channel_names = []
    for c in datas[0].channels:
        channel_names.append(c.natural_name)
        channel_units.append(c.units)
    # get output data
    out = Data(name='join', parent=parent)
    # variables
    vs = collections.OrderedDict()
    for name, units in zip(variable_names, variable_units):
        values = set()
        for data in datas:
            v = data[name]
            arr = v[:]  # TODO: units arr = wt_units.converter(v[:], v.units, units)
            for _, x in np.ndenumerate(arr):
                values.add(x)
        values = np.array(sorted(values))
        vs[name] = {'values': values, 'units': units}
    # TODO: the following should become a new from method
    def from_dict(d):
        ndim = len(d)
        i = 0
        for k, v in d.items():
            values = v['values']
            units = v['units']
            shape = [1] * ndim
            shape[i] = values.size
            print(shape, values.size)
            values.shape = tuple(shape)
            out.create_variable(name=k, values=values, units=units)
            i += 1
    from_dict(vs)
    # channels
    for channel_name, units in zip(channel_names, channel_units):
        new = out.create_channel(name=channel_name, units=units)
        for data in datas:
            old = data[channel_name]
            old /= old.max()
            for old_idx, value in np.ndenumerate(old):
                new_idx = []
                for variable_name in out.variable_names:
                    p = data[variable_name][old_idx]
                    arr = out[variable_name][:]
                    i = np.argmin(np.abs(arr - p))
                    new_idx.append(i)
                new[tuple(new_idx)] = old[old_idx]
    # axes
    out.transform(axis_expressions)
    # finish
    if verbose and False:
        print(len(datas), 'datas joined to create new data:')
        print('  axes:')
        for axis in out.axes:
            points = axis[:]
            print('    {0} : {1} points from {2} to {3} {4}'.format(
                axis.name, points.size, min(points), max(points), axis.units))
        print('  channels:')
        for channel in out.channels:
            percent_nan = np.around(100. * (np.isnan(channel[:]).sum() /
                                            float(channel.size)), decimals=2)
            print('    {0} : {1} to {2} ({3}% NaN)'.format(
                channel.name, channel.min(), channel.max(), percent_nan))
    return out
